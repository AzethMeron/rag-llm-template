"""The hybrid retriever: lexical + dense fusion, optional cross-encoder reranking, MMR diversity.

Composes two first-stage :class:`~ragkit.core.ports.Retriever` arms — lexical (exact-term) and
dense (semantic) — behind one Retriever, rather than forking either. Per query:

1. Each arm retrieves its own ``candidate_pool``, gated by its own calibrated floor (the arms'
   scores are on different scales, so each keeps its own floor).
2. The two rankings are fused with Reciprocal Rank Fusion — a candidate both arms agree on outranks
   one only one arm found, with no need to calibrate incomparable scores.
3. If a reranker is configured, the fused pool is re-scored by the cross-encoder (which returns
   ``[0, 1]`` already — see :mod:`.rerank`). Without one, the fused score is divided by
   ``best_possible_rrf`` to a
   fixed, query-independent ``[0, 1]`` scale — so a floor keeps a stable meaning rather than
   becoming a per-query relative rank.
4. Candidates below the caller's ``min_score`` (on that final relevance) are dropped.
5. The survivors are diversified with MMR so a top-k of near-duplicates does not crowd out a varied
   set, using trigram similarity over the texts (no extra embedding round-trip).

**Fails loud.** A dead reranker propagates ``RerankError`` — there is no silent fallback to the
unreranked order, because a misbehaving reranker demands attention, not a quietly degraded answer.

**Measured caution, carried from the reference project:** equal-weight RRF fusion blends its arms
rather than taking the better one, and on two real corpora it scored *below* the better single arm
unless a reranker was also attached. Measure on your own corpus before adopting it as a default;
the plain embedding retriever is often the better baseline.
"""
from __future__ import annotations

from collections.abc import Sequence

from ragkit.core.errors import RagkitError
from ragkit.core.ports import Retrieved, Retriever

from .fusion import best_possible_rrf, mmr_order, rrf
from .rerank import RerankClient
from .retrievers import trigram_similarity

_ARM_COUNT = 2


class HybridError(RagkitError):
    """The hybrid retriever was misconfigured."""


class HybridRetriever:
    """Lexical + dense fusion, optional rerank, MMR diversity — one Retriever."""

    def __init__(self, lexical: Retriever, dense: Retriever, *,
                 reranker: RerankClient | None = None, candidate_pool: int = 40,
                 mmr_lambda: float = 0.7, lexical_min_score: float = 0.30,
                 dense_min_score: float = 0.55) -> None:
        if candidate_pool < 1:
            raise HybridError(f"candidate_pool must be >= 1, got {candidate_pool}")
        if not 0.0 <= mmr_lambda <= 1.0:
            raise HybridError(f"mmr_lambda must be in [0, 1], got {mmr_lambda}")
        self._lexical = lexical
        self._dense = dense
        self._reranker = reranker
        self._pool = candidate_pool
        self._mmr_lambda = mmr_lambda
        self._lexical_min = lexical_min_score
        self._dense_min = dense_min_score

    def retrieve(self, query: str, *, k: int, min_score: float = 0.0) -> tuple[Retrieved, ...]:
        if k <= 0:
            return ()
        lex_hits = self._lexical.retrieve(query, k=self._pool, min_score=self._lexical_min)
        dense_hits = self._dense.retrieve(query, k=self._pool, min_score=self._dense_min)
        by_id: dict[str, Retrieved] = {}
        for hit in (*lex_hits, *dense_hits):
            by_id.setdefault(hit.chunk_id, hit)
        fused = rrf([[h.chunk_id for h in lex_hits], [h.chunk_id for h in dense_hits]])
        if not fused:
            return ()
        confidence = self._confidence(lex_hits, dense_hits)
        candidates = sorted(fused, key=lambda cid: (-fused[cid], -confidence.get(cid, 0.0)))
        candidates = candidates[:self._pool]
        relevance = self._relevance(query, candidates, fused, by_id)
        survivors = [cid for cid in candidates if relevance[cid] >= min_score]
        if not survivors:
            return ()

        def similarity(a: str, b: str) -> float:
            return trigram_similarity(by_id[a].text, by_id[b].text)

        ordered = mmr_order(survivors, relevance, similarity, lambda_=self._mmr_lambda, k=k)
        return tuple(Retrieved(chunk_id=cid, text=by_id[cid].text, score=relevance[cid],
                               meta=by_id[cid].meta) for cid in ordered)

    def _confidence(self, lex_hits: Sequence[Retrieved],
                    dense_hits: Sequence[Retrieved]) -> dict[str, float]:
        """How far above its own floor each arm placed a candidate, best arm winning — a fair
        secondary signal for separating candidates the fused rank ties, derived from scores already
        computed. Ties are not rare: when the arms disagree outright both candidates score exactly
        ``1/(k+1)``, and a plain sort would then settle a retrieval-quality question by argument
        order, handing every disagreement to the same arm."""
        confidence: dict[str, float] = {}
        for hits, floor in ((lex_hits, self._lexical_min), (dense_hits, self._dense_min)):
            headroom = max(1e-9, 1.0 - floor)
            for hit in hits:
                scaled = min(1.0, max(0.0, (hit.score - floor) / headroom))
                if scaled > confidence.get(hit.chunk_id, -1.0):
                    confidence[hit.chunk_id] = scaled
        return confidence

    def _relevance(self, query: str, candidates: list[str], fused: dict[str, float],
                   by_id: dict[str, Retrieved]) -> dict[str, float]:
        """Final ``[0, 1]`` relevance per candidate, on a scale that does not shift per query. With
        a reranker: its relevance verbatim — the ``RerankClient`` normalises to ``[0, 1]`` itself,
        knowing whether its endpoint speaks logits or unit scores (squashing here as well mapped an
        already-``[0, 1]`` Jina/Cohere score into ``[0.5, 0.731]``, leaving ranking intact but every
        floor meaningless). Without one: the fused score over ``best_possible_rrf`` — a fixed
        denominator, so a floor means the same thing for every query, and deliberately not min-max
        over the query's own pool (which would score the best of three terrible candidates 1.0 and
        the worst of three excellent ones 0.0)."""
        if self._reranker is not None:
            scored = self._reranker.rerank(query, [by_id[cid].text for cid in candidates])
            return {candidates[index]: score for index, score in scored}
        ceiling = best_possible_rrf(_ARM_COUNT)
        return {cid: min(1.0, fused[cid] / ceiling) for cid in candidates}

    def close(self) -> None:
        if self._reranker is not None:
            self._reranker.close()
