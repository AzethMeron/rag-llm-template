"""Retrieval evaluation: rank-quality metrics, and a harness that refuses a circular configuration.

The metrics are the standard ones for a ranked list against a set of known-relevant ids: recall@k,
reciprocal rank (→ MRR), average precision (→ MAP), and NDCG@k with binary relevance. Each is a pure
function of ``(ranked_ids, relevant_ids)`` so it is testable in isolation and independent of how the
ranking was produced.

The load-bearing part is :func:`evaluate_retrieval`'s **circularity guard**. The `.audit/` record
this milestone opens documents the failure it prevents: a metric once defined its ground truth as
"what one retrieval arm returns", so that arm scored ``1.000`` by construction — and it replicated
cleanly across two corpora, which made it *persuasive*, not *true*. The rule that falls out is: a
metric must be independent of the thing it ranks. Here that is mechanical — the ground-truth
:class:`Qrels` carry a ``source`` label, and the harness **refuses to run** if that source is one of
the systems under evaluation, rather than reporting a number that is true by construction.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ragkit.core.errors import RagkitError
from ragkit.core.ports import Retriever


class CircularEvaluationError(RagkitError):
    """The evaluation is configured so a result would be true by construction."""


@dataclass(frozen=True, slots=True)
class Qrels:
    """Ground-truth relevance: for each query id, the set of relevant document ids, plus a
    ``source`` label naming where the judgments came from (a gold dataset, an annotator). The label
    is not decoration: the harness checks it is not one of the systems being ranked."""

    relevant: Mapping[str, frozenset[str]]
    source: str

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("Qrels needs a non-empty 'source' naming where the judgments came "
                             "from; an unlabelled ground truth cannot be checked for independence")


def recall_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """Fraction of the relevant ids that appear in the top ``k``. Undefined with no relevant ids,
    which the harness excludes before calling this."""
    top = set(ranked[:k])
    return len(top & relevant) / len(relevant)


def hit_rate_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """1.0 if at least one relevant id appears in the top ``k``, else 0.0 -- a binary per-query
    hit/miss, unlike ``recall_at_k``'s fraction of *all* relevant ids captured. This is the "top-k
    accuracy" metric reported by PolQA (Rybak et al., 2022) and similar OpenQA retrieval papers;
    kept alongside recall_at_k/mrr/ndcg so a recipe can report a literature-comparable number
    instead of only this framework's own recall/MRR/NDCG convention."""
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: frozenset[str]) -> float:
    """``1 / rank`` of the first relevant id (rank counted from 1), or 0 if none is retrieved."""
    for index, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / index
    return 0.0


def average_precision(ranked: Sequence[str], relevant: frozenset[str]) -> float:
    """Mean of the precision values taken at each rank where a relevant id is hit, normalised by
    the number of relevant ids — the per-query term of MAP."""
    hits = 0
    summed = 0.0
    for index, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            hits += 1
            summed += hits / index
    return summed / len(relevant)


def ndcg_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """NDCG@k with binary relevance: DCG of the top ``k`` over the ideal DCG (every relevant id
    ranked first). 1.0 when the top ``k`` hold as many relevant ids, as high as possible."""
    dcg = sum(1.0 / math.log2(index + 1)
              for index, doc_id in enumerate(ranked[:k], start=1) if doc_id in relevant)
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


@dataclass(frozen=True, slots=True)
class RetrievalScores:
    """Metrics for one system, averaged over the evaluated queries."""

    k: int
    queries: int
    recall_at_k: float
    hit_rate_at_k: float
    mrr: float
    map: float
    ndcg_at_k: float


def evaluate_retrieval(systems: Mapping[str, Retriever], queries: Mapping[str, str], qrels: Qrels,
                       *, k: int = 10) -> dict[str, RetrievalScores]:
    """Score each named retriever over ``queries`` against ``qrels``, returning per-system metrics.

    Refuses, before doing any work, a configuration that would be true by construction: ground truth
    produced by a system under evaluation (:class:`CircularEvaluationError`). Also refuses a query
    with no ground-truth relevant ids, since scoring it would silently invent a 0 (or a 1) for a
    question the gold set never answered.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not systems:
        raise ValueError("no systems to evaluate")
    if qrels.source in systems:
        raise CircularEvaluationError(
            f"the ground truth's source {qrels.source!r} is also a system under evaluation, so it "
            f"would score perfectly by construction. Ground truth must be independent of the "
            f"systems it ranks; supply gold judgments from a source that is not being evaluated.")
    missing = [qid for qid in queries if not qrels.relevant.get(qid)]
    if missing:
        raise CircularEvaluationError(
            f"{len(missing)} quer(y/ies) have no ground-truth relevant ids "
            f"(e.g. {sorted(missing)[:3]}); scoring them would report a made-up number. Provide "
            f"judgments for every evaluated query, or drop it from the query set.")

    scores: dict[str, RetrievalScores] = {}
    for name, retriever in systems.items():
        recalls, hits, rrs, aps, ndcgs = [], [], [], [], []
        for qid, text in queries.items():
            relevant = qrels.relevant[qid]
            ranked = [hit.chunk_id for hit in retriever.retrieve(text, k=k)]
            recalls.append(recall_at_k(ranked, relevant, k))
            hits.append(hit_rate_at_k(ranked, relevant, k))
            rrs.append(reciprocal_rank(ranked, relevant))
            aps.append(average_precision(ranked, relevant))
            ndcgs.append(ndcg_at_k(ranked, relevant, k))
        scores[name] = RetrievalScores(
            k=k, queries=len(queries), recall_at_k=_mean(recalls), hit_rate_at_k=_mean(hits),
            mrr=_mean(rrs), map=_mean(aps), ndcg_at_k=_mean(ndcgs))
    return scores


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
