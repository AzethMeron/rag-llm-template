"""First-stage retrievers over the storage indexes: lexical (BM25) and dense (embeddings).

Each adapts a storage index to the :class:`~ragkit.core.ports.Retriever` port — ``retrieve(query,
k, min_score) -> tuple[Retrieved, ...]``, best-first, deterministic, safe to share across a run's
workers. The index returns ``(chunk_id, score)``; a ``resolve`` callable maps an id back to its
text. A hit whose score is below ``min_score`` is dropped.

An id the search index returns but ``resolve`` cannot map to text is handled by *where* the two
come from, never silently dropped without a trace:

* **lexical** — the search index and the text store are the *same* co-located
  :class:`~ragkit.core.ports.PairingStore`, whose contract guarantees a hit and its text cannot
  drift apart, so an unresolvable hit is an internal invariant violation (a corrupt store) and is
  raised as a :class:`RetrieverError`;
* **dense** — the vector index and the pairing store are separately written and can legitimately
  drift between reconciles, so an unresolvable hit is dropped, but with a ``logger.warning`` naming
  the count so a badly-stale index cannot silently halve every result set.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

from ragkit.core.errors import RagkitError
from ragkit.core.ports import Retrieved, SearchIndex, VectorIndex

from .embedding import EmbeddingClient

logger = logging.getLogger(__name__)

# Maps a chunk id to its text, or None if the text store no longer has it.
Resolver = Callable[[str], str | None]

# Optionally maps a chunk id to its metadata (for filtering / display); defaults to empty.
MetaResolver = Callable[[str], Mapping[str, Any]]


class RetrieverError(RagkitError):
    """A retriever hit an inconsistency it cannot paper over — e.g. a co-located store returned a
    search hit whose text it cannot resolve, which its port contract says is impossible."""


def _no_meta(_chunk_id: str) -> Mapping[str, Any]:
    return {}


def _collect(scored: list[tuple[str, float]], resolve: Resolver, meta: MetaResolver,
             min_score: float, *, arm: str, colocated: bool) -> tuple[Retrieved, ...]:
    hits: list[Retrieved] = []
    dropped = 0
    for chunk_id, score in scored:
        if score < min_score:
            continue
        text = resolve(chunk_id)
        if text is None:
            if colocated:
                raise RetrieverError(
                    f"{arm} retriever: the search index returned id {chunk_id!r} that the "
                    f"co-located store cannot resolve to text — the pairing store is internally "
                    f"inconsistent (its rows and search index have drifted apart)")
            dropped += 1
            continue
        hits.append(Retrieved(chunk_id=chunk_id, text=text, score=score, meta=meta(chunk_id)))
    if dropped:
        logger.warning(
            "%s retriever: dropped %d of %d hit(s) whose text the store no longer has; the vector "
            "index may be stale versus the pairing store — run reconcile",
            arm, dropped, len(scored))
    return tuple(hits)


class LexicalRetriever:
    """A :class:`~ragkit.core.ports.Retriever` over a :class:`~ragkit.core.ports.SearchIndex`
    (BM25). Strong on names, ids, and recurring terminology. Depends only on the narrow
    ``SearchIndex`` read side (never indexing/deleting through a retriever), which is exactly what
    :class:`~ragkit.core.ports.PairingStore` exposes."""

    def __init__(self, index: SearchIndex, resolve: Resolver, *,
                 meta: MetaResolver = _no_meta, default_k: int = 10) -> None:
        self._index = index
        self._resolve = resolve
        self._meta = meta
        self._default_k = default_k

    def retrieve(self, query: str, *, k: int, min_score: float = 0.0) -> tuple[Retrieved, ...]:
        if k <= 0:
            return ()
        return _collect(self._index.search(query, k=k), self._resolve, self._meta, min_score,
                        arm="lexical", colocated=True)


class DenseRetriever:
    """A :class:`~ragkit.core.ports.Retriever` over a :class:`~ragkit.core.ports.VectorIndex` plus
    an :class:`~ragkit.retrieve.embedding.EmbeddingClient`. Strong on paraphrase and cross-script
    matches the lexical retriever is blind to."""

    def __init__(self, index: VectorIndex, embedder: EmbeddingClient, resolve: Resolver, *,
                 meta: MetaResolver = _no_meta) -> None:
        self._index = index
        self._embedder = embedder
        self._resolve = resolve
        self._meta = meta

    def retrieve(self, query: str, *, k: int, min_score: float = 0.0) -> tuple[Retrieved, ...]:
        if k <= 0:
            return ()
        vector = self._embedder.embed_one(query)
        scored = self._index.search(vector, k=k)
        return _collect(scored, self._resolve, self._meta, min_score,
                        arm="dense", colocated=False)


def _trigrams(text: str) -> set[str]:
    normalized = text.casefold().strip()
    if len(normalized) < 3:
        return {normalized} if normalized else set()
    return {normalized[i:i + 3] for i in range(len(normalized) - 2)}


def trigram_similarity(a: str, b: str) -> float:
    """Jaccard similarity over character trigrams: a stdlib, embedding-free measure of how alike two
    texts look, used to diversify a shortlist without an embedding round-trip per candidate."""
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
