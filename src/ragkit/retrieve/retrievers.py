"""First-stage retrievers over the storage indexes: lexical (BM25) and dense (embeddings).

Each adapts a storage index to the :class:`~ragkit.core.ports.Retriever` port — ``retrieve(query,
k, min_score) -> tuple[Retrieved, ...]``, best-first, deterministic, safe to share across a run's
workers. The index returns ``(chunk_id, score)``; a ``resolve`` callable maps an id back to its
text (returning ``None`` for an id the index knows but the text store has lost — dropped rather
than surfaced as an empty result). A hit whose score is below ``min_score`` is dropped.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ragkit.core.ports import LexicalIndex, Retrieved, VectorIndex

from .embedding import EmbeddingClient

# Maps a chunk id to its text, or None if the text store no longer has it.
Resolver = Callable[[str], str | None]

# Optionally maps a chunk id to its metadata (for filtering / display); defaults to empty.
MetaResolver = Callable[[str], Mapping[str, Any]]


def _no_meta(_chunk_id: str) -> Mapping[str, Any]:
    return {}


def _collect(scored: list[tuple[str, float]], resolve: Resolver, meta: MetaResolver,
             min_score: float) -> tuple[Retrieved, ...]:
    hits: list[Retrieved] = []
    for chunk_id, score in scored:
        if score < min_score:
            continue
        text = resolve(chunk_id)
        if text is None:
            continue
        hits.append(Retrieved(chunk_id=chunk_id, text=text, score=score, meta=meta(chunk_id)))
    return tuple(hits)


class LexicalRetriever:
    """A :class:`~ragkit.core.ports.Retriever` over a :class:`~ragkit.core.ports.LexicalIndex`
    (BM25). Strong on names, ids, and recurring terminology."""

    def __init__(self, index: LexicalIndex, resolve: Resolver, *,
                 meta: MetaResolver = _no_meta, default_k: int = 10) -> None:
        self._index = index
        self._resolve = resolve
        self._meta = meta
        self._default_k = default_k

    def retrieve(self, query: str, *, k: int, min_score: float = 0.0) -> tuple[Retrieved, ...]:
        if k <= 0:
            return ()
        return _collect(self._index.search(query, k=k), self._resolve, self._meta, min_score)


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
        return _collect(scored, self._resolve, self._meta, min_score)


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
