"""Build a retrieval corpus: index texts into a lexical and/or vector store, and hand back
retrievers over them.

This is the ingest end of retrieval. Each item has a stable id, an **index text** (what matching
happens on — a reference source line, a chunk body), an optional **display text** (what a retrieved
hit shows — a worked ``source -> target`` example, a chunk with a citation), and metadata. The
lexical index tokenises the index text; the vector index embeds it (through an embedding cache, so
re-adding the same text costs nothing); the display text and metadata are what a hit resolves to.

Deterministic and read-only once built, so one corpus is safely shared by a run's workers — the
same guarantee the retrievers need.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ragkit.core.ports import LexicalIndex, Retriever, VectorIndex

from ..retrieve.embedding import EmbeddingClient
from ..retrieve.hybrid import HybridRetriever
from ..retrieve.rerank import RerankClient
from ..retrieve.retrievers import DenseRetriever, LexicalRetriever


@dataclass(frozen=True, slots=True)
class CorpusItem:
    """One thing to index. ``display_text`` defaults to ``index_text`` when omitted."""

    chunk_id: str
    index_text: str
    display_text: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def shown(self) -> str:
        return self.display_text if self.display_text is not None else self.index_text


class Corpus:
    """A built retrieval corpus over the storage indexes it is given.

    Pass a lexical index for BM25 retrieval, a vector index + embedder for dense retrieval, or both
    for hybrid. Text and metadata for resolving hits are held in memory (a corpus is rebuilt per
    run from its source, matching how a reference retriever is seeded); the heavy artefacts (the
    inverted index, the vectors) live in the stores.
    """

    def __init__(self, *, lexical: LexicalIndex | None = None, vector: VectorIndex | None = None,
                 embedder: EmbeddingClient | None = None) -> None:
        if vector is not None and embedder is None:
            raise ValueError("a vector index needs an embedder to build the corpus")
        self._lexical = lexical
        self._vector = vector
        self._embedder = embedder
        self._shown: dict[str, str] = {}
        self._meta: dict[str, Mapping[str, Any]] = {}
        self._embed_cache: dict[str, Sequence[float]] = {}

    def add_all(self, items: Iterable[CorpusItem]) -> int:
        """Index every item; returns the count. Embeddings are batched and cached, so re-adding the
        same index text (a duplicate across the corpus) is free."""
        batch = list(items)
        for item in batch:
            self._shown[item.chunk_id] = item.shown
            self._meta[item.chunk_id] = dict(item.meta)
            if self._lexical is not None:
                self._lexical.index(item.chunk_id, item.index_text)
        if self._vector is not None and self._embedder is not None:
            self._embed_and_upsert(batch)
        return len(batch)

    def _embed_and_upsert(self, batch: Sequence[CorpusItem]) -> None:
        assert self._embedder is not None and self._vector is not None
        # Only embed texts not already cached, then reuse the cache for the rest.
        to_embed = [item.index_text for item in batch
                    if _key(item.index_text) not in self._embed_cache]
        unique = list(dict.fromkeys(to_embed))  # de-dup within the batch, preserve order
        if unique:
            vectors = self._embedder.embed(unique)
            for text, vector in zip(unique, vectors, strict=True):
                self._embed_cache[_key(text)] = vector
        ids = [item.chunk_id for item in batch]
        vecs = [self._embed_cache[_key(item.index_text)] for item in batch]
        metas = [self._meta[item.chunk_id] for item in batch]
        self._vector.upsert(ids, vecs, metas)

    def resolve(self, chunk_id: str) -> str | None:
        return self._shown.get(chunk_id)

    def resolve_meta(self, chunk_id: str) -> Mapping[str, Any]:
        return self._meta.get(chunk_id, {})

    def __len__(self) -> int:
        return len(self._shown)

    # -- retrievers ----------------------------------------------------------

    def lexical_retriever(self, *, default_k: int = 10) -> LexicalRetriever:
        if self._lexical is None:
            raise ValueError("this corpus has no lexical index; build it with one")
        return LexicalRetriever(self._lexical, self.resolve, meta=self.resolve_meta,
                                default_k=default_k)

    def dense_retriever(self) -> DenseRetriever:
        if self._vector is None or self._embedder is None:
            raise ValueError("this corpus has no vector index/embedder; build it with both")
        return DenseRetriever(self._vector, self._embedder, self.resolve, meta=self.resolve_meta)

    def hybrid_retriever(self, *, reranker: RerankClient | None = None, candidate_pool: int = 40,
                         mmr_lambda: float = 0.7, lexical_min_score: float = 0.30,
                         dense_min_score: float = 0.55) -> HybridRetriever:
        return HybridRetriever(
            self.lexical_retriever(), self.dense_retriever(), reranker=reranker,
            candidate_pool=candidate_pool, mmr_lambda=mmr_lambda,
            lexical_min_score=lexical_min_score, dense_min_score=dense_min_score)

    def retriever(self) -> Retriever:
        """The single best retriever this corpus can offer with what it was built from: hybrid if
        both indexes are present, else whichever one is."""
        if self._lexical is not None and self._vector is not None:
            return self.hybrid_retriever()
        if self._vector is not None:
            return self.dense_retriever()
        return self.lexical_retriever()


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
