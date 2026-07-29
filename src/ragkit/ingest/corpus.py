"""Build a retrieval corpus: index texts into a lexical and/or vector store, and hand back
retrievers over them.

This is the ingest end of retrieval. Each item has a stable id, an **index text** (what matching
happens on — a reference source line, a chunk body), an optional **display text** (what a retrieved
hit shows — a worked ``source -> target`` example, a chunk with a citation), and metadata. The
lexical index tokenises the index text and the vector index embeds it — both are search indexes
that return ids. The display text + metadata are the *row*, held in the :class:`DocumentStore`
(relational, on disk by default), so resolving a hit is a database lookup and a corpus of any size
is never held in RAM.

Deterministic and read-only once built, so one corpus is safely shared by a run's workers — the
same guarantee the retrievers need.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ragkit.core.ports import DocumentStore, LexicalIndex, Retriever, VectorIndex

from ..retrieve.embedding import EmbeddingClient
from ..retrieve.hybrid import HybridRetriever
from ..retrieve.rerank import RerankClient
from ..retrieve.retrievers import DenseRetriever, LexicalRetriever
from ..store.documents.sqlite import SqliteDocuments


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
    for hybrid. Every path resolves a hit's display text + metadata through the one
    :class:`~ragkit.core.ports.DocumentStore` (relational; on disk when configured, in-memory
    SQLite otherwise), so nothing is held in a RAM map. The heavy artefacts — the inverted index,
    the vectors, the rows — all live in the stores.
    """

    def __init__(self, *, lexical: LexicalIndex | None = None, vector: VectorIndex | None = None,
                 embedder: EmbeddingClient | None = None, documents: DocumentStore | None = None,
                 batch_size: int = 1000) -> None:
        if vector is not None and embedder is None:
            raise ValueError("a vector index needs an embedder to build the corpus")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._lexical = lexical
        self._vector = vector
        self._embedder = embedder
        self._batch_size = batch_size
        # Resolution always goes through a document store; absent an explicit one, an in-memory
        # SQLite store is used, so a hit still resolves through the database abstraction.
        self._documents: DocumentStore = documents if documents is not None else SqliteDocuments()

    def add_all(self, items: Iterable[CorpusItem]) -> int:
        """Index every item, streaming in batches so the source is never fully materialised in RAM;
        returns the count. Within a batch, identical index texts are embedded once; the rows,
        vectors, and inverted index all live in the stores, not a RAM cache."""
        total = 0
        batch: list[CorpusItem] = []
        for item in items:
            batch.append(item)
            if len(batch) >= self._batch_size:
                total += self._flush(batch)
                batch = []
        if batch:
            total += self._flush(batch)
        return total

    def _flush(self, batch: Sequence[CorpusItem]) -> int:
        if self._lexical is not None:
            index_many = getattr(self._lexical, "index_many", None)
            if index_many is not None:
                index_many((it.chunk_id, it.index_text) for it in batch)
            else:
                for item in batch:
                    self._lexical.index(item.chunk_id, item.index_text)
        if self._vector is not None and self._embedder is not None:
            self._embed_and_upsert(batch)
        self._documents.add_documents((it.chunk_id, it.shown, it.meta) for it in batch)
        return len(batch)

    def _embed_and_upsert(self, batch: Sequence[CorpusItem]) -> None:
        assert self._embedder is not None and self._vector is not None
        # De-dup identical index texts within the batch so each is embedded once (no cross-batch
        # RAM cache — a rare cross-batch duplicate is simply re-embedded).
        unique = list(dict.fromkeys(item.index_text for item in batch))
        embedded = dict(zip(unique, self._embedder.embed(unique), strict=True))
        ids = [item.chunk_id for item in batch]
        vecs = [embedded[item.index_text] for item in batch]
        metas = [dict(item.meta) for item in batch]
        self._vector.upsert(ids, vecs, metas)

    def resolve(self, chunk_id: str) -> str | None:
        doc = self._documents.document(chunk_id)
        return doc[0] if doc is not None else None

    def resolve_meta(self, chunk_id: str) -> Mapping[str, Any]:
        doc = self._documents.document(chunk_id)
        return doc[1] if doc is not None else {}

    def __len__(self) -> int:
        return self._documents.count()

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
