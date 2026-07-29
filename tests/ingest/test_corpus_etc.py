"""The corpus builder, normalisation, and deduplication."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from ragkit.core.ports import Chunk, Document
from ragkit.ingest import Corpus, CorpusItem, content_hash, dedup_chunks, dedup_documents, normalise
from ragkit.retrieve.embedding import EmbeddingClient
from ragkit.store.lexical.fts5 import Fts5Index
from ragkit.store.vector.lancedb import LanceVectorIndex


def _embedder(vector_of) -> EmbeddingClient:  # type: ignore[no-untyped-def]
    def handler(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        return httpx.Response(200, json={"data": [{"embedding": vector_of(t)} for t in inputs]})
    return EmbeddingClient(base_url="http://x/v1",
                           client=httpx.Client(transport=httpx.MockTransport(handler)))


class TestCorpus:
    def test_lexical_only(self) -> None:
        corpus = Corpus(lexical=Fts5Index())
        corpus.add_all([CorpusItem("1", "the cat sat", "cat -> kot"),
                        CorpusItem("2", "a dog barks")])
        assert len(corpus) == 2
        hits = corpus.retriever().retrieve("cat", k=3)
        assert hits[0].text == "cat -> kot"  # display text, not index text

    def test_dense_and_hybrid(self, tmp_path: Path) -> None:
        embedder = _embedder(lambda t: [1.0, 0.0] if "cat" in t else [0.0, 1.0])
        corpus = Corpus(lexical=Fts5Index(),
                        vector=LanceVectorIndex(str(tmp_path / "v"), dim=2), embedder=embedder)
        corpus.add_all([CorpusItem("1", "the cat sat"), CorpusItem("2", "a dog barks")])
        dense = corpus.dense_retriever().retrieve("cat query", k=2)
        assert dense[0].chunk_id == "1"
        # retriever() returns a hybrid when both indexes are present.
        from ragkit.retrieve.hybrid import HybridRetriever
        assert isinstance(corpus.retriever(), HybridRetriever)

    def test_embedding_cache_dedups(self, tmp_path: Path) -> None:
        calls: list[int] = []

        def counting(t: str) -> list[float]:
            calls.append(1)
            return [1.0, 0.0]

        embedder = _embedder(counting)
        corpus = Corpus(vector=LanceVectorIndex(str(tmp_path / "v"), dim=2), embedder=embedder)
        corpus.add_all([CorpusItem("1", "same text"), CorpusItem("2", "same text")])
        assert len(calls) == 1  # identical index text embedded once

    def test_vector_without_embedder_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="needs an embedder"):
            Corpus(vector=LanceVectorIndex(str(tmp_path / "v"), dim=2))

    def test_dense_without_index_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no vector index"):
            Corpus(lexical=Fts5Index()).dense_retriever()

    def test_batch_size_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            Corpus(lexical=Fts5Index(), batch_size=0)

    def test_streams_in_batches(self) -> None:
        # batch_size smaller than the item count exercises the mid-loop flush and the empty final
        # batch — the streaming path that keeps a multi-GB corpus off the heap.
        corpus = Corpus(lexical=Fts5Index(), batch_size=2)
        corpus.add_all([CorpusItem(str(i), f"passage {i} cat") for i in range(4)])
        assert len(corpus) == 4
        assert corpus.retriever().retrieve("cat", k=10).__len__() == 4

    def test_plain_lexical_index_resolves_in_memory(self) -> None:
        # A LexicalIndex without on-disk document storage: display/meta are kept in RAM and the
        # per-item index() path is used (no index_many).
        corpus = Corpus(lexical=_PlainLexicalIndex())
        corpus.add_all([CorpusItem("1", "cat sat", "cat -> kot", meta={"n": 1})])
        assert len(corpus) == 1
        assert corpus.retriever().retrieve("cat", k=1)[0].text == "cat -> kot"
        assert corpus.resolve("1") == "cat -> kot" and corpus.resolve_meta("1") == {"n": 1}


class _PlainLexicalIndex:
    """A minimal LexicalIndex with no ``document``/``index_many`` — forces the corpus onto its
    in-memory resolution and per-item indexing paths."""

    def __init__(self) -> None:
        self._docs: dict[str, str] = {}

    def index(self, chunk_id: str, text: str) -> None:
        self._docs[chunk_id] = text

    def search(self, query: str, *, k: int) -> list[tuple[str, float]]:
        return [(cid, 1.0) for cid, text in self._docs.items() if query in text][:k]

    def delete(self, chunk_id: str) -> None:
        self._docs.pop(chunk_id, None)


class TestNormalise:
    def test_nfc_and_whitespace(self) -> None:
        assert normalise("  café  \n\n  x ") == "café x"

    def test_dehyphenate(self) -> None:
        assert normalise("hyphen-\nated word", dehyphenate=True) == "hyphenated word"

    def test_bad_form(self) -> None:
        with pytest.raises(ValueError, match="normalisation form"):
            normalise("x", form="NFZ")

    def test_no_collapse(self) -> None:
        assert normalise("a  b", collapse_whitespace=False) == "a  b"


class TestDedup:
    def test_content_hash_stable(self) -> None:
        assert content_hash("x") == content_hash("x") != content_hash("y")

    def test_dedup_documents(self) -> None:
        docs = [Document("a", "same"), Document("b", "same"), Document("c", "other")]
        assert [d.doc_id for d in dedup_documents(docs)] == ["a", "c"]

    def test_dedup_chunks(self) -> None:
        chunks = [Chunk("1", "same"), Chunk("2", "same"), Chunk("3", "diff")]
        assert [c.chunk_id for c in dedup_chunks(chunks)] == ["1", "3"]
