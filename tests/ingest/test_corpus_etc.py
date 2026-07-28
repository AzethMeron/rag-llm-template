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
