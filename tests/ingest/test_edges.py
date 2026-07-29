"""Remaining branch coverage for the ingest layer: from_config paths and empty/edge inputs."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from ragkit.core.ports import Document
from ragkit.ingest.chunk import CHUNKERS, FixedChunker, SentenceChunker, StructureChunker
from ragkit.ingest.corpus import Corpus, CorpusItem
from ragkit.ingest.extract import HtmlExtractor, JsonlExtractor, MarkdownExtractor, TextExtractor
from ragkit.retrieve.embedding import EmbeddingClient
from ragkit.store.lexical.fts5 import Fts5Index
from ragkit.store.vector.lancedb import LanceVectorIndex


def _embedder() -> EmbeddingClient:
    def handler(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 0.0]} for _ in inputs]})
    return EmbeddingClient(base_url="http://x/v1",
                           client=httpx.Client(transport=httpx.MockTransport(handler)))


class TestFromConfig:
    def test_extractors(self) -> None:
        assert JsonlExtractor.from_config({"field": "t", "id_field": "k"}) is not None
        assert HtmlExtractor.from_config({"doc_id": "h"}) is not None
        assert MarkdownExtractor.from_config({"doc_id": "m"}) is not None
        assert TextExtractor.from_config({}) is not None

    def test_chunkers(self) -> None:
        assert FixedChunker.from_config({"size": 500, "overlap": 50}) is not None
        assert SentenceChunker.from_config({"target": 400}) is not None
        assert StructureChunker.from_config({"target": 300, "maximum": 900}) is not None


class TestExtractEdges:
    def test_html_with_whitespace_and_no_heading(self) -> None:
        [doc] = list(HtmlExtractor().extract("<p>  </p><p>real text</p>"))
        assert doc.text == "real text" and "headings" not in doc.meta

    def test_markdown_heading_only_section(self) -> None:
        docs = list(MarkdownExtractor().extract("# Heading only\n\n# Another\n\nbody"))
        assert any(d.meta.get("section_path") == "Heading only" for d in docs)

    def test_markdown_empty_source(self) -> None:
        assert list(MarkdownExtractor().extract("   ")) == []

    def test_markdown_whitespace_preamble_before_first_heading(self) -> None:
        # The preamble section holds only whitespace: it renders to empty and is skipped.
        docs = list(MarkdownExtractor().extract("   \n\n# Real\n\nbody"))
        assert [d.meta.get("section_path") for d in docs] == ["Real"]

    def test_structure_all_whitespace_yields_nothing(self) -> None:
        assert list(StructureChunker(target=50).chunk(Document("d", "   \n\n   "))) == []


class TestChunkEdges:
    def test_fixed_skips_blank_windows(self) -> None:
        chunks = list(FixedChunker(size=3, overlap=0).chunk(Document("d", "a\n\n\n\n\n\nb")))
        assert all(c.text.strip() for c in chunks)

    def test_fixed_all_whitespace_yields_nothing(self) -> None:
        assert list(FixedChunker(size=10, overlap=0).chunk(Document("d", "     "))) == []

    def test_sentence_trailing_blank_piece(self) -> None:
        # A trailing "A. " splits into ["A.", ""]; the blank piece is skipped in packing.
        chunks = list(SentenceChunker(target=50).chunk(Document("d", "One. Two. ")))
        assert chunks and all(c.text.strip() for c in chunks)

    def test_structure_leading_blank_paragraphs(self) -> None:
        chunks = list(StructureChunker(target=50).chunk(Document("d", "\n\n\n\nReal para.")))
        assert len(chunks) == 1 and "Real para" in chunks[0].text

    def test_structure_registered(self) -> None:
        assert CHUNKERS.create("fixed", {"size": 100, "overlap": 0}) is not None


class TestCorpusEdges:
    def test_no_lexical_index_refuses_lexical_retriever(self, tmp_path: Path) -> None:
        corpus = Corpus(vector=LanceVectorIndex(str(tmp_path / "v"), dim=2), embedder=_embedder())
        with pytest.raises(ValueError, match="no lexical index"):
            corpus.lexical_retriever()

    def test_vector_only_retriever_is_dense(self, tmp_path: Path) -> None:
        from ragkit.retrieve.retrievers import DenseRetriever
        corpus = Corpus(vector=LanceVectorIndex(str(tmp_path / "v"), dim=2), embedder=_embedder())
        corpus.add_all([CorpusItem("1", "text")])
        assert isinstance(corpus.retriever(), DenseRetriever)

    def test_lexical_only_add_without_vector(self) -> None:
        corpus = Corpus(lexical=Fts5Index())
        assert corpus.add_all([CorpusItem("1", "a", meta={"k": "v"})]) == 1
        assert corpus.resolve_meta("1")["k"] == "v"

    def test_dedup_is_within_batch_not_across_adds(self, tmp_path: Path) -> None:
        calls: list[int] = []

        def counting(request: httpx.Request) -> httpx.Response:
            inputs = json.loads(request.content)["input"]
            calls.extend([1] * len(inputs))
            return httpx.Response(200, json={"data": [{"embedding": [1.0, 0.0]} for _ in inputs]})

        embedder = EmbeddingClient(base_url="http://x/v1",
                                   client=httpx.Client(transport=httpx.MockTransport(counting)))
        corpus = Corpus(vector=LanceVectorIndex(str(tmp_path / "v"), dim=2), embedder=embedder)
        # Identical texts in one batch embed once; a separate add re-embeds (no cross-batch RAM
        # cache — the vectors live in the store, not the heap).
        corpus.add_all([CorpusItem("1", "same"), CorpusItem("2", "same")])
        assert len(calls) == 1
        corpus.add_all([CorpusItem("3", "same")])
        assert len(calls) == 2
