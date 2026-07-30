"""Exact-duplicate detection: content_hash, dedup_documents, dedup_chunks."""
from __future__ import annotations

from ragkit.core.ports import Chunk, Document
from ragkit.ingest.dedup import content_hash, dedup_chunks, dedup_documents


class TestContentHash:
    def test_deterministic(self) -> None:
        assert content_hash("hello") == content_hash("hello")

    def test_distinguishes_different_text(self) -> None:
        assert content_hash("hello") != content_hash("world")

    def test_is_a_sha256_hex_digest(self) -> None:
        assert len(content_hash("x")) == 64
        assert all(c in "0123456789abcdef" for c in content_hash("x"))


class TestDedupDocuments:
    def test_drops_byte_identical_text(self) -> None:
        docs = [Document("a", "same text"), Document("b", "same text"), Document("c", "other")]
        kept = list(dedup_documents(docs))
        assert [d.doc_id for d in kept] == ["a", "c"]  # first occurrence kept, order preserved

    def test_no_duplicates_keeps_everything(self) -> None:
        docs = [Document("a", "one"), Document("b", "two")]
        assert list(dedup_documents(docs)) == docs

    def test_empty_input(self) -> None:
        assert list(dedup_documents([])) == []


class TestDedupChunks:
    def test_drops_duplicate_chunk_text(self) -> None:
        chunks = [Chunk("1", "same"), Chunk("2", "same"), Chunk("3", "different")]
        kept = list(dedup_chunks(chunks))
        assert [c.chunk_id for c in kept] == ["1", "3"]

    def test_no_duplicates_keeps_everything(self) -> None:
        chunks = [Chunk("1", "one"), Chunk("2", "two")]
        assert list(dedup_chunks(chunks)) == chunks

    def test_empty_input(self) -> None:
        assert list(dedup_chunks([])) == []
