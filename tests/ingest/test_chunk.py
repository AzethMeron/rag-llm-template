"""The chunkers and their provenance."""
from __future__ import annotations

import pytest

from ragkit.core.ports import Document
from ragkit.ingest.chunk import (
    CHUNKERS,
    ChunkError,
    FixedChunker,
    SentenceChunker,
    StructureChunker,
)


def _doc(text: str) -> Document:
    return Document(doc_id="d", text=text, meta={"source": "test"})


class TestFixed:
    def test_windows_with_overlap(self) -> None:
        chunks = list(FixedChunker(size=10, overlap=3).chunk(_doc("abcdefghijklmnopqrst")))
        assert len(chunks) >= 2
        assert all(len(c.text) <= 10 for c in chunks)

    def test_provenance(self) -> None:
        [chunk] = list(FixedChunker(size=100, overlap=0).chunk(_doc("short")))
        assert chunk.meta["document_id"] == "d" and chunk.meta["ordinal"] == 0
        assert chunk.meta["chunker"] == "fixed" and chunk.meta["source"] == "test"

    def test_bad_size(self) -> None:
        with pytest.raises(ChunkError, match="size must be >= 1"):
            FixedChunker(size=0)

    def test_bad_overlap(self) -> None:
        with pytest.raises(ChunkError, match="overlap"):
            FixedChunker(size=10, overlap=10)


class TestSentence:
    def test_packs_whole_sentences(self) -> None:
        text = "First sentence. Second one. Third here. Fourth ends."
        chunks = list(SentenceChunker(target=25).chunk(_doc(text)))
        assert len(chunks) >= 2
        # No chunk cuts a sentence: each ends with terminal punctuation.
        assert all(c.text.rstrip().endswith((".", "!", "?")) for c in chunks)

    def test_bad_target(self) -> None:
        with pytest.raises(ChunkError, match="target must be >= 1"):
            SentenceChunker(target=0)


class TestStructure:
    def test_respects_paragraph_boundaries(self) -> None:
        text = "Para one here.\n\nPara two here.\n\nPara three."
        chunks = list(StructureChunker(target=20).chunk(_doc(text)))
        assert len(chunks) >= 2

    def test_splits_oversized_paragraph(self) -> None:
        big = "Sentence one is here. " * 20  # one huge paragraph
        chunks = list(StructureChunker(target=50, maximum=60).chunk(_doc(big)))
        assert len(chunks) > 1  # the oversized element was split by sentence

    def test_bad_bounds(self) -> None:
        with pytest.raises(ChunkError, match="target <= maximum"):
            StructureChunker(target=100, maximum=50)


def test_builtins_registered() -> None:
    assert set(CHUNKERS.available()) >= {"fixed", "sentence", "structure"}
    assert CHUNKERS.create("structure", {"target": 500}) is not None
