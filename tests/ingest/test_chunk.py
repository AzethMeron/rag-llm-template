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


class TestProvenanceSpansAreReal:
    """Regression: ``_pack`` computed offsets from a running cursor that added one synthetic space
    per piece, so ``char_start``/``char_end`` were positions in a reconstructed
    single-space-joined string, not in ``document.text``. The real text has ``\\n\\n`` and
    arbitrary whitespace between pieces, so every span past the first was wrong -- and a citation
    or highlight feature slicing ``document.text[char_start:char_end]`` got the wrong text. No
    test asserted these fields at all, which is why it survived. Only ``FixedChunker``, which
    slices the text directly, was correct."""

    @pytest.mark.parametrize("chunker", [
        FixedChunker(size=30, overlap=0),
        SentenceChunker(target=25),
        StructureChunker(target=25),
    ])
    def test_the_recorded_span_contains_the_chunks_own_words(self, chunker: object) -> None:
        text = ("First paragraph here.   It has two sentences.\n\n"
                "Second paragraph follows.\n\n\n"
                "Third paragraph, after extra blank lines.")
        document = _doc(text)
        for chunk in chunker.chunk(document):  # type: ignore[attr-defined]
            span = text[chunk.meta["char_start"]:chunk.meta["char_end"]]
            # The chunk's own text is the pieces joined by single spaces, so it need not equal the
            # span verbatim -- but every word of it must come from that span, in order.
            assert span.split() == chunk.text.split(), (
                f"span {chunk.meta['char_start']}:{chunk.meta['char_end']} is not this chunk's "
                f"text")

    def test_spans_advance_and_stay_in_bounds(self) -> None:
        text = "Alpha para.\n\nBeta para.\n\nGamma para.\n\nDelta para."
        document = _doc(text)
        chunks = list(StructureChunker(target=12).chunk(document))
        assert len(chunks) > 1
        previous_end = 0
        for chunk in chunks:
            start, end = chunk.meta["char_start"], chunk.meta["char_end"]
            assert 0 <= start < end <= len(text)
            assert start >= previous_end  # spans never overlap or go backwards
            previous_end = end

    def test_the_first_span_is_not_the_only_correct_one(self) -> None:
        # The precise shape of the old bug: piece 1's offset was right, piece 2's was short by
        # exactly the width of the real separator minus the one synthetic space.
        text = "Alpha.\n\nBeta."
        [first, second] = list(StructureChunker(target=6).chunk(_doc(text)))
        assert (first.meta["char_start"], first.meta["char_end"]) == (0, 6)
        assert (second.meta["char_start"], second.meta["char_end"]) == (8, 13)
        assert text[8:13] == "Beta."

    def test_a_piece_that_is_not_in_the_document_is_refused(self) -> None:
        # _pack's guarantee depends on every piece being a stripped substring of document.text.
        # A chunker that transformed its pieces would otherwise record a silently wrong span.
        from ragkit.ingest.chunk import _pack
        with pytest.raises(ChunkError, match="not a substring"):
            list(_pack(_doc("real text"), ["invented"], 100, "broken"))


def test_builtins_registered() -> None:
    assert set(CHUNKERS.available()) >= {"fixed", "sentence", "structure"}
    assert CHUNKERS.create("structure", {"target": 500}) is not None
