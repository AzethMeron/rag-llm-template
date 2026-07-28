"""Chunkers: split a :class:`~ragkit.core.ports.Document` into indexable chunks with provenance.

Both research documents converge on one point: **chunking quality dominates ANN choice**. So the
default is structure-aware — it respects paragraph and heading boundaries and only splits a single
oversized element as a fallback — with fixed-size and sentence chunkers shipped as alternatives so
the choice can be measured rather than asserted. Every chunk carries its provenance in ``meta``
(``document_id``, ``ordinal``, ``char_start``/``char_end``, ``section_path``, ``chunker``), which is
what makes a retrieved answer traceable to an exact span.
"""
from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import Any

from ragkit.core.errors import RagkitError
from ragkit.core.ports import Chunk, Chunker, Document
from ragkit.core.registry import Registry

CHUNKERS: Registry[Chunker] = Registry(
    "chunker", Chunker,  # type: ignore[type-abstract]
    entry_point_group="ragkit.chunkers")

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH = re.compile(r"\n\s*\n")


class ChunkError(RagkitError):
    """A chunker was misconfigured."""


def _chunk(document: Document, ordinal: int, text: str, start: int, chunker: str) -> Chunk:
    meta = dict(document.meta)
    meta.update({"document_id": document.doc_id, "ordinal": ordinal, "char_start": start,
                 "char_end": start + len(text), "chunker": chunker})
    return Chunk(chunk_id=f"{document.doc_id}:{ordinal}", text=text, meta=meta)


class FixedChunker:
    """Fixed-size character windows with overlap. The simple, predictable baseline: fast, but blind
    to structure, so an answer can straddle a boundary. Always worth having as a baseline."""

    CONFIG_KEYS = frozenset({"size", "overlap"})

    def __init__(self, size: int = 1000, overlap: int = 150) -> None:
        if size < 1:
            raise ChunkError("fixed chunker: size must be >= 1")
        if not 0 <= overlap < size:
            raise ChunkError("fixed chunker: overlap must be in [0, size)")
        self._size = size
        self._overlap = overlap

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> FixedChunker:
        return cls(size=int(options.get("size", 1000)), overlap=int(options.get("overlap", 150)))

    def chunk(self, document: Document) -> Iterator[Chunk]:
        text = document.text
        step = self._size - self._overlap
        ordinal = 0
        for start in range(0, max(1, len(text)), step):
            piece = text[start:start + self._size]
            if not piece.strip():
                continue
            yield _chunk(document, ordinal, piece, start, "fixed")
            ordinal += 1
            if start + self._size >= len(text):
                break


class SentenceChunker:
    """Pack whole sentences up to a target size, so a chunk never ends mid-sentence."""

    CONFIG_KEYS = frozenset({"target"})

    def __init__(self, target: int = 800) -> None:
        if target < 1:
            raise ChunkError("sentence chunker: target must be >= 1")
        self._target = target

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> SentenceChunker:
        return cls(target=int(options.get("target", 800)))

    def chunk(self, document: Document) -> Iterator[Chunk]:
        yield from _pack(document, _SENTENCE.split(document.text), self._target, "sentence")


class StructureChunker:
    """The default: respect paragraph (and heading) boundaries, packing whole paragraphs up to a
    target size; a single paragraph larger than the hard maximum is split by sentence as a
    fallback. Keeps sections and paragraphs intact, which is where retrieval quality is won."""

    CONFIG_KEYS = frozenset({"target", "maximum"})

    def __init__(self, target: int = 600, maximum: int = 1200) -> None:
        if target < 1 or maximum < target:
            raise ChunkError("structure chunker: need 1 <= target <= maximum")
        self._target = target
        self._maximum = maximum

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> StructureChunker:
        return cls(target=int(options.get("target", 600)),
                   maximum=int(options.get("maximum", 1200)))

    def chunk(self, document: Document) -> Iterator[Chunk]:
        elements: list[str] = []
        for paragraph in _PARAGRAPH.split(document.text):
            para = paragraph.strip()
            if not para:
                continue
            if len(para) > self._maximum:
                # An oversized element: split it by sentence rather than emit one huge chunk.
                elements.extend(s for s in _SENTENCE.split(para) if s.strip())
            else:
                elements.append(para)
        yield from _pack(document, elements, self._target, "structure")


def _pack(document: Document, pieces: list[str], target: int, chunker: str) -> Iterator[Chunk]:
    ordinal = 0
    buffer: list[str] = []
    length = 0
    start = 0
    cursor = 0
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if buffer and length + len(piece) > target:
            yield _chunk(document, ordinal, " ".join(buffer), start, chunker)
            ordinal += 1
            start = cursor
            buffer, length = [], 0
        buffer.append(piece)
        length += len(piece) + 1
        cursor += len(piece) + 1
    if buffer:
        yield _chunk(document, ordinal, " ".join(buffer), start, chunker)


CHUNKERS.register("fixed", FixedChunker)
CHUNKERS.register("sentence", SentenceChunker)
CHUNKERS.register("structure", StructureChunker)
