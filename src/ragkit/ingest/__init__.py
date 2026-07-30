"""Ingestion: build a reference memory from a source, through extract -> normalise -> dedup ->
chunk -> embed -> import. ``reference.py`` streams a JSONL corpus into a ``PairingStore`` and hands
back retrievers; the extractors and chunkers are registered components, swappable by config.
Depends on retrieve, store, and core."""
from __future__ import annotations

from .chunk import CHUNKERS, ChunkError, FixedChunker, SentenceChunker, StructureChunker
from .dedup import content_hash, dedup_chunks, dedup_documents
from .extract import (
    EXTRACTORS,
    ExtractError,
    HtmlExtractor,
    JsonlExtractor,
    MarkdownExtractor,
    TextExtractor,
)
from .normalise import normalise
from .reference import (
    PairingRetrievers,
    ReferenceImportError,
    import_reference,
    reference_pairings,
)

__all__ = [
    "PairingRetrievers", "ReferenceImportError", "import_reference", "reference_pairings",
    "EXTRACTORS", "TextExtractor", "JsonlExtractor", "HtmlExtractor", "MarkdownExtractor",
    "ExtractError",
    "CHUNKERS", "FixedChunker", "SentenceChunker", "StructureChunker", "ChunkError",
    "normalise", "content_hash", "dedup_documents", "dedup_chunks",
]
