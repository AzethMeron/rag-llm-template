"""Ingestion: build a reference memory from a source, through extract -> normalise -> dedup ->
chunk -> embed -> import. ``reference.py`` streams a JSONL corpus into a ``PairingStore`` and hands
back retrievers; the extractors and chunkers are registered components, swappable by config.
Depends on retrieve, store, and core.

**One ordering constraint, if you compose these yourself.** ``normalise``'s default
``collapse_whitespace=True`` folds every whitespace run — including the ``\\n\\n`` between
paragraphs — to a single space. :class:`~ragkit.ingest.chunk.StructureChunker`, the default
chunker, finds its boundaries by splitting on exactly that, so normalising first leaves it one
giant paragraph and it silently degrades to whole-document sentence packing: the
structure-preserving behaviour that is its entire point, gone, with no error. Chunk **before**
you collapse whitespace (normalise each chunk instead), or pass ``collapse_whitespace=False``.
The other chunkers are unaffected — ``FixedChunker`` ignores structure, and ``SentenceChunker``
splits on sentence punctuation, which survives collapsing."""
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
