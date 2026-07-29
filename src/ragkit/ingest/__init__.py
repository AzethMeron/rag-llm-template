"""Ingestion: build a retrieval corpus from a source, through extract -> normalise -> dedup ->
chunk -> embed -> index. The corpus builder ties chunks into the storage indexes with an embedding
cache and hands back retrievers; the extractors and chunkers are registered components, swappable
by config. Depends on retrieve, store, and core."""
from __future__ import annotations

from .chunk import CHUNKERS, ChunkError, FixedChunker, SentenceChunker, StructureChunker
from .corpus import Corpus, CorpusItem
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

__all__ = [
    "Corpus", "CorpusItem",
    "EXTRACTORS", "TextExtractor", "JsonlExtractor", "HtmlExtractor", "MarkdownExtractor",
    "ExtractError",
    "CHUNKERS", "FixedChunker", "SentenceChunker", "StructureChunker", "ChunkError",
    "normalise", "content_hash", "dedup_documents", "dedup_chunks",
]
