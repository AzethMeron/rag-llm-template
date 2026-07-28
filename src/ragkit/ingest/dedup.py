"""Deduplication of documents and chunks before indexing.

Exact duplicates (a document ingested twice, a boilerplate paragraph repeated) waste storage and,
worse, let a retriever return several near-identical hits that crowd the prompt with one idea. This
does the cheap, exact half — a content hash over normalised text — deterministically. Near-
duplicate detection (MinHash/SimHash) is a heavier, separate concern left to a plugin.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator

from ragkit.core.ports import Chunk, Document


def content_hash(text: str) -> str:
    """A stable SHA-256 over the text — the exact-dedup key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dedup_documents(documents: Iterable[Document]) -> Iterator[Document]:
    """Yield documents in order, dropping any whose text is byte-identical to an earlier one."""
    seen: set[str] = set()
    for document in documents:
        digest = content_hash(document.text)
        if digest not in seen:
            seen.add(digest)
            yield document


def dedup_chunks(chunks: Iterable[Chunk]) -> Iterator[Chunk]:
    """Yield chunks in order, dropping any whose text duplicates an earlier chunk's."""
    seen: set[str] = set()
    for chunk in chunks:
        digest = content_hash(chunk.text)
        if digest not in seen:
            seen.add(digest)
            yield chunk
