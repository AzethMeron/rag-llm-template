"""Deduplication of documents and chunks before indexing.

Exact duplicates (a document ingested twice, a boilerplate paragraph repeated) waste storage and,
worse, let a retriever return several near-identical hits that crowd the prompt with one idea. This
does the cheap, exact half — a content hash over the text **exactly as given**, byte for byte —
deterministically. Normalising first is the caller's choice, not this module's: two texts differing
only in whitespace are *not* deduplicated here (the docstring used to claim otherwise). Near-
duplicate detection (MinHash/SimHash) is a heavier, separate concern left to a plugin.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator

from ragkit.core.ports import Chunk, Document


def content_hash(text: str) -> str:
    """A stable SHA-256 hex digest over the text — the exact-dedup key, and the public form.

    The in-process ``seen`` sets below use the raw 32-byte :func:`_digest` instead: a hex string
    is 64 characters, so holding one per document costs about twice the memory for no benefit
    when nothing outside the loop ever reads it.
    """
    return _digest(text).hex()


def _digest(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def dedup_documents(documents: Iterable[Document]) -> Iterator[Document]:
    """Yield documents in order, dropping any whose text is byte-identical to an earlier one."""
    seen: set[bytes] = set()
    for document in documents:
        digest = _digest(document.text)
        if digest not in seen:
            seen.add(digest)
            yield document


def dedup_chunks(chunks: Iterable[Chunk]) -> Iterator[Chunk]:
    """Yield chunks in order, dropping any whose text duplicates an earlier chunk's."""
    seen: set[bytes] = set()
    for chunk in chunks:
        digest = _digest(chunk.text)
        if digest not in seen:
            seen.add(digest)
            yield chunk
