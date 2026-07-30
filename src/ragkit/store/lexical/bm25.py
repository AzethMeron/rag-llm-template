"""BM25 scoring and FTS5 query-escaping helpers, shared by every driver built on SQLite FTS5.

Extracted so the score convention and query-escaping live in exactly one place (CLAUDE.md: single
source of truth): both the standalone :class:`~ragkit.store.lexical.fts5.Fts5Index` and the
co-located pairing store's FTS5 table use the identical transform and the identical escaping, so a
ranking or a quoting bug can only exist once.
"""
from __future__ import annotations


def bm25_to_relevance(bm25: float) -> float:
    """Map FTS5's negative, lower-is-better ``bm25()`` to a higher-is-better score in ``[0, 1)``.

    ``bm25`` is <= 0 for a match; more negative means a better match. ``1 - 2**bm25`` is monotone
    increasing in the match quality: 0 for a marginal match, approaching 1 for a very strong one.
    A bounded, order-preserving transform, which is all a relevance floor and a fuser need.
    """
    return 1.0 - 2.0 ** bm25


def as_match(query: str) -> str:
    """Turn a free-text query into an FTS5 MATCH expression that treats every word as a term,
    OR-combined, quoting each so punctuation cannot be read as FTS5 query syntax (which would raise
    on an ordinary user query containing a quote, a colon, or a bare ``AND``)."""
    words = [w.replace('"', '""') for w in query.split() if w]
    return " OR ".join(f'"{w}"' for w in words) if words else '""'
