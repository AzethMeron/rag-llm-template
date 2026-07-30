"""BM25 lexical index over SQLite FTS5 — the zero-dependency default lexical retriever.

FTS5 gives a real ``bm25()`` ranking (k1=1.2, b=0.75) with no extra dependency. The one sharp edge
is the score convention: FTS5's ``bm25()`` is **negative and lower-is-better**. The
:class:`~ragkit.core.ports.LexicalIndex` port promises the opposite — higher-is-better toward
``[0, 1]`` — so the sign is converted *here*, inside the driver, and can never leak into the fuser
to silently invert a ranking.

This is a search index only: it maps a query to ``(chunk_id, score)``. Turning an id back into its
display text + metadata is the :class:`~ragkit.core.ports.DocumentStore`'s job, so a huge corpus is
never held in RAM. Bulk ingest is batched (:meth:`index_many`); the per-row :meth:`index` stays for
the small, idempotent single-item path the port requires.
"""
from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable, Mapping
from typing import Any

from ragkit.core.errors import RagkitError

from .bm25 import as_match, bm25_to_relevance


class LexicalIndexError(RagkitError):
    """The lexical index could not be built or queried (FTS5 is likely not compiled in)."""


class Fts5Index:
    """A :class:`~ragkit.core.ports.LexicalIndex` backed by an FTS5 virtual table."""

    CONFIG_KEYS = frozenset({"path", "tokenizer"})

    def __init__(self, path: str = ":memory:", *, tokenizer: str = "unicode61") -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        try:
            # chunk_id is UNINDEXED (stored, not searched); only text is tokenised. The tokenizer
            # is validated by FTS5 itself at creation -- a bad name fails loudly here.
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS lex USING fts5("
                f"chunk_id UNINDEXED, text, tokenize='{tokenizer}')")
            self._conn.commit()
        except sqlite3.Error as exc:
            raise LexicalIndexError(
                f"could not create an FTS5 table (is FTS5 compiled into this SQLite?): {exc}"
                ) from exc

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> Fts5Index:
        return cls(path=str(options.get("path", ":memory:")),
                   tokenizer=str(options.get("tokenizer", "unicode61")))

    def index(self, chunk_id: str, text: str) -> None:
        with self._lock:
            # Replace any existing row for this id, so re-indexing is idempotent.
            self._conn.execute("DELETE FROM lex WHERE chunk_id = ?", (chunk_id,))
            self._conn.execute("INSERT INTO lex(chunk_id, text) VALUES (?, ?)", (chunk_id, text))
            self._conn.commit()

    def index_many(self, rows: Iterable[tuple[str, str]]) -> int:
        """Bulk-ingest ``(chunk_id, text)`` pairs in one transaction. Assumes fresh, unique ids (the
        corpus-build path assigns ``ref-N``): it inserts without a per-row delete, since deleting by
        the UNINDEXED ``chunk_id`` would scan the whole table for every row and make large ingest
        quadratic. Returns the number of rows written."""
        pairs = list(rows)
        if not pairs:
            return 0
        with self._lock:
            self._conn.executemany("INSERT INTO lex(chunk_id, text) VALUES (?, ?)", pairs)
            self._conn.commit()
        return len(pairs)

    def search(self, query: str, *, k: int) -> list[tuple[str, float]]:
        if k <= 0 or not query.strip():
            return []
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT chunk_id, bm25(lex) AS score FROM lex WHERE lex MATCH ? "
                    "ORDER BY score LIMIT ?", (as_match(query), k)).fetchall()
            except sqlite3.Error as exc:
                raise LexicalIndexError(f"FTS5 query failed: {exc}", query=query) from exc
        return [(chunk_id, bm25_to_relevance(score)) for chunk_id, score in rows]

    def delete(self, chunk_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM lex WHERE chunk_id = ?", (chunk_id,))
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT count(*) FROM lex").fetchone()[0])

    def close(self) -> None:
        self._conn.close()
