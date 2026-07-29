"""BM25 lexical index over SQLite FTS5 — the zero-dependency default lexical retriever.

FTS5 gives a real ``bm25()`` ranking (k1=1.2, b=0.75) with no extra dependency, and when the index
lives in the same database file as the chunk rows it cannot drift from them. The one sharp edge is
the score convention: FTS5's ``bm25()`` is **negative and lower-is-better**. The
:class:`~ragkit.core.ports.LexicalIndex` port promises the opposite — higher-is-better toward
``[0, 1]`` — so the sign is converted *here*, inside the driver, and can never leak into the fuser
to silently invert a ranking.

Alongside the FTS index this driver keeps a plain ``docs`` table (``chunk_id`` primary key →
display text + metadata) in the *same* database. That is what lets a large corpus be resolved from
disk rather than a RAM dict: retrieval returns ids from the FTS index, and each id's display text is
read back from ``docs`` by an indexed primary-key lookup — so neither ingest nor resolution holds
the corpus in memory. Ingest is streamed and batched (:meth:`index_many`); the per-row
:meth:`index` stays for the small, idempotent single-item path the port requires.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Mapping
from typing import Any

from ragkit.core.errors import RagkitError


class LexicalIndexError(RagkitError):
    """The lexical index could not be built or queried (FTS5 is likely not compiled in)."""


def _bm25_to_relevance(bm25: float) -> float:
    """Map FTS5's negative, lower-is-better ``bm25()`` to a higher-is-better score in ``[0, 1)``.

    ``bm25`` is <= 0 for a match; more negative means a better match. ``1 - 2**bm25`` is monotone
    increasing in the match quality: 0 for a marginal match, approaching 1 for a very strong one.
    A bounded, order-preserving transform, which is all a relevance floor and a fuser need.
    """
    return 1.0 - 2.0 ** bm25


class Fts5Index:
    """A :class:`~ragkit.core.ports.LexicalIndex` backed by an FTS5 virtual table, with an adjacent
    ``docs`` table so display text and metadata resolve from disk instead of a RAM dict."""

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
            # The chunk-row store, keyed for O(log n) reverse lookup at resolution time (FTS5's own
            # UNINDEXED chunk_id would force a full scan per hit). meta is JSON, "" when absent.
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS docs ("
                "chunk_id TEXT PRIMARY KEY, display TEXT NOT NULL, meta TEXT NOT NULL)")
            self._conn.commit()
        except sqlite3.Error as exc:
            raise LexicalIndexError(
                f"could not create the FTS5 tables (is FTS5 compiled into this SQLite?): {exc}"
                ) from exc

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> Fts5Index:
        return cls(path=str(options.get("path", ":memory:")),
                   tokenizer=str(options.get("tokenizer", "unicode61")))

    def index(self, chunk_id: str, text: str) -> None:
        """Index one item for BM25 and store it for resolution (display = the indexed text, no
        metadata). Idempotent: re-indexing an id replaces its row. Commits immediately — the
        single-item path; use :meth:`index_many` for bulk ingest."""
        with self._lock:
            self._replace_one(chunk_id, text, text, {})
            self._conn.commit()

    def index_many(self, rows: Iterable[tuple[str, str, str, Mapping[str, Any]]]) -> int:
        """Bulk-ingest ``(chunk_id, index_text, display_text, meta)`` tuples in one transaction.

        Assumes fresh, unique ids (the corpus-build path assigns ``ref-N``): it inserts without a
        per-row delete, since deleting by the UNINDEXED ``chunk_id`` would scan the whole FTS table
        for every row and make large ingest quadratic. Returns the number of rows written.
        """
        lex_rows: list[tuple[str, str]] = []
        doc_rows: list[tuple[str, str, str]] = []
        for chunk_id, index_text, display_text, meta in rows:
            lex_rows.append((chunk_id, index_text))
            doc_rows.append((chunk_id, display_text, json.dumps(dict(meta), ensure_ascii=False)))
        if not lex_rows:
            return 0
        with self._lock:
            self._conn.executemany("INSERT INTO lex(chunk_id, text) VALUES (?, ?)", lex_rows)
            self._conn.executemany(
                "INSERT OR REPLACE INTO docs(chunk_id, display, meta) VALUES (?, ?, ?)", doc_rows)
            self._conn.commit()
        return len(lex_rows)

    def _replace_one(self, chunk_id: str, index_text: str, display: str,
                     meta: Mapping[str, Any]) -> None:
        self._conn.execute("DELETE FROM lex WHERE chunk_id = ?", (chunk_id,))
        self._conn.execute("INSERT INTO lex(chunk_id, text) VALUES (?, ?)", (chunk_id, index_text))
        self._conn.execute(
            "INSERT OR REPLACE INTO docs(chunk_id, display, meta) VALUES (?, ?, ?)",
            (chunk_id, display, json.dumps(dict(meta), ensure_ascii=False)))

    def document(self, chunk_id: str) -> tuple[str, Mapping[str, Any]] | None:
        """Resolve a chunk id to ``(display_text, meta)`` from disk, or ``None`` if unknown."""
        with self._lock:
            row = self._conn.execute(
                "SELECT display, meta FROM docs WHERE chunk_id = ?", (chunk_id,)).fetchone()
        if row is None:
            return None
        display, meta = row
        return display, json.loads(meta)

    def search(self, query: str, *, k: int) -> list[tuple[str, float]]:
        if k <= 0 or not query.strip():
            return []
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT chunk_id, bm25(lex) AS score FROM lex WHERE lex MATCH ? "
                    "ORDER BY score LIMIT ?", (_as_match(query), k)).fetchall()
            except sqlite3.Error as exc:
                raise LexicalIndexError(f"FTS5 query failed: {exc}", query=query) from exc
        return [(chunk_id, _bm25_to_relevance(score)) for chunk_id, score in rows]

    def delete(self, chunk_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM lex WHERE chunk_id = ?", (chunk_id,))
            self._conn.execute("DELETE FROM docs WHERE chunk_id = ?", (chunk_id,))
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT count(*) FROM docs").fetchone()[0])

    def close(self) -> None:
        self._conn.close()


def _as_match(query: str) -> str:
    """Turn a free-text query into an FTS5 MATCH expression that treats every word as a term,
    OR-combined, quoting each so punctuation cannot be read as FTS5 query syntax (which would raise
    on an ordinary user query containing a quote, a colon, or a bare ``AND``)."""
    words = [w.replace('"', '""') for w in query.split() if w]
    return " OR ".join(f'"{w}"' for w in words) if words else '""'
