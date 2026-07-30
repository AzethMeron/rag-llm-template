"""The co-located pairing store over SQLite — rows and a BM25 search index in one database.

An FTS5 *external-content* table (``pairings_fts``) mirrors ``pairings`` via ``AFTER INSERT/UPDATE/
DELETE`` triggers, so every write to the row and its search entry happens inside the very same
statement's implicit transaction: there is no window where one exists without the other, and an
aborted write leaves neither behind. This is what the pre-overhaul split store (a relational row
table plus a separate FTS5 index, kept in sync only by write ordering) could only approximate.
"""
from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from ragkit.core.ports import Pairing

from ..lexical.bm25 import as_match, bm25_to_relevance
from .common import PairingStoreError, pairing_display

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pairings(
    id INTEGER PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    source TEXT NOT NULL,
    context TEXT NOT NULL DEFAULT '',
    target TEXT NOT NULL DEFAULT '',
    meta TEXT NOT NULL DEFAULT '{{}}',
    verified INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL DEFAULT 0
);
CREATE VIRTUAL TABLE IF NOT EXISTS pairings_fts USING fts5(
    source, context, target, content='pairings', content_rowid='id', tokenize='{tokenizer}'
);
CREATE TRIGGER IF NOT EXISTS pairings_ai AFTER INSERT ON pairings BEGIN
    INSERT INTO pairings_fts(rowid, source, context, target)
    VALUES (new.id, new.source, new.context, new.target);
END;
CREATE TRIGGER IF NOT EXISTS pairings_ad AFTER DELETE ON pairings BEGIN
    INSERT INTO pairings_fts(pairings_fts, rowid, source, context, target)
    VALUES ('delete', old.id, old.source, old.context, old.target);
END;
CREATE TRIGGER IF NOT EXISTS pairings_au AFTER UPDATE ON pairings BEGIN
    INSERT INTO pairings_fts(pairings_fts, rowid, source, context, target)
    VALUES ('delete', old.id, old.source, old.context, old.target);
    INSERT INTO pairings_fts(rowid, source, context, target)
    VALUES (new.id, new.source, new.context, new.target);
END;
"""


class SqlitePairings:
    """A :class:`~ragkit.core.ports.PairingStore` over one co-located SQLite database. Also
    satisfies :class:`~ragkit.core.ports.SearchIndex` (``search``), by design (see the port
    docstring)."""

    CONFIG_KEYS = frozenset({"path", "tokenizer"})

    def __init__(self, path: str = ":memory:", *, tokenizer: str = "unicode61") -> None:
        self._lock = threading.Lock()
        try:
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._conn.executescript(_SCHEMA.format(tokenizer=tokenizer))
            self._conn.commit()
        except sqlite3.Error as exc:
            raise PairingStoreError(
                f"could not open the pairing store at {path!r}: {exc}") from exc

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> SqlitePairings:
        return cls(path=str(options.get("path", ":memory:")),
                   tokenizer=str(options.get("tokenizer", "unicode61")))

    def add(self, pairings: Iterable[Pairing]) -> int:
        rows = [(p.chunk_id, p.source, p.context, p.target,
                 json.dumps(dict(p.meta), ensure_ascii=False), int(p.verified), p.created_at)
                for p in pairings]
        if not rows:
            return 0
        with self._lock:
            try:
                before = self._count_locked()
                self._conn.executemany(
                    "INSERT OR IGNORE INTO pairings"
                    "(chunk_id, source, context, target, meta, verified, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
                self._conn.commit()
                return self._count_locked() - before
            except sqlite3.Error as exc:
                self._safe_rollback()
                raise PairingStoreError(f"could not add pairings: {exc}") from exc

    def _safe_rollback(self) -> None:
        """Roll back, unless the connection itself is unusable (already closed) -- in which case
        there is nothing to roll back, and letting that failure replace the real one would mask
        the actual cause behind a confusing 'closed database' error."""
        with contextlib.suppress(sqlite3.Error):
            self._conn.rollback()

    def search(self, query: str, *, k: int) -> list[tuple[str, float]]:
        if k <= 0 or not query.strip():
            return []
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT p.chunk_id, bm25(pairings_fts) AS score FROM pairings_fts "
                    "JOIN pairings p ON p.id = pairings_fts.rowid "
                    "WHERE pairings_fts MATCH ? ORDER BY score LIMIT ?",
                    (as_match(query), k)).fetchall()
            except sqlite3.Error as exc:
                raise PairingStoreError(f"FTS5 query failed: {exc}", query=query) from exc
        return [(chunk_id, bm25_to_relevance(score)) for chunk_id, score in rows]

    def document(self, chunk_id: str) -> tuple[str, Mapping[str, Any]] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT source, target, meta FROM pairings WHERE chunk_id = ?",
                (chunk_id,)).fetchone()
        if row is None:
            return None
        source, target, meta = row
        return pairing_display(source, target), json.loads(meta)

    def get(self, chunk_id: str) -> Pairing | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT chunk_id, source, target, context, meta, verified, created_at "
                "FROM pairings WHERE chunk_id = ?", (chunk_id,)).fetchone()
        if row is None:
            return None
        chunk_id_, source, target, context, meta, verified, created_at = row
        return Pairing(chunk_id=chunk_id_, source=source, target=target, context=context,
                       meta=json.loads(meta), verified=bool(verified), created_at=created_at)

    def all_ids(self) -> Iterator[str]:
        with self._lock:
            rows = self._conn.execute("SELECT chunk_id FROM pairings ORDER BY id").fetchall()
        return (row[0] for row in rows)

    def _count_locked(self) -> int:
        return int(self._conn.execute("SELECT count(*) FROM pairings").fetchone()[0])

    def count(self) -> int:
        with self._lock:
            return self._count_locked()

    def close(self) -> None:
        self._conn.close()
