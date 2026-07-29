"""The chunk-row store over SQLite — the relational default for a :class:`DocumentStore`.

Every retrieval path (lexical, dense, hybrid) resolves a hit's display text + metadata through this
one store, so the corpus lives in the database rather than a RAM map: a search index returns ids,
and ``document`` turns an id back into ``(display, meta)`` by an indexed primary-key lookup. Writes
are batched into one transaction (:meth:`add_documents`); ``meta`` is stored as JSON.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Mapping
from typing import Any

from ragkit.core.errors import RagkitError


class DocumentStoreError(RagkitError):
    """The document store could not be built or queried."""


class SqliteDocuments:
    """A :class:`~ragkit.core.ports.DocumentStore` backed by a single SQLite table."""

    CONFIG_KEYS = frozenset({"path"})

    def __init__(self, path: str = ":memory:") -> None:
        self._lock = threading.Lock()
        try:
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS docs ("
                "chunk_id TEXT PRIMARY KEY, display TEXT NOT NULL, meta TEXT NOT NULL)")
            self._conn.commit()
        except sqlite3.Error as exc:
            raise DocumentStoreError(
                f"could not open the document store at {path!r}: {exc}") from exc

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> SqliteDocuments:
        return cls(path=str(options.get("path", ":memory:")))

    def add_documents(self, rows: Iterable[tuple[str, str, Mapping[str, Any]]]) -> None:
        """Insert (or replace) chunk rows in one transaction. ``meta`` is JSON-encoded."""
        prepared = [(chunk_id, display, json.dumps(dict(meta), ensure_ascii=False))
                    for chunk_id, display, meta in rows]
        if not prepared:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO docs(chunk_id, display, meta) VALUES (?, ?, ?)", prepared)
            self._conn.commit()

    def document(self, chunk_id: str) -> tuple[str, Mapping[str, Any]] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT display, meta FROM docs WHERE chunk_id = ?", (chunk_id,)).fetchone()
        if row is None:
            return None
        display, meta = row
        return display, json.loads(meta)

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT count(*) FROM docs").fetchone()[0])

    def close(self) -> None:
        self._conn.close()
