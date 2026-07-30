"""The established-terminology store over SQLite — DB-native lexicon, replacing a JSONL file as
the accumulating source of truth for a project's terminology.

Kept a separate small table from the pairing store (rather than folded into it) because a lexicon
entry is a different shape entirely (a term/rendering pair, not a source/target/context triple) and
has its own key (``term``, ``category``) — pointing a ``[lexicon]`` binding at the same file path as
a recipe's ``[pairings]`` binding still puts both in one physical database; nothing here requires
it.
"""
from __future__ import annotations

import contextlib
import sqlite3
import threading
from collections.abc import Iterable, Mapping
from typing import Any

from ragkit.core.errors import RagkitError
from ragkit.core.lexicon import Entry


class LexiconStoreError(RagkitError):
    """A lexicon-store operation failed, or the lexicon database is misconfigured."""


class SqliteLexicon:
    """A :class:`~ragkit.core.ports.LexiconStore` over one SQLite table (WAL). ``add`` upserts
    (keyed on ``term``+``category``): re-importing a corrected rendering replaces the stale one,
    unlike the pairing store's first-write-wins idempotency -- a lexicon is a *current* mapping,
    not a log of distinct produced facts."""

    CONFIG_KEYS = frozenset({"path"})

    def __init__(self, path: str = ":memory:") -> None:
        self._lock = threading.Lock()
        try:
            self._conn = sqlite3.connect(path, check_same_thread=False)
            # WAL + busy_timeout: see SqlitePairings.__init__ for why (a reader must not fail on
            # "database is locked" just because a writer's commit is in flight elsewhere).
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS lexicon("
                "term TEXT NOT NULL, rendering TEXT NOT NULL, category TEXT NOT NULL DEFAULT '', "
                "entity_id INTEGER NOT NULL DEFAULT 0, "
                "PRIMARY KEY (term, category))")
            self._conn.commit()
        except sqlite3.Error as exc:
            raise LexiconStoreError(
                f"could not open the lexicon store at {path!r}: {exc}") from exc

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> SqliteLexicon:
        return cls(path=str(options.get("path", ":memory:")))

    def entries(self) -> list[Entry]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT term, rendering, category, entity_id FROM lexicon").fetchall()
        return [Entry(term=term, rendering=rendering, category=category, entity_id=entity_id)
                for term, rendering, category, entity_id in rows]

    def add(self, entries: Iterable[Entry]) -> int:
        rows = [(e.term, e.rendering, e.category, e.entity_id) for e in entries]
        if not rows:
            return 0
        with self._lock:
            try:
                before = self._count_locked()
                self._conn.executemany(
                    "INSERT OR REPLACE INTO lexicon(term, rendering, category, entity_id) "
                    "VALUES (?, ?, ?, ?)", rows)
                self._conn.commit()
                return self._count_locked() - before
            except sqlite3.Error as exc:
                self._safe_rollback()
                raise LexiconStoreError(f"could not add entries: {exc}") from exc

    def _count_locked(self) -> int:
        return int(self._conn.execute("SELECT count(*) FROM lexicon").fetchone()[0])

    def _safe_rollback(self) -> None:
        with contextlib.suppress(sqlite3.Error):
            self._conn.rollback()

    def close(self) -> None:
        self._conn.close()
