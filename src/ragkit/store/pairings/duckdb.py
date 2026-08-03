"""The pairing store over DuckDB — a second real ``[pairings].driver``, proving the store is
swappable by a config edit exactly like every other port.

Unlike the SQLite driver's FTS5 triggers (kept in sync incrementally, inside the same transaction
as every write), DuckDB's ``fts`` extension builds a **separate index structure that must be
rebuilt** wholesale (``PRAGMA create_fts_index(..., overwrite=1)``). Rebuilding it inside every
:meth:`add` made a batched load quadratic — a 7M-row corpus at 5k per batch re-indexed the whole
growing table ~1,400 times. The rebuild is therefore **deferred**: a write only marks the index
stale, and :meth:`search` rebuilds first if it is. A bulk load followed by queries now pays one
rebuild instead of one per batch, while ``search`` still never observes a base-table row that is
missing from the index — the port's contract is unchanged, so no caller has to know. (An
alternating add/search workload still rebuilds per search; that is inherent to DuckDB's FTS
design, and the realistic import-then-query shape is what this fixes.)

:meth:`add` is all-or-nothing, like the SQLite driver's. DuckDB does not roll back a failed
``executemany`` on its own — each row lands or fails independently — so the batch is wrapped in an
explicit ``BEGIN``/``COMMIT``/``ROLLBACK``, which does.

``get``/``document``/``count``/``all_ids`` are unaffected by any of this: they read the base table
directly, never the search index.
"""
from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from ragkit.core.config import read_required_path
from ragkit.core.ports import Pairing

from ..lexical.bm25 import bm25_to_relevance
from .common import PairingStoreError, pairing_display

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pairings(
    chunk_id VARCHAR PRIMARY KEY,
    source VARCHAR NOT NULL,
    context VARCHAR NOT NULL DEFAULT '',
    target VARCHAR NOT NULL DEFAULT '',
    meta VARCHAR NOT NULL DEFAULT '{}',
    verified BOOLEAN NOT NULL DEFAULT false,
    created_at DOUBLE NOT NULL DEFAULT 0
)
"""


def _require_duckdb() -> Any:
    """Import duckdb lazily (so a sqlite-only run need not have it installed) and make sure the
    ``fts`` extension is loaded, installing it on first use if it is not already cached."""
    try:
        import duckdb
    except ImportError as exc:
        raise PairingStoreError(
            "the duckdb pairings driver needs the duckdb package; install it (pip install duckdb) "
            "or use the 'sqlite' driver") from exc
    return duckdb


def _load_fts(conn: Any) -> None:
    try:
        conn.execute("LOAD fts")
    except Exception:  # noqa: BLE001 -- broad: duckdb's own exception hierarchy, retried below
        try:
            conn.execute("INSTALL fts")
            conn.execute("LOAD fts")
        except Exception as exc:
            raise PairingStoreError(
                "the duckdb pairings driver needs the DuckDB 'fts' extension, which could not be "
                f"installed or loaded (offline, and not already cached?): {exc}. Run "
                "tools/setup_python_env.sh to pre-fetch it, or use the sqlite pairings driver "
                "instead.") from exc


class DuckDBPairings:
    """A :class:`~ragkit.core.ports.PairingStore` over DuckDB + its ``fts`` extension."""

    CONFIG_KEYS = frozenset({"path"})

    def __init__(self, path: str = ":memory:") -> None:
        self._lock = threading.Lock()
        # Starts stale rather than building here: an existing on-disk store is not re-indexed just
        # to be opened, and a caller that only reads rows (get/count/all_ids) never pays for an
        # index it does not use. The first search builds it.
        self._fts_stale = True
        duckdb = _require_duckdb()
        try:
            self._conn = duckdb.connect(path)
            _load_fts(self._conn)
            self._conn.execute(_SCHEMA)
        except PairingStoreError:
            raise
        except Exception as exc:
            raise PairingStoreError(f"could not open the pairing store at {path!r}: {exc}") from exc

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> DuckDBPairings:
        return cls(path=read_required_path(dict(options), "path", label="[pairings] store"))

    def _rebuild_fts(self) -> None:
        # stemmer/stopwords='none': the SQLite driver's FTS5 unicode61 tokenizer does neither, and
        # the two drivers must retrieve identically on a config swap (DuckDB's default 'porter'
        # stemmer + 'english' stopword list silently drops ordinary words -- 'hello' among them).
        self._conn.execute(
            "PRAGMA create_fts_index('pairings', 'chunk_id', 'source', 'context', 'target', "
            "stemmer='none', stopwords='none', overwrite=1)")

    def add(self, pairings: Iterable[Pairing]) -> int:
        rows = [(p.chunk_id, p.source, p.context, p.target,
                 json.dumps(dict(p.meta), ensure_ascii=False), bool(p.verified), p.created_at)
                for p in pairings]
        if not rows:
            return 0
        with self._lock:
            try:
                before = self._count_locked()
                # Explicit transaction: DuckDB does not roll back a failed executemany by itself,
                # so without this a mid-batch failure left a partial write -- and the returned
                # added-count described that partial result, breaking PairingStore.add's
                # "in one transaction" promise that the SQLite driver honours.
                self._conn.execute("BEGIN")
                self._conn.executemany(
                    "INSERT INTO pairings"
                    "(chunk_id, source, context, target, meta, verified, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING", rows)
                self._conn.execute("COMMIT")
            except Exception as exc:
                self._safe_rollback()
                raise PairingStoreError(f"could not add pairings: {exc}") from exc
            self._fts_stale = True  # rebuilt by the next search, not here -- see the docstring
            return self._count_locked() - before

    def _safe_rollback(self) -> None:
        """Roll back, unless the connection itself is unusable -- letting *that* failure replace
        the real one would mask the actual cause (the same guard the SQLite driver uses)."""
        with contextlib.suppress(Exception):
            self._conn.execute("ROLLBACK")

    def search(self, query: str, *, k: int) -> list[tuple[str, float]]:
        if k <= 0 or not query.strip():
            return []
        with self._lock:
            try:
                if self._fts_stale:
                    self._rebuild_fts()
                    self._fts_stale = False
                rows = self._conn.execute(
                    "SELECT chunk_id, score FROM ("
                    "  SELECT chunk_id, fts_main_pairings.match_bm25(chunk_id, ?) AS score "
                    "  FROM pairings"
                    ") sq WHERE score IS NOT NULL ORDER BY score DESC LIMIT ?",
                    [query, k]).fetchall()
            except Exception as exc:
                raise PairingStoreError(f"fts query failed: {exc}", query=query) from exc
        return [(chunk_id, bm25_to_relevance(score)) for chunk_id, score in rows]

    def document(self, chunk_id: str) -> tuple[str, Mapping[str, Any]] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT source, target, meta FROM pairings WHERE chunk_id = ?",
                [chunk_id]).fetchone()
        if row is None:
            return None
        source, target, meta = row
        return pairing_display(source, target), json.loads(meta)

    def get(self, chunk_id: str) -> Pairing | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT chunk_id, source, target, context, meta, verified, created_at "
                "FROM pairings WHERE chunk_id = ?", [chunk_id]).fetchone()
        if row is None:
            return None
        chunk_id_, source, target, context, meta, verified, created_at = row
        return Pairing(chunk_id=chunk_id_, source=source, target=target, context=context,
                       meta=json.loads(meta), verified=bool(verified), created_at=created_at)

    def all_ids(self) -> Iterator[str]:
        # fetchmany, not fetchall: the ids of a multi-million-row corpus must not all be resident
        # at once. A dedicated cursor (not self._conn.execute() directly) is required here: DuckDB's
        # connection.execute() returns the connection itself and reuses one shared result-set
        # state, so an unrelated self._conn.execute() call from another method while this generator
        # is paused between yields would silently clobber this iteration's position (verified: a
        # second execute() on the same connection redirects a still-open fetchmany() to its result
        # set). con.cursor() gives an independent result-set, immune to that. The lock is only ever
        # held for one fetch at a time (never across a yield), so a slow/lazy consumer cannot hold
        # this store's lock indefinitely and starve other threads.
        with self._lock:
            try:
                cursor = self._conn.cursor()
                cursor.execute("SELECT chunk_id FROM pairings ORDER BY chunk_id")
            except Exception as exc:
                raise PairingStoreError(f"could not list pairing ids: {exc}") from exc
        while True:
            with self._lock:
                try:
                    rows = cursor.fetchmany(1000)
                except Exception as exc:
                    raise PairingStoreError(f"could not list pairing ids: {exc}") from exc
            if not rows:
                return
            yield from (row[0] for row in rows)

    def _count_locked(self) -> int:
        return int(self._conn.execute("SELECT count(*) FROM pairings").fetchone()[0])

    def count(self) -> int:
        with self._lock:
            return self._count_locked()

    def close(self) -> None:
        self._conn.close()
