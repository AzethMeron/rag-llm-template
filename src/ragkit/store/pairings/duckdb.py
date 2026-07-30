"""The pairing store over DuckDB — a second real ``[pairings].driver``, proving the store is
swappable by a config edit exactly like every other port.

Unlike the SQLite driver's FTS5 triggers (kept in sync incrementally, inside the same transaction
as every write), DuckDB's ``fts`` extension builds a **separate index structure that must be
rebuilt** after a write (``PRAGMA create_fts_index(..., overwrite=1)``); this driver rebuilds it on
every :meth:`add`, so ``search`` never observes a base-table row not yet in the index. It is
correctness-equivalent but **not** the SQLite driver's atomicity guarantee: DuckDB does not roll
back a whole ``executemany`` batch when one row in it fails (each row lands or fails on its own),
so a partial batch can leave a partial write — documented here rather than hidden, and covered by
its own test rather than the conformance suite (which only asserts what every driver honours).
``get``/``document``/``count``/``all_ids`` are unaffected either way: they read the base table
directly, never the search index.
"""
from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from ragkit.core.ports import Pairing

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


def _bm25_to_relevance(score: float) -> float:
    """DuckDB's ``match_bm25`` is already higher-is-better but unbounded; ``score / (1 + score)``
    is a monotone, order-preserving map into ``[0, 1)`` (score is always >= 0 for a match)."""
    return score / (1.0 + score)


class DuckDBPairings:
    """A :class:`~ragkit.core.ports.PairingStore` over DuckDB + its ``fts`` extension."""

    CONFIG_KEYS = frozenset({"path"})

    def __init__(self, path: str = ":memory:") -> None:
        self._lock = threading.Lock()
        duckdb = _require_duckdb()
        try:
            self._conn = duckdb.connect(path)
            _load_fts(self._conn)
            self._conn.execute(_SCHEMA)
            self._rebuild_fts()
        except PairingStoreError:
            raise
        except Exception as exc:
            raise PairingStoreError(f"could not open the pairing store at {path!r}: {exc}") from exc

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> DuckDBPairings:
        return cls(path=str(options.get("path", ":memory:")))

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
                self._conn.executemany(
                    "INSERT INTO pairings"
                    "(chunk_id, source, context, target, meta, verified, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING", rows)
            except Exception as exc:
                # Rebuilt even after a partial-batch failure, so search reflects whatever the base
                # table actually holds rather than lagging it (see the module docstring) -- but a
                # rebuild failure here (e.g. the connection is also closed) must not replace and
                # mask the add failure actually being reported.
                self._safe_rebuild_fts()
                raise PairingStoreError(f"could not add pairings: {exc}") from exc
            try:
                self._rebuild_fts()
                return self._count_locked() - before
            except Exception as exc:
                raise PairingStoreError(f"could not add pairings: {exc}") from exc

    def _safe_rebuild_fts(self) -> None:
        # An add() failure is already being reported; a rebuild failure here must not replace it.
        with contextlib.suppress(Exception):
            self._rebuild_fts()

    def search(self, query: str, *, k: int) -> list[tuple[str, float]]:
        if k <= 0 or not query.strip():
            return []
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT chunk_id, score FROM ("
                    "  SELECT chunk_id, fts_main_pairings.match_bm25(chunk_id, ?) AS score "
                    "  FROM pairings"
                    ") sq WHERE score IS NOT NULL ORDER BY score DESC LIMIT ?",
                    [query, k]).fetchall()
            except Exception as exc:
                raise PairingStoreError(f"fts query failed: {exc}", query=query) from exc
        return [(chunk_id, _bm25_to_relevance(score)) for chunk_id, score in rows]

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
