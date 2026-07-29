"""DuckDB-backed relational store and schema introspector — a second real ``SqlStore`` driver,
proving a database is swappable by a config edit (``driver = "sqlite"`` -> ``"duckdb"``) with no
change to any other part of the code.

DuckDB is an embedded, in-process analytical SQL database (a single self-contained wheel, no server,
MIT-licensed), so it keeps the zero-server, self-contained property the SQLite default has while
being a genuinely different engine. Everything about the port is honoured identically: a binding is
tagged ``read_only`` at construction and a write through one is refused **at the port**; the driver
returns rows as plain dicts and normalises any error into the same ``SqlStoreError`` the SQLite
driver raises, so a caller cannot tell which engine raised.

Thread-safe like the SQLite driver: one connection shared by the runner's concurrent workers,
guarded by a lock. DuckDB's own connection is not safe for concurrent use, so reads serialise —
correct and simple for the moderate concurrency here.
"""
from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .sqlite import SqlStoreError


def _require_duckdb() -> Any:
    """Import duckdb lazily, so the core/store import path never pulls it and a run that uses only
    SQLite need not have it installed."""
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - exercised only where duckdb is absent
        raise SqlStoreError(
            "the 'duckdb' driver needs the duckdb package; install it (pip install duckdb) or use "
            "the 'sqlite' driver") from exc
    return duckdb


class DuckDBStore:
    """A :class:`~ragkit.core.ports.SqlStore` over a DuckDB database (a file, or ``:memory:``)."""

    CONFIG_KEYS = frozenset({"path", "read_only", "schema_sql"})

    def __init__(self, path: str = ":memory:", *, read_only: bool = False,
                 schema_sql: str | None = None) -> None:
        # A read-only store creating schema is a contradiction -- caught before opening the
        # connection, so the error names the real mistake rather than a downstream open failure.
        if schema_sql and read_only:
            raise SqlStoreError.schema_on_read_only()
        self.read_only = read_only
        self._lock = threading.Lock()
        duckdb = _require_duckdb()
        # A read-only binding opens the database read-only, so even a bug that slips a write past
        # the port cannot mutate it. :memory: cannot be opened read-only (it is ephemeral and
        # per-connection), matching the SQLite driver's rule.
        if read_only and path != ":memory:":
            self._conn = duckdb.connect(str(Path(path).resolve()), read_only=True)
        else:
            self._conn = duckdb.connect(path)
        if schema_sql:
            with self._lock:
                self._conn.execute(schema_sql)

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> DuckDBStore:
        return cls(path=str(options.get("path", ":memory:")),
                   read_only=bool(options.get("read_only", False)),
                   schema_sql=options.get("schema_sql"))

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[Mapping[str, Any]]:
        with self._lock:
            try:
                cursor = self._conn.execute(sql, list(params))
            except Exception as exc:  # normalise any duckdb error into the port's SqlStoreError
                raise SqlStoreError(f"query failed: {exc}", sql=sql) from exc
            columns = [d[0] for d in cursor.description or ()]
            return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        if self.read_only:
            raise SqlStoreError.write_on_read_only(sql)
        with self._lock:
            try:
                self._conn.execute(sql, list(params))
            except Exception as exc:  # normalise any duckdb error into the port's SqlStoreError
                raise SqlStoreError(f"execute failed: {exc}", sql=sql) from exc

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> DuckDBStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class DuckDBIntrospector:
    """A :class:`~ragkit.core.ports.SchemaIntrospector` over a DuckDB database, using
    ``information_schema`` — so the NL->SQL feature can read a DuckDB schema exactly as it reads a
    SQLite one. Returns each table's ordered ``(column, declared_type)`` pairs."""

    CONFIG_KEYS = frozenset({"path"})

    def __init__(self, path: str) -> None:
        self._path = path

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> DuckDBIntrospector:
        return cls(path=str(options.get("path", ":memory:")))

    def schema(self) -> Mapping[str, Sequence[tuple[str, str]]]:
        duckdb = _require_duckdb()
        # Introspection only reads. Open the file read-only so it is compatible with a read-only
        # store already holding the same database: DuckDB refuses two connections to one file with
        # different read_only settings. (:memory: cannot open read-only; it is per-connection.)
        in_memory = self._path == ":memory:"
        conn = duckdb.connect(":memory:" if in_memory else str(Path(self._path).resolve()),
                              read_only=not in_memory)
        try:
            tables = [row[0] for row in conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' ORDER BY table_name").fetchall()]
            result: dict[str, list[tuple[str, str]]] = {}
            for table in tables:
                columns = conn.execute(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_name = ? ORDER BY ordinal_position", [table]).fetchall()
                result[table] = [(str(name), str(dtype or "")) for name, dtype in columns]
            return result
        finally:
            conn.close()
