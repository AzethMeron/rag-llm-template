"""SQLite-backed relational store and schema introspector — the zero-dependency default.

The framework's own record/metadata store, and the read side of an external task data source (the
database NL->SQL queries, or form-autofill reads), both sit behind the :class:`~ragkit.core.ports.
SqlStore` port. A binding is tagged ``read_only`` at construction: a write attempt through one is
refused **at the port**, before the database — the first line of the generated-SQL safety model,
not a reliance on database permissions alone.

Thread-safe: one connection shared by the runner's concurrent workers, guarded by a lock, opened
with ``check_same_thread=False``. Reads serialise, which is correct and simple for the moderate
concurrency here.
"""
from __future__ import annotations

import sqlite3
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ragkit.core.errors import RagkitError


class SqlStoreError(RagkitError):
    """A SQL store operation failed, or a write was attempted on a read-only binding.

    The two read-only refusals below live here as constructors so every :class:`~ragkit.core.ports.
    SqlStore` driver (SQLite, DuckDB, ...) raises the identical message from one home, rather than
    each re-typing it."""

    @classmethod
    def write_on_read_only(cls, sql: str) -> SqlStoreError:
        return cls(
            "write refused: this SqlStore binding is read_only. An external data source is never "
            "written by the framework; only the framework's own store is writable.", sql=sql)

    @classmethod
    def schema_on_read_only(cls) -> SqlStoreError:
        return cls("a read_only store cannot run schema_sql")

    @classmethod
    def memory_introspection(cls) -> SqlStoreError:
        # A ``:memory:`` database is per-connection and ephemeral, so an introspector -- which opens
        # its own connection to read the schema -- sees a *different*, empty in-memory database, not
        # whatever a store elsewhere in the process wrote. It would silently return ``{}`` (the same
        # silent-empty-schema failure the missing-file guard already refuses), so it is refused too.
        return cls("a schema introspector cannot read a ':memory:' database: it is per-connection "
                   "and ephemeral, so the introspector's own connection sees an empty database. "
                   "Point [introspector].path at the real database file to introspect.")


def _connect_read_only(path: str) -> sqlite3.Connection:
    """A read-only connection, which is also the only way a missing file raises rather than being
    silently created. ``:memory:`` cannot be opened read-only and has nothing to protect, so it is
    passed through — one home for the URI form, used by the read-only store and the introspector."""
    if path == ":memory:":
        return sqlite3.connect(path, check_same_thread=False)
    return sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True,
                           check_same_thread=False)


class SqliteStore:
    """A :class:`~ragkit.core.ports.SqlStore` over a SQLite database (a file, or ``:memory:``)."""

    CONFIG_KEYS = frozenset({"path", "read_only", "schema_sql"})

    def __init__(self, path: str = ":memory:", *, read_only: bool = False,
                 schema_sql: str | None = None) -> None:
        # A read-only store running schema_sql is a contradiction -- rejected before opening the
        # connection, so a bad config never orphans an open file handle (matching DuckDBStore) and
        # the error names the real mistake rather than a downstream open failure.
        if schema_sql and read_only:
            raise SqlStoreError.schema_on_read_only()
        self.read_only = read_only
        self._lock = threading.Lock()
        # A read-only binding opens the file in read-only mode via URI, so even a bug that slips a
        # write past the port cannot mutate the database. :memory: cannot be opened read-only, and
        # a read-only store over an ephemeral in-memory database is meaningless anyway.
        self._conn = (_connect_read_only(path) if read_only
                      else sqlite3.connect(path, check_same_thread=False))
        self._conn.row_factory = sqlite3.Row
        if schema_sql:
            with self._lock:
                self._conn.executescript(schema_sql)
                self._conn.commit()

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> SqliteStore:
        return cls(path=str(options.get("path", ":memory:")),
                   read_only=bool(options.get("read_only", False)),
                   schema_sql=options.get("schema_sql"))

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[Mapping[str, Any]]:
        with self._lock:
            try:
                cursor = self._conn.execute(sql, tuple(params))
            except sqlite3.Error as exc:
                raise SqlStoreError(f"query failed: {exc}", sql=sql) from exc
            return [dict(row) for row in cursor.fetchall()]

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        if self.read_only:
            raise SqlStoreError.write_on_read_only(sql)
        with self._lock:
            try:
                self._conn.execute(sql, tuple(params))
                self._conn.commit()
            except sqlite3.Error as exc:
                raise SqlStoreError(f"execute failed: {exc}", sql=sql) from exc

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SqliteStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class SqliteIntrospector:
    """A :class:`~ragkit.core.ports.SchemaIntrospector` over a SQLite database, using ``PRAGMA
    table_info`` — no SQLAlchemy dependency, so the NL->SQL feature reads a schema without pulling
    a heavier stack. Returns each table's ordered ``(column, declared_type)`` pairs."""

    CONFIG_KEYS = frozenset({"path"})

    def __init__(self, path: str) -> None:
        self._path = path

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> SqliteIntrospector:
        return cls(path=str(options.get("path", ":memory:")))

    def schema(self) -> Mapping[str, Sequence[tuple[str, str]]]:
        """The database's tables and their columns.

        Opened **read-only**, which is also what makes a missing file an error. A plain
        ``sqlite3.connect`` creates the database it cannot find, so a typo'd ``[introspector].path``
        used to produce an empty file and return ``{}`` — no error anywhere, and NL->SQL then
        generated against an empty schema. The DuckDB introspector already failed loudly on the
        same misconfiguration; the two now agree. A ``:memory:`` path is refused for the same
        silent-empty reason (see :meth:`SqlStoreError.memory_introspection`).
        """
        if self._path == ":memory:":
            raise SqlStoreError.memory_introspection()
        conn = _connect_read_only(self._path)
        conn.row_factory = sqlite3.Row
        try:
            tables = [row["name"] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name")]
            result: dict[str, list[tuple[str, str]]] = {}
            for table in tables:
                # PRAGMA does not accept a bound parameter for the table name; the name comes from
                # sqlite_master (not user input) and is quote-escaped defensively.
                safe = table.replace('"', '""')
                columns = conn.execute(f'PRAGMA table_info("{safe}")').fetchall()
                result[table] = [(c["name"], c["type"] or "") for c in columns]
            return result
        finally:
            conn.close()
