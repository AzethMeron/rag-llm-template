"""The storage layer: relational, vector, and lexical stores behind their ports, plus a schema
introspector, each resolved through a registry so a driver is swappable by config.

Two database roles are kept apart (see ``docs/architecture.md``): the framework's *own* store
(records, chunks, metadata — writable, SQLite by default) and an *external* task data source (the
database NL->SQL queries or form-autofill reads — read-only, never written by the framework). The
default vector database is LanceDB and the default lexical index is SQLite FTS5; both are real and
need no server, and both keep their optional third-party dependency confined to their driver
module so the core import path stays clean.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragkit.core.config import ConfigError, load_toml, reject_unknown
from ragkit.core.ports import LexicalIndex, SchemaIntrospector, SqlStore, VectorIndex
from ragkit.core.registry import Registry

from .filters import FilterError, to_sql
from .lexical.fts5 import Fts5Index, LexicalIndexError
from .sql.duckdb import DuckDBIntrospector, DuckDBStore
from .sql.sqlite import SqliteIntrospector, SqliteStore, SqlStoreError
from .vector.lancedb import LanceVectorIndex, VectorIndexError

SQL_STORES: Registry[SqlStore] = Registry(
    "sql store", SqlStore,  # type: ignore[type-abstract]
    entry_point_group="ragkit.sql_stores")
VECTOR_INDEXES: Registry[VectorIndex] = Registry(
    "vector index", VectorIndex,  # type: ignore[type-abstract]
    entry_point_group="ragkit.vector_indexes")
LEXICAL_INDEXES: Registry[LexicalIndex] = Registry(
    "lexical index", LexicalIndex,  # type: ignore[type-abstract]
    entry_point_group="ragkit.lexical_indexes")
SCHEMA_INTROSPECTORS: Registry[SchemaIntrospector] = Registry(
    "schema introspector", SchemaIntrospector,  # type: ignore[type-abstract]
    entry_point_group="ragkit.schema_introspectors")

SQL_STORES.register("sqlite", SqliteStore)
SQL_STORES.register("duckdb", DuckDBStore)
VECTOR_INDEXES.register("lancedb", LanceVectorIndex)
LEXICAL_INDEXES.register("fts5", Fts5Index)
SCHEMA_INTROSPECTORS.register("sqlite", SqliteIntrospector)
SCHEMA_INTROSPECTORS.register("duckdb", DuckDBIntrospector)


@dataclass(frozen=True, slots=True)
class Storage:
    """The stores a run assembles: any may be absent (a lexical-only run has no vector index; a
    run with no external data source has no introspector)."""

    sql: SqlStore | None = None
    vector: VectorIndex | None = None
    lexical: LexicalIndex | None = None
    introspector: SchemaIntrospector | None = None


def load_storage(path: Path, *, base_dir: Path | None = None) -> Storage:
    """Build the configured stores from a ``storage.toml``. Each ``[<port>]`` table names a
    ``driver`` and passes the rest of its keys as that driver's options (unknown keys refused by
    the driver's own schema). A relative ``path`` option is resolved against ``base_dir`` (the
    config directory), so a config is portable rather than tied to the caller's working directory;
    a ``:memory:`` path is left as-is."""
    data = load_toml(path, what="storage file")
    reject_unknown(data, {"sql", "vector", "lexical", "introspector"},
                   label="the storage file", path=path)
    base = base_dir or path.parent
    return Storage(
        sql=_build(SQL_STORES, data.get("sql"), label="[sql]", path=path, base=base),
        vector=_build(VECTOR_INDEXES, data.get("vector"), label="[vector]", path=path, base=base),
        lexical=_build(LEXICAL_INDEXES, data.get("lexical"), label="[lexical]", path=path,
                       base=base),
        introspector=_build(SCHEMA_INTROSPECTORS, data.get("introspector"),
                            label="[introspector]", path=path, base=base))


def _build(registry: Registry[Any], section: object, *, label: str, path: Path, base: Path) -> Any:
    if section is None:
        return None
    if not isinstance(section, dict):
        raise ConfigError(f"{label} must be a table", path=path)
    driver = section.get("driver")
    if not isinstance(driver, str) or not driver.strip():
        raise ConfigError(f"{label} needs a 'driver'", path=path)
    options = {k: v for k, v in section.items() if k != "driver"}
    raw_path = options.get("path")
    if isinstance(raw_path, str) and raw_path and raw_path != ":memory:" \
            and not Path(raw_path).is_absolute():
        options["path"] = str((base / raw_path).resolve())
    return registry.create(driver, options, path=path)


__all__ = [
    "Storage", "load_storage",
    "SQL_STORES", "VECTOR_INDEXES", "LEXICAL_INDEXES", "SCHEMA_INTROSPECTORS",
    "SqliteStore", "SqliteIntrospector", "DuckDBStore", "DuckDBIntrospector",
    "Fts5Index", "LanceVectorIndex",
    "SqlStoreError", "LexicalIndexError", "VectorIndexError", "FilterError", "to_sql",
]
