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
from ragkit.core.ports import (
    DocumentStore,
    LexicalIndex,
    LexiconStore,
    PairingStore,
    RunStore,
    SchemaIntrospector,
    SqlStore,
    VectorIndex,
)
from ragkit.core.registry import Registry

from .documents.sqlite import DocumentStoreError, SqliteDocuments
from .filters import FilterError, to_sql
from .lexical.fts5 import Fts5Index, LexicalIndexError
from .lexicon.sqlite import LexiconStoreError, SqliteLexicon
from .pairings.common import PairingStoreError
from .pairings.duckdb import DuckDBPairings
from .pairings.sqlite import SqlitePairings
from .run.sqlite import RunStoreError, SqliteRunStore
from .sql.duckdb import DuckDBIntrospector, DuckDBStore
from .sql.sqlite import SqliteIntrospector, SqliteStore, SqlStoreError
from .vector.lancedb import LanceVectorIndex, VectorIndexError
from .vector.qdrant import QdrantVectorIndex

SQL_STORES: Registry[SqlStore] = Registry(
    "sql store", SqlStore,  # type: ignore[type-abstract]
    entry_point_group="ragkit.sql_stores")
VECTOR_INDEXES: Registry[VectorIndex] = Registry(
    "vector index", VectorIndex,  # type: ignore[type-abstract]
    entry_point_group="ragkit.vector_indexes")
LEXICAL_INDEXES: Registry[LexicalIndex] = Registry(
    "lexical index", LexicalIndex,  # type: ignore[type-abstract]
    entry_point_group="ragkit.lexical_indexes")
DOCUMENT_STORES: Registry[DocumentStore] = Registry(
    "document store", DocumentStore,  # type: ignore[type-abstract]
    entry_point_group="ragkit.document_stores")
PAIRING_STORES: Registry[PairingStore] = Registry(
    "pairing store", PairingStore,  # type: ignore[type-abstract]
    entry_point_group="ragkit.pairing_stores")
RUN_STORES: Registry[RunStore] = Registry(
    "run store", RunStore,  # type: ignore[type-abstract]
    entry_point_group="ragkit.run_stores")
LEXICON_STORES: Registry[LexiconStore] = Registry(
    "lexicon store", LexiconStore,  # type: ignore[type-abstract]
    entry_point_group="ragkit.lexicon_stores")
SCHEMA_INTROSPECTORS: Registry[SchemaIntrospector] = Registry(
    "schema introspector", SchemaIntrospector,  # type: ignore[type-abstract]
    entry_point_group="ragkit.schema_introspectors")

SQL_STORES.register("sqlite", SqliteStore)
SQL_STORES.register("duckdb", DuckDBStore)
VECTOR_INDEXES.register("lancedb", LanceVectorIndex)
VECTOR_INDEXES.register("qdrant", QdrantVectorIndex)
LEXICAL_INDEXES.register("fts5", Fts5Index)
DOCUMENT_STORES.register("sqlite", SqliteDocuments)
PAIRING_STORES.register("sqlite", SqlitePairings)
PAIRING_STORES.register("duckdb", DuckDBPairings)
RUN_STORES.register("sqlite", SqliteRunStore)
LEXICON_STORES.register("sqlite", SqliteLexicon)
SCHEMA_INTROSPECTORS.register("sqlite", SqliteIntrospector)
SCHEMA_INTROSPECTORS.register("duckdb", DuckDBIntrospector)


@dataclass(frozen=True, slots=True)
class Storage:
    """The stores a run assembles: any may be absent (a lexical-only run has no vector index; a
    run with no external data source has no introspector). ``pairings`` is the DB-native reference
    memory (co-located rows + search index); ``lexical``/``documents`` are the legacy split-store
    pair, kept for recipes not yet migrated to ``[pairings]`` (see
    docs/storage-overhaul-plan.md)."""

    sql: SqlStore | None = None
    vector: VectorIndex | None = None
    lexical: LexicalIndex | None = None
    documents: DocumentStore | None = None
    pairings: PairingStore | None = None
    run: RunStore | None = None
    lexicon: LexiconStore | None = None
    introspector: SchemaIntrospector | None = None


def load_storage(path: Path, *, base_dir: Path | None = None) -> Storage:
    """Build the configured stores from a ``storage.toml``. Each ``[<port>]`` table names a
    ``driver`` and passes the rest of its keys as that driver's options (unknown keys refused by
    the driver's own schema). A relative ``path`` option is resolved against ``base_dir`` (the
    config directory), so a config is portable rather than tied to the caller's working directory;
    a ``:memory:`` path is left as-is."""
    data = load_toml(path, what="storage file")
    reject_unknown(data, {"sql", "vector", "lexical", "documents", "pairings", "run", "lexicon",
                         "introspector"}, label="the storage file", path=path)
    base = base_dir or path.parent
    return Storage(
        sql=_build(SQL_STORES, data.get("sql"), label="[sql]", path=path, base=base),
        vector=_build(VECTOR_INDEXES, data.get("vector"), label="[vector]", path=path, base=base),
        lexical=_build(LEXICAL_INDEXES, data.get("lexical"), label="[lexical]", path=path,
                       base=base),
        documents=_build(DOCUMENT_STORES, data.get("documents"), label="[documents]", path=path,
                         base=base),
        pairings=_build(PAIRING_STORES, data.get("pairings"), label="[pairings]", path=path,
                        base=base),
        run=_build(RUN_STORES, data.get("run"), label="[run]", path=path, base=base),
        lexicon=_build(LEXICON_STORES, data.get("lexicon"), label="[lexicon]", path=path,
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
    "SQL_STORES", "VECTOR_INDEXES", "LEXICAL_INDEXES", "DOCUMENT_STORES", "PAIRING_STORES",
    "RUN_STORES", "LEXICON_STORES", "SCHEMA_INTROSPECTORS",
    "SqliteStore", "SqliteIntrospector", "DuckDBStore", "DuckDBIntrospector",
    "Fts5Index", "LanceVectorIndex", "QdrantVectorIndex", "SqliteDocuments",
    "SqlitePairings", "DuckDBPairings", "SqliteRunStore", "SqliteLexicon",
    "SqlStoreError", "LexicalIndexError", "VectorIndexError", "DocumentStoreError",
    "PairingStoreError", "RunStoreError", "LexiconStoreError", "FilterError", "to_sql",
]
