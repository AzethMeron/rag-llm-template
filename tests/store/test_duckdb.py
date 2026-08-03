"""DuckDB SqlStore + introspector: the branches the shared conformance suite does not reach
(from_config, the read-only/schema_sql guard, :memory:, the introspector, error normalisation)."""
from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.store.sql.duckdb import DuckDBIntrospector, DuckDBStore
from ragkit.store.sql.sqlite import SqlStoreError

SCHEMA = ("CREATE TABLE singer(id INTEGER, name VARCHAR, country VARCHAR);"
          "INSERT INTO singer VALUES (1, 'Ada', 'PL');")


class TestStore:
    def test_in_memory_query_and_execute(self) -> None:
        store = DuckDBStore(schema_sql=SCHEMA)  # :memory: default
        assert store.query("SELECT name FROM singer")[0]["name"] == "Ada"
        store.execute("INSERT INTO singer VALUES (2, 'Bo', 'US')")
        assert len(store.query("SELECT * FROM singer")) == 2
        store.close()

    def test_from_config(self, tmp_path: Path) -> None:
        store = DuckDBStore.from_config(
            {"path": str(tmp_path / "d.duckdb"), "read_only": False, "schema_sql": SCHEMA})
        assert store.read_only is False
        assert store.query("SELECT count(*) AS n FROM singer")[0]["n"] == 1
        store.close()

    def test_read_only_cannot_run_schema_sql(self, tmp_path: Path) -> None:
        with pytest.raises(SqlStoreError, match="read_only store cannot run schema_sql"):
            DuckDBStore(str(tmp_path / "d.duckdb"), read_only=True, schema_sql=SCHEMA)

    def test_read_only_write_refused_at_the_port(self, tmp_path: Path) -> None:
        path = str(tmp_path / "d.duckdb")
        DuckDBStore(path, schema_sql=SCHEMA).close()
        readonly = DuckDBStore(path, read_only=True)
        with pytest.raises(SqlStoreError, match="read_only"):
            readonly.execute("DELETE FROM singer")
        readonly.close()

    def test_execute_error_is_structured(self) -> None:
        store = DuckDBStore(schema_sql=SCHEMA)
        with pytest.raises(SqlStoreError, match="execute failed"):
            store.execute("INSERT INTO nope VALUES (1)")
        store.close()

    def test_context_manager_closes(self) -> None:
        with DuckDBStore(schema_sql=SCHEMA) as store:
            assert store.query("SELECT 1 AS one")[0]["one"] == 1


class TestIntrospector:
    def test_reads_tables_and_columns(self, tmp_path: Path) -> None:
        path = str(tmp_path / "d.duckdb")
        DuckDBStore(path, schema_sql=SCHEMA).close()
        schema = DuckDBIntrospector(path).schema()
        assert set(schema) == {"singer"}
        names = [name for name, _type in schema["singer"]]
        assert names == ["id", "name", "country"]

    def test_from_config(self, tmp_path: Path) -> None:
        path = str(tmp_path / "d.duckdb")
        DuckDBStore(path, schema_sql=SCHEMA).close()
        introspector = DuckDBIntrospector.from_config({"path": path})
        assert "singer" in introspector.schema()

    def test_memory_path_is_refused(self) -> None:
        # A :memory: introspector opens its own empty per-connection database, so it would silently
        # return {}; refused loudly, matching the SQLite introspector and the missing-file guard.
        with pytest.raises(SqlStoreError, match="per-connection"):
            DuckDBIntrospector(":memory:").schema()
        with pytest.raises(SqlStoreError, match="per-connection"):
            DuckDBIntrospector.from_config({}).schema()
