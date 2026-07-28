"""The SQLite store, its read-only guard, and the schema introspector."""
from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.store.sql.sqlite import SqliteIntrospector, SqliteStore, SqlStoreError

SCHEMA = ("CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT NOT NULL);"
          "INSERT INTO customers(id, name) VALUES (1, 'Ann'), (2, 'Bob');")


class TestSqliteStore:
    def test_query_returns_dicts(self) -> None:
        store = SqliteStore(schema_sql=SCHEMA)
        rows = store.query("SELECT name FROM customers ORDER BY id")
        assert rows == [{"name": "Ann"}, {"name": "Bob"}]

    def test_parameterised_query(self) -> None:
        store = SqliteStore(schema_sql=SCHEMA)
        assert store.query("SELECT name FROM customers WHERE id = ?", [2]) == [{"name": "Bob"}]

    def test_execute_writes(self) -> None:
        store = SqliteStore(schema_sql=SCHEMA)
        store.execute("INSERT INTO customers(id, name) VALUES (3, 'Cy')")
        assert store.query("SELECT count(*) AS n FROM customers")[0]["n"] == 3

    def test_read_only_refuses_writes(self) -> None:
        store = SqliteStore(read_only=True)
        with pytest.raises(SqlStoreError, match="read_only"):
            store.execute("CREATE TABLE t(a)")

    def test_read_only_cannot_run_schema_sql(self) -> None:
        with pytest.raises(SqlStoreError, match="cannot run schema_sql"):
            SqliteStore(read_only=True, schema_sql="CREATE TABLE t(a)")

    def test_read_only_file_binding_cannot_be_written(self, tmp_path: Path) -> None:
        db = tmp_path / "d.db"
        SqliteStore(str(db), schema_sql=SCHEMA).close()
        # Even the port guard aside, the underlying connection is opened mode=ro.
        ro = SqliteStore(str(db), read_only=True)
        assert ro.query("SELECT count(*) AS n FROM customers")[0]["n"] == 2
        with pytest.raises(SqlStoreError):
            ro.execute("INSERT INTO customers(id, name) VALUES (9, 'X')")

    def test_bad_sql_is_a_structured_error(self) -> None:
        store = SqliteStore(schema_sql=SCHEMA)
        with pytest.raises(SqlStoreError, match="query failed"):
            store.query("SELECT * FROM no_such_table")

    def test_bad_execute_is_structured(self) -> None:
        with pytest.raises(SqlStoreError, match="execute failed"):
            SqliteStore().execute("NOT SQL")

    def test_from_config_and_context_manager(self, tmp_path: Path) -> None:
        with SqliteStore.from_config({"path": ":memory:", "read_only": False}) as store:
            assert store.read_only is False

    def test_read_only_flag_exposed(self) -> None:
        assert SqliteStore(read_only=True).read_only is True


class TestIntrospector:
    def test_reads_table_columns(self, tmp_path: Path) -> None:
        db = tmp_path / "d.db"
        SqliteStore(str(db), schema_sql=SCHEMA).close()
        schema = SqliteIntrospector(str(db)).schema()
        assert set(schema) == {"customers"}
        assert schema["customers"] == [("id", "INTEGER"), ("name", "TEXT")]

    def test_skips_sqlite_internal_tables(self, tmp_path: Path) -> None:
        db = tmp_path / "d.db"
        schema = "CREATE TABLE t(a INTEGER PRIMARY KEY AUTOINCREMENT)"
        SqliteStore(str(db), schema_sql=schema).close()
        # AUTOINCREMENT creates sqlite_sequence; it must not appear in the schema.
        assert "sqlite_sequence" not in SqliteIntrospector(str(db)).schema()

    def test_from_config(self) -> None:
        assert SqliteIntrospector.from_config({"path": ":memory:"}).schema() == {}
