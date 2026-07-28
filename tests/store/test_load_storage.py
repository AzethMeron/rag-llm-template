"""Building the configured stores from storage.toml, and the swap property."""
from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.core.config import ConfigError
from ragkit.store import (
    LEXICAL_INDEXES,
    SQL_STORES,
    VECTOR_INDEXES,
    Fts5Index,
    LanceVectorIndex,
    SqliteStore,
    load_storage,
)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "storage.toml"
    path.write_text(text, encoding="utf-8")
    return path


class TestLoad:
    def test_builds_all_stores(self, tmp_path: Path) -> None:
        text = f"""
[sql]
driver = "sqlite"
path = ":memory:"

[vector]
driver = "lancedb"
path = "{(tmp_path / 'v.lance').as_posix()}"
dim = 8

[lexical]
driver = "fts5"

[introspector]
driver = "sqlite"
path = ":memory:"
"""
        storage = load_storage(_write(tmp_path, text))
        assert isinstance(storage.sql, SqliteStore)
        assert isinstance(storage.vector, LanceVectorIndex)
        assert isinstance(storage.lexical, Fts5Index)
        assert storage.introspector is not None

    def test_absent_sections_are_none(self, tmp_path: Path) -> None:
        storage = load_storage(_write(tmp_path, '[lexical]\ndriver = "fts5"\n'))
        assert storage.sql is None and storage.vector is None
        assert isinstance(storage.lexical, Fts5Index)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="storage file not found"):
            load_storage(tmp_path / "absent.toml")

    def test_unknown_section(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="unknown key"):
            load_storage(_write(tmp_path, "[bogus]\n"))

    def test_missing_driver(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="needs a 'driver'"):
            load_storage(_write(tmp_path, "[sql]\npath = ':memory:'\n"))

    def test_unknown_driver_option_is_refused(self, tmp_path: Path) -> None:
        text = '[sql]\ndriver = "sqlite"\nbogus = 1\n'
        with pytest.raises(ConfigError, match="unknown key"):
            load_storage(_write(tmp_path, text))

    def test_unknown_driver_name(self, tmp_path: Path) -> None:
        text = '[vector]\ndriver = "pinecone"\n'
        from ragkit.core.registry import RegistryError
        with pytest.raises(RegistryError, match="unknown vector index"):
            load_storage(_write(tmp_path, text))

    def test_non_table_section_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="must be a table"):
            load_storage(_write(tmp_path, "sql = 5\n"))


class TestRegistries:
    def test_builtins_are_registered(self) -> None:
        assert "sqlite" in SQL_STORES.available()
        assert "lancedb" in VECTOR_INDEXES.available()
        assert "fts5" in LEXICAL_INDEXES.available()

    def test_a_custom_driver_resolves_by_dotted_path(self, tmp_path: Path) -> None:
        # The swap property: a third-party driver is selected by dotted path, no framework change.
        text = ('[vector]\n'
                'driver = "tests.store.test_load_storage:_FakeVectorIndex"\n')
        storage = load_storage(_write(tmp_path, text))
        assert isinstance(storage.vector, _FakeVectorIndex)


class _FakeVectorIndex:
    """A minimal third-party VectorIndex, resolvable by dotted path (proves the extension API)."""

    CONFIG_KEYS = frozenset()

    def upsert(self, ids, vectors, metas) -> None:  # type: ignore[no-untyped-def]
        return None

    def search(self, vector, *, k, where=()):  # type: ignore[no-untyped-def]
        return []

    def delete(self, ids) -> None:  # type: ignore[no-untyped-def]
        return None

    def count(self) -> int:
        return 0

    def reconcile(self, chunk_ids) -> set:  # type: ignore[no-untyped-def]
        return set()
