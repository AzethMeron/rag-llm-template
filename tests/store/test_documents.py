"""The relational chunk-row store (SqliteDocuments) that every retrieval path resolves through."""
from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.store.documents.sqlite import DocumentStoreError, SqliteDocuments


class TestDocuments:
    def test_add_and_resolve_round_trip(self) -> None:
        store = SqliteDocuments()
        store.add_documents([("a", "cat -> kot", {"n": 1}), ("b", "dog -> pies", {})])
        assert store.count() == 2
        assert store.document("a") == ("cat -> kot", {"n": 1})
        assert store.document("b") == ("dog -> pies", {})

    def test_unknown_id_resolves_to_none(self) -> None:
        assert SqliteDocuments().document("missing") is None

    def test_add_empty_is_a_noop(self) -> None:
        store = SqliteDocuments()
        store.add_documents([])
        assert store.count() == 0

    def test_reinsert_replaces(self) -> None:
        store = SqliteDocuments()
        store.add_documents([("a", "first", {})])
        store.add_documents([("a", "second", {"v": 2})])
        assert store.count() == 1
        assert store.document("a") == ("second", {"v": 2})

    def test_from_config_and_on_disk_persists(self, tmp_path: Path) -> None:
        db = str(tmp_path / "rows.db")
        first = SqliteDocuments.from_config({"path": db})
        first.add_documents([("a", "kept", {"k": True})])
        first.close()
        reopened = SqliteDocuments.from_config({"path": db})
        assert reopened.count() == 1 and reopened.document("a") == ("kept", {"k": True})

    def test_bad_path_is_a_structured_error(self, tmp_path: Path) -> None:
        # A path whose parent directory does not exist cannot open — a structured, catchable error.
        with pytest.raises(DocumentStoreError, match="document store"):
            SqliteDocuments(str(tmp_path / "no_such_dir" / "rows.db"))
