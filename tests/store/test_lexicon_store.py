"""The SqliteLexicon driver: lifecycle, upsert-on-duplicate, and compatibility with the existing
JSONL-sourced relevant_entries()/LexiconBlock consumers."""
from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.core.lexicon import Entry, relevant_entries
from ragkit.store.lexicon.sqlite import LexiconStoreError, SqliteLexicon


class TestLifecycle:
    def test_add_and_list_entries(self) -> None:
        store = SqliteLexicon()
        added = store.add([Entry(term="cat", rendering="kot"),
                           Entry(term="dog", rendering="pies")])
        assert added == 2
        assert {e.term for e in store.entries()} == {"cat", "dog"}

    def test_add_empty_is_a_noop(self) -> None:
        assert SqliteLexicon().add([]) == 0

    def test_upsert_replaces_the_rendering_not_a_duplicate(self) -> None:
        store = SqliteLexicon()
        store.add([Entry(term="cat", rendering="kot")])
        added_again = store.add([Entry(term="cat", rendering="KOTEK")])
        assert added_again == 0  # a replace, not a new row
        [entry] = store.entries()
        assert entry.rendering == "KOTEK"

    def test_category_distinguishes_the_same_term(self) -> None:
        store = SqliteLexicon()
        store.add([Entry(term="rock", rendering="genre", category="music"),
                   Entry(term="rock", rendering="stone", category="geology")])
        assert len(store.entries()) == 2

    def test_entity_id_round_trips(self) -> None:
        store = SqliteLexicon()
        store.add([Entry(term="cat", rendering="kot", entity_id=42)])
        [entry] = store.entries()
        assert entry.entity_id == 42

    def test_from_config(self, tmp_path: Path) -> None:
        store = SqliteLexicon.from_config({"path": str(tmp_path / "lex.db")})
        store.add([Entry(term="cat", rendering="kot")])
        assert len(store.entries()) == 1

    def test_in_memory_default(self) -> None:
        store = SqliteLexicon()
        store.add([Entry(term="cat", rendering="kot")])
        assert len(store.entries()) == 1

    def test_persists_on_disk_across_reopen(self, tmp_path: Path) -> None:
        path = str(tmp_path / "lex.db")
        first = SqliteLexicon(path)
        first.add([Entry(term="cat", rendering="kot")])
        first.close()
        reopened = SqliteLexicon(path)
        assert len(reopened.entries()) == 1


class TestCompat:
    def test_relevant_entries_works_over_a_store_backed_list(self) -> None:
        store = SqliteLexicon()
        store.add([Entry(term="cat", rendering="kot"), Entry(term="dog", rendering="pies")])
        hits = relevant_entries("the cat sat", store.entries())
        assert [e.term for e in hits] == ["cat"]


class TestConcurrency:
    """WAL + busy_timeout: a concurrent reader must not be locked out by a writer's in-flight
    transaction."""

    def test_journal_mode_is_wal(self, tmp_path: Path) -> None:
        store = SqliteLexicon(str(tmp_path / "lex.db"))
        assert store._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    def test_busy_timeout_is_set(self) -> None:
        store = SqliteLexicon()
        assert store._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000

    def test_a_reader_is_not_locked_out_by_an_open_writer(self, tmp_path: Path) -> None:
        import sqlite3

        path = str(tmp_path / "lex.db")
        store = SqliteLexicon(path)
        store.add([Entry(term="cat", rendering="kot")])

        writer = sqlite3.connect(path)
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("INSERT INTO lexicon(term, rendering, category, entity_id) "
                       "VALUES ('dog', 'pies', '', 0)")
        try:
            assert len(store.entries()) == 1
        finally:
            writer.rollback()
            writer.close()


class TestErrors:
    def test_bad_path_is_a_structured_error(self, tmp_path: Path) -> None:
        with pytest.raises(LexiconStoreError, match="could not open the lexicon store"):
            SqliteLexicon(str(tmp_path / "no_such_dir" / "lex.db"))

    def test_add_on_a_closed_store_is_structured(self) -> None:
        store = SqliteLexicon()
        store.add([Entry(term="cat", rendering="kot")])
        store.close()
        with pytest.raises(LexiconStoreError, match="could not add entries"):
            store.add([Entry(term="dog", rendering="pies")])
