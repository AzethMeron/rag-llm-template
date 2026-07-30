"""The legacy-artifact migrations: documents->pairings (resumable, ignores the old search index),
lexicon.jsonl->LexiconStore, and catalog+journal->RunStore."""
from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.core.lexicon import Entry, write_lexicon
from ragkit.core.records import Record, Status, write_catalog
from ragkit.store.documents.sqlite import SqliteDocuments
from ragkit.store.lexicon.sqlite import SqliteLexicon
from ragkit.store.migrate import (
    migrate_documents_to_pairings,
    migrate_lexicon_to_store,
    migrate_run_to_store,
)
from ragkit.store.pairings.sqlite import SqlitePairings
from ragkit.store.run.sqlite import SqliteRunStore


class TestMigrateDocumentsToPairings:
    def _old_store(self, tmp_path: Path, rows: list[tuple[str, str, dict]], *,
                   name: str = "old.docs.db") -> Path:
        path = tmp_path / name
        docs = SqliteDocuments(str(path))
        docs.add_documents(rows)
        docs.close()
        return path

    def test_migrates_all_rows(self, tmp_path: Path) -> None:
        old = self._old_store(tmp_path, [
            ("ref-1", "the quick brown fox", {"n": 1}),
            ("ref-2", "a lazy dog", {"n": 2}),
        ])
        pairings = SqlitePairings(str(tmp_path / "new.pairings.db"))
        added = migrate_documents_to_pairings(old, pairings)
        assert added == 2
        assert pairings.count() == 2
        assert pairings.document("ref-1") == ("the quick brown fox", {"n": 1})

    def test_the_migrated_search_index_actually_works(self, tmp_path: Path) -> None:
        old = self._old_store(tmp_path, [("ref-1", "the quick brown fox", {}),
                                         ("ref-2", "a lazy dog", {})])
        pairings = SqlitePairings(str(tmp_path / "new.pairings.db"))
        migrate_documents_to_pairings(old, pairings)
        results = pairings.search("quick fox", k=5)
        assert [chunk_id for chunk_id, _score in results] == ["ref-1"]

    def test_empty_old_store_migrates_nothing(self, tmp_path: Path) -> None:
        old = self._old_store(tmp_path, [])
        pairings = SqlitePairings(str(tmp_path / "new.pairings.db"))
        assert migrate_documents_to_pairings(old, pairings) == 0

    def test_resumes_from_the_durable_floor(self, tmp_path: Path) -> None:
        old = self._old_store(tmp_path, [(f"ref-{i}", f"passage {i}", {}) for i in range(1, 6)])
        pairings = SqlitePairings(str(tmp_path / "new.pairings.db"))
        migrate_documents_to_pairings(old, pairings, batch_size=2)
        assert pairings.count() == 5
        # Re-running (e.g. after an interruption) is idempotent, not a restart.
        added_again = migrate_documents_to_pairings(old, pairings, batch_size=2)
        assert added_again == 0 and pairings.count() == 5

    def test_streams_in_batches(self, tmp_path: Path) -> None:
        old = self._old_store(tmp_path, [(f"ref-{i}", f"passage {i}", {}) for i in range(1, 8)])
        pairings = SqlitePairings(str(tmp_path / "new.pairings.db"))
        added = migrate_documents_to_pairings(old, pairings, batch_size=3)
        assert added == 7 and pairings.count() == 7

    def test_bad_batch_size_is_refused(self, tmp_path: Path) -> None:
        old = self._old_store(tmp_path, [])
        pairings = SqlitePairings()
        with pytest.raises(ValueError, match="batch_size"):
            migrate_documents_to_pairings(old, pairings, batch_size=0)

    def test_on_batch_reports_the_destinations_running_total(self, tmp_path: Path) -> None:
        old = self._old_store(tmp_path, [(f"ref-{i}", f"passage {i}", {}) for i in range(1, 6)])
        pairings = SqlitePairings(str(tmp_path / "new.pairings.db"))
        seen: list[int] = []
        migrate_documents_to_pairings(old, pairings, batch_size=2, on_batch=seen.append)
        # Two full batches of 2, then one partial batch of 1 -- on_batch fires for each, including
        # the trailing partial one, and reports the store's running total, not a per-batch delta.
        assert seen == [2, 4, 5]

    def test_on_batch_reflects_a_nonzero_resume_floor(self, tmp_path: Path) -> None:
        old = self._old_store(tmp_path, [(f"ref-{i}", f"passage {i}", {}) for i in range(1, 4)])
        pairings = SqlitePairings(str(tmp_path / "new.pairings.db"))
        migrate_documents_to_pairings(old, pairings)  # floor becomes 3
        old2 = self._old_store(tmp_path, [(f"ref-{i}", f"passage {i}", {})
                                          for i in range(1, 6)], name="old2.docs.db")
        seen: list[int] = []
        migrate_documents_to_pairings(old2, pairings, on_batch=seen.append)
        assert seen == [5]

    def test_old_store_is_opened_read_only(self, tmp_path: Path) -> None:
        # Never write through the connection this migration opens against the legacy artifact.
        import sqlite3

        old = self._old_store(tmp_path, [("ref-1", "x", {})])
        pairings = SqlitePairings()
        migrate_documents_to_pairings(old, pairings)
        conn = sqlite3.connect(f"file:{old.resolve()}?mode=ro", uri=True)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM docs")
        conn.close()


class TestMigrateLexiconToStore:
    def test_migrates_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "lexicon.jsonl"
        write_lexicon([Entry(term="cat", rendering="kot"), Entry(term="dog", rendering="pies")],
                      path)
        store = SqliteLexicon()
        added = migrate_lexicon_to_store(path, store)
        assert added == 2
        assert {e.term for e in store.entries()} == {"cat", "dog"}

    def test_missing_file_is_a_noop(self, tmp_path: Path) -> None:
        store = SqliteLexicon()
        assert migrate_lexicon_to_store(tmp_path / "absent.jsonl", store) == 0
        assert store.entries() == []


class TestMigrateRunToStore:
    def test_adds_records_and_replays_results(self, tmp_path: Path) -> None:
        catalog = tmp_path / "c.jsonl"
        write_catalog([Record(record_id="1", source="a"), Record(record_id="2", source="b")],
                      catalog)
        journal = tmp_path / "j.jsonl"
        journal.write_text(
            Record(record_id="1", source="a", status=Status.VERIFIED, output="A").to_json()
            + "\n", encoding="utf-8")
        store = SqliteRunStore()
        added = migrate_run_to_store(catalog, journal, store)
        assert added == 2
        assert store.completed_ids() == {"1"}
        assert {r.record_id for r in store.pending()} == {"2"}
        [result] = list(store.results())
        assert result.record.output == "A"

    def test_later_journal_entry_wins(self, tmp_path: Path) -> None:
        catalog = tmp_path / "c.jsonl"
        write_catalog([Record(record_id="1", source="a")], catalog)
        journal = tmp_path / "j.jsonl"
        journal.write_text(
            Record(record_id="1", source="a", status=Status.PRODUCED, output="first").to_json()
            + "\n" +
            Record(record_id="1", source="a", status=Status.VERIFIED, output="second").to_json()
            + "\n", encoding="utf-8")
        store = SqliteRunStore()
        migrate_run_to_store(catalog, journal, store)
        [result] = list(store.results())
        assert result.record.output == "second"

    def test_empty_journal_leaves_everything_pending(self, tmp_path: Path) -> None:
        catalog = tmp_path / "c.jsonl"
        write_catalog([Record(record_id="1", source="a")], catalog)
        store = SqliteRunStore()
        added = migrate_run_to_store(catalog, tmp_path / "absent_journal.jsonl", store)
        assert added == 1
        assert store.completed_ids() == set()
