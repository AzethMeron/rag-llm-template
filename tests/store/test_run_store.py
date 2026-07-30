"""The SqliteRunStore driver: the record catalogue, the append-only result history, crash-safety
(WAL kill-and-reopen), the FK orphan guard, later-wins via seq, and the capture round-trip."""
from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.core.ports import RetrievedRef, RunResult
from ragkit.core.records import Record, Status
from ragkit.core.rules import Severity, Violation
from ragkit.store.run.sqlite import RunStoreError, SqliteRunStore


def _record(record_id: str = "1", source: str = "hello", **kwargs: object) -> Record:
    return Record(record_id=record_id, source=source, **kwargs)  # type: ignore[arg-type]


class TestLifecycle:
    def test_add_records_and_pending(self) -> None:
        store = SqliteRunStore()
        assert store.count_records() == 0
        added = store.add_records([_record("1", "a"), _record("2", "b")])
        assert added == 2
        assert store.count_records() == 2
        assert {r.record_id for r in store.pending()} == {"1", "2"}

    def test_add_records_empty_is_a_noop(self) -> None:
        assert SqliteRunStore().add_records([]) == 0

    def test_add_records_is_idempotent_on_a_duplicate_id(self) -> None:
        store = SqliteRunStore()
        store.add_records([_record("1", "a")])
        assert store.add_records([_record("1", "a")]) == 0
        assert store.count_records() == 1

    def test_pending_excludes_skipped_records(self) -> None:
        store = SqliteRunStore()
        store.add_records([_record("1", "a"), _record("2", "   ", status=Status.SKIPPED)])
        assert {r.record_id for r in store.pending()} == {"1"}

    def test_pending_is_ordered_by_rel_path_then_line_no(self) -> None:
        store = SqliteRunStore()
        store.add_records([
            _record("c", "x", rel_path="b.txt", line_no=1),
            _record("a", "x", rel_path="a.txt", line_no=2),
            _record("b", "x", rel_path="a.txt", line_no=1),
        ])
        assert [r.record_id for r in store.pending()] == ["b", "a", "c"]

    def test_append_result_and_completed_ids(self) -> None:
        store = SqliteRunStore()
        store.add_records([_record("1", "a"), _record("2", "b")])
        applied = _record("1", "a", status=Status.VERIFIED, output="A")
        store.append_result(RunResult(record=applied))
        assert store.completed_ids() == {"1"}
        assert {r.record_id for r in store.pending()} == {"2"}

    def test_results_returns_the_latest_by_seq(self) -> None:
        store = SqliteRunStore()
        store.add_records([_record("1", "a")])
        store.append_result(RunResult(
            record=_record("1", "a", status=Status.PRODUCED, output="first")))
        store.append_result(RunResult(
            record=_record("1", "a", status=Status.VERIFIED, output="second")))
        [result] = list(store.results())
        assert result.record.output == "second" and result.record.status is Status.VERIFIED

    def test_from_config(self, tmp_path: Path) -> None:
        store = SqliteRunStore.from_config({"path": str(tmp_path / "run.db")})
        store.add_records([_record("1", "a")])
        assert store.count_records() == 1

    def test_in_memory_default(self) -> None:
        store = SqliteRunStore()
        store.add_records([_record()])
        assert store.count_records() == 1


class TestCaptureRoundTrip:
    def test_context_and_retrieved_and_violations_and_reviews_round_trip(self) -> None:
        store = SqliteRunStore()
        store.add_records([_record("1", "a")])
        result = RunResult(
            record=_record("1", "a", status=Status.VERIFIED, output="A", notes=("n1", "n2")),
            context_passage="the assembled passage",
            retrieved=(RetrievedRef("c1", "hit text", 0.75),),
            reviews=({"role": "r", "acceptable": True, "issues": [], "improved": None,
                     "error": None},),
            violations=(Violation("rule_x", Severity.WARNING, "a warning"),),
            rounds=2, error=None)
        store.append_result(result)
        [back] = list(store.results())
        assert back.record.notes == ("n1", "n2")
        assert back.context_passage == "the assembled passage"
        assert back.retrieved == (RetrievedRef("c1", "hit text", 0.75),)
        assert back.reviews == ({"role": "r", "acceptable": True, "issues": [], "improved": None,
                                "error": None},)
        assert back.violations == (Violation("rule_x", Severity.WARNING, "a warning"),)
        assert back.rounds == 2

    def test_error_field_round_trips(self) -> None:
        store = SqliteRunStore()
        store.add_records([_record("1", "a")])
        store.append_result(RunResult(
            record=_record("1", "a", status=Status.REJECTED, output=None),
            error="mechanical rules still violated after repairs"))
        [back] = list(store.results())
        assert back.error == "mechanical rules still violated after repairs"
        assert back.record.output is None and back.record.status is Status.REJECTED


class TestForeignKeyGuard:
    def test_append_result_for_an_unknown_record_is_refused(self) -> None:
        store = SqliteRunStore()
        with pytest.raises(RunStoreError, match="never added to the catalogue"):
            store.append_result(RunResult(
                record=_record("ghost", "x", status=Status.VERIFIED, output="o")))


class TestCrashSafety:
    """WAL + fsync-on-commit (default synchronous=FULL): a kill with no graceful close must not
    lose a committed result, and a resumed run must see exactly what was actually committed."""

    def test_kill_and_reopen_preserves_committed_results(self, tmp_path: Path) -> None:
        path = str(tmp_path / "run.db")
        store = SqliteRunStore(path)
        store.add_records([_record(str(i), f"s{i}") for i in range(5)])
        for i in range(3):
            store.append_result(RunResult(
                record=_record(str(i), f"s{i}", status=Status.VERIFIED, output=f"o{i}")))
        del store  # no close(): simulates a hard kill, no graceful checkpoint/shutdown

        reopened = SqliteRunStore(path)
        assert reopened.completed_ids() == {"0", "1", "2"}
        assert {r.record_id for r in reopened.pending()} == {"3", "4"}
        assert reopened.count_records() == 5

    def test_synchronous_normal_also_survives_a_kill_and_reopen(self, tmp_path: Path) -> None:
        path = str(tmp_path / "run.db")
        store = SqliteRunStore(path, synchronous="NORMAL")
        store.add_records([_record("1", "a")])
        store.append_result(RunResult(
            record=_record("1", "a", status=Status.VERIFIED, output="A")))
        del store

        reopened = SqliteRunStore(path, synchronous="NORMAL")
        assert reopened.completed_ids() == {"1"}

    def test_synchronous_is_case_insensitive(self, tmp_path: Path) -> None:
        SqliteRunStore(str(tmp_path / "run.db"), synchronous="normal").close()

    def test_bad_synchronous_is_refused(self) -> None:
        with pytest.raises(RunStoreError, match="synchronous must be"):
            SqliteRunStore(synchronous="BOGUS")


class TestConcurrency:
    """busy_timeout: a concurrent reader must not fail outright on the brief window WAL doesn't
    cover (e.g. its own checkpoint) -- it retries instead of raising immediately."""

    def test_busy_timeout_is_set(self) -> None:
        store = SqliteRunStore()
        assert store._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000  # type: ignore[attr-defined]

    def test_a_reader_is_not_locked_out_by_an_open_writer(self, tmp_path: Path) -> None:
        import sqlite3

        path = str(tmp_path / "run.db")
        store = SqliteRunStore(path)
        store.add_records([_record("1", "a")])

        writer = sqlite3.connect(path)
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("INSERT INTO records(record_id, source, status) VALUES ('2', 'b', "
                       "'pending')")
        try:
            assert store.count_records() == 1
        finally:
            writer.rollback()
            writer.close()

    def test_busy_timeout_exhausted_is_a_structured_error_not_a_raw_one(
            self, tmp_path: Path) -> None:
        # Two writers genuinely contend even under WAL (only one at a time); this holds a
        # competing write lock for longer than the store's busy_timeout so the retry budget is
        # actually exhausted, and checks the resulting sqlite3.OperationalError comes back as
        # RunStoreError, not the raw driver exception.
        import sqlite3
        import threading
        import time

        path = str(tmp_path / "run.db")
        store = SqliteRunStore(path)
        store._conn.execute("PRAGMA busy_timeout=100")  # type: ignore[attr-defined]
        store.add_records([_record("1", "a")])

        lock_acquired = threading.Event()

        def _hold_write_lock_for(seconds: float) -> None:
            writer = sqlite3.connect(path)
            writer.execute("BEGIN IMMEDIATE")
            writer.execute(
                "INSERT INTO records(record_id, source, status) VALUES ('2', 'b', 'pending')")
            lock_acquired.set()
            time.sleep(seconds)
            writer.rollback()
            writer.close()

        holder = threading.Thread(target=_hold_write_lock_for, args=(0.4,))
        holder.start()
        lock_acquired.wait(timeout=5)
        try:
            with pytest.raises(RunStoreError, match="could not add records"):
                store.add_records([_record("3", "c")])
        finally:
            holder.join(timeout=5)


class TestErrors:
    def test_bad_path_is_a_structured_error(self, tmp_path: Path) -> None:
        with pytest.raises(RunStoreError, match="could not open the run store"):
            SqliteRunStore(str(tmp_path / "no_such_dir" / "run.db"))

    def test_add_records_sql_error_is_structured(self) -> None:
        store = SqliteRunStore()
        bad = _record("1", object())  # type: ignore[arg-type]  # unbindable sqlite type
        with pytest.raises(RunStoreError, match="could not add records"):
            store.add_records([bad])

    def test_add_records_on_a_closed_store_is_structured(self) -> None:
        # Regression: rolling back on an already-closed connection used to raise the driver's raw
        # exception instead of the intended structured error, masking the real cause.
        store = SqliteRunStore()
        store.add_records([_record("1", "a")])
        store.close()
        with pytest.raises(RunStoreError, match="could not add records"):
            store.add_records([_record("2", "b")])

    def test_append_result_on_a_closed_store_is_structured(self) -> None:
        store = SqliteRunStore()
        store.add_records([_record("1", "a")])
        store.close()
        with pytest.raises(RunStoreError, match="could not append a result"):
            store.append_result(RunResult(
                record=_record("1", "a", status=Status.VERIFIED, output="A")))

    def test_pending_on_a_closed_store_is_structured(self) -> None:
        store = SqliteRunStore()
        store.add_records([_record("1", "a")])
        store.close()
        with pytest.raises(RunStoreError, match="pending query failed"):
            list(store.pending())

    def test_results_on_a_closed_store_is_structured(self) -> None:
        store = SqliteRunStore()
        store.add_records([_record("1", "a")])
        store.append_result(RunResult(record=_record("1", "a", status=Status.VERIFIED,
                                                      output="A")))
        store.close()
        with pytest.raises(RunStoreError, match="results query failed"):
            list(store.results())

    def test_latest_records_on_a_closed_store_is_structured(self) -> None:
        store = SqliteRunStore()
        store.add_records([_record("1", "a")])
        store.append_result(RunResult(record=_record("1", "a", status=Status.VERIFIED,
                                                      output="A")))
        store.close()
        with pytest.raises(RunStoreError, match="latest_records query failed"):
            list(store.latest_records())


class TestStreamingAtScale:
    """Regression: pending()/results()/latest_records() fetch internally in pages (never the
    whole table in one fetchall()), so this proves the paging loop is correct across a page
    boundary, not just for a handful of rows in one page."""

    def test_pending_spans_multiple_fetch_batches_correctly(self) -> None:
        store = SqliteRunStore()
        ids = {f"r{i}" for i in range(2500)}
        store.add_records([_record(rid, "x") for rid in ids])
        assert {r.record_id for r in store.pending()} == ids

    def test_results_and_latest_records_span_multiple_fetch_batches_correctly(self) -> None:
        store = SqliteRunStore()
        ids = {f"r{i}" for i in range(2500)}
        store.add_records([_record(rid, "x") for rid in ids])
        for rid in ids:
            store.append_result(RunResult(
                record=_record(rid, "x", status=Status.VERIFIED, output=f"out-{rid}")))
        assert {r.record.record_id for r in store.results()} == ids
        assert {r.record_id for r in store.latest_records()} == ids

    def test_injected_clock_drives_created_at(self) -> None:
        ticks = iter([1.0, 2.0])
        store = SqliteRunStore(clock=lambda: next(ticks))
        store.add_records([_record("1", "a")])
        store.append_result(RunResult(
            record=_record("1", "a", status=Status.VERIFIED, output="A")))
        # created_at is not part of the RunResult surface -- read it back through raw SQL to
        # confirm the injected clock (not wall time) actually drove it.
        row = store._conn.execute(  # type: ignore[attr-defined]
            "SELECT created_at FROM results").fetchone()
        assert row[0] == 1.0
