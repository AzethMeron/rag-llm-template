"""The PairingStore drivers: lifecycle over both, the SQLite driver's co-located rows+index
invariant, and the DuckDB driver's documented weaker-atomicity tradeoff."""
from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import Any

import pytest

from ragkit.core.ports import Pairing
from ragkit.store.pairings.common import PairingStoreError
from ragkit.store.pairings.duckdb import DuckDBPairings
from ragkit.store.pairings.sqlite import SqlitePairings

DRIVERS: list[tuple[type[Any], str]] = [
    (SqlitePairings, "p.sqlite"),
    (DuckDBPairings, "p.duckdb"),
]


def _open_duckdb_pairings_in_a_new_process(path: str, result: multiprocessing.Queue[str]
                                            ) -> None:
    # Module-level (not nested): the "spawn" context re-imports this module in the child, so the
    # target must be picklable by reference, not a closure.
    try:
        DuckDBPairings(path)
        result.put("ok")
    except PairingStoreError as exc:
        result.put(f"PairingStoreError: {exc}")
    except Exception as exc:  # noqa: BLE001 -- reports any type back so the assertion names it
        result.put(f"{type(exc).__name__}: {exc}")


@pytest.mark.parametrize(("driver", "filename"), DRIVERS)
class TestLifecycle:
    def test_add_search_document_get_round_trip(self, driver: type[Any], filename: str,
                                                 tmp_path: Path) -> None:
        store = driver(str(tmp_path / filename))
        assert store.count() == 0
        added = store.add([
            Pairing(chunk_id="c1", source="the quick brown fox", target="a fast animal",
                   meta={"n": 1}),
            Pairing(chunk_id="c2", source="a slow green turtle", target="not fast at all"),
        ])
        assert added == 2
        assert store.count() == 2
        assert store.document("c1") == ("the quick brown fox -> a fast animal", {"n": 1})
        assert store.document("missing") is None

        pairing = store.get("c1")
        assert pairing is not None
        assert pairing.source == "the quick brown fox" and pairing.target == "a fast animal"
        assert pairing.meta == {"n": 1} and pairing.verified is False
        assert store.get("missing") is None
        assert set(store.all_ids()) == {"c1", "c2"}
        store.close()

    def test_search_ranks_by_relevance_higher_is_better(self, driver: type[Any], filename: str,
                                                         tmp_path: Path) -> None:
        store = driver(str(tmp_path / filename))
        store.add([Pairing(chunk_id="c1", source="the quick brown fox"),
                  Pairing(chunk_id="c2", source="a slow green turtle")])
        results = store.search("quick fox", k=5)
        assert [chunk_id for chunk_id, _score in results] == ["c1"]
        assert all(score >= 0 for _id, score in results)
        store.close()

    def test_empty_query_returns_nothing(self, driver: type[Any], filename: str,
                                         tmp_path: Path) -> None:
        assert driver(str(tmp_path / filename)).search("", k=5) == []

    def test_k_zero_returns_nothing(self, driver: type[Any], filename: str,
                                    tmp_path: Path) -> None:
        store = driver(str(tmp_path / filename))
        store.add([Pairing(chunk_id="c1", source="hello world")])
        assert store.search("hello", k=0) == []

    def test_add_empty_is_a_noop(self, driver: type[Any], filename: str, tmp_path: Path) -> None:
        assert driver(str(tmp_path / filename)).add([]) == 0

    def test_re_add_same_chunk_id_is_ignored_not_overwritten(
            self, driver: type[Any], filename: str, tmp_path: Path) -> None:
        store = driver(str(tmp_path / filename))
        store.add([Pairing(chunk_id="c1", source="first")])
        added_again = store.add([Pairing(chunk_id="c1", source="second")])
        assert added_again == 0
        assert store.count() == 1
        pairing = store.get("c1")
        assert pairing is not None and pairing.source == "first"

    def test_display_falls_back_to_source_without_a_target(
            self, driver: type[Any], filename: str, tmp_path: Path) -> None:
        store = driver(str(tmp_path / filename))
        store.add([Pairing(chunk_id="c1", source="just a source, no target")])
        assert store.document("c1") == ("just a source, no target", {})

    def test_persists_on_disk_across_reopen(self, driver: type[Any], filename: str,
                                            tmp_path: Path) -> None:
        path = str(tmp_path / filename)
        first = driver(path)
        first.add([Pairing(chunk_id="c1", source="hello world", target="cześć świecie")])
        first.close()
        reopened = driver(path)
        assert reopened.count() == 1
        assert reopened.document("c1") == ("hello world -> cześć świecie", {})
        assert reopened.search("hello", k=5)  # the search index, not just the row, survives reopen
        reopened.close()

    def test_from_config(self, driver: type[Any], filename: str, tmp_path: Path) -> None:
        store = driver.from_config({"path": str(tmp_path / filename)})
        store.add([Pairing(chunk_id="c1", source="x")])
        assert store.count() == 1

    def test_in_memory_default(self, driver: type[Any], filename: str) -> None:
        store = driver()
        store.add([Pairing(chunk_id="c1", source="x")])
        assert store.count() == 1

    def test_bad_path_is_a_structured_error(self, driver: type[Any], filename: str,
                                            tmp_path: Path) -> None:
        with pytest.raises(PairingStoreError):
            driver(str(tmp_path / "no_such_dir" / filename))

    def test_query_on_a_closed_store_is_structured(self, driver: type[Any], filename: str,
                                                    tmp_path: Path) -> None:
        store = driver(str(tmp_path / filename))
        store.add([Pairing(chunk_id="c1", source="hello world")])
        store.close()
        with pytest.raises(PairingStoreError, match="query failed"):
            store.search("hello", k=5)

    def test_add_on_a_closed_store_is_structured(self, driver: type[Any], filename: str,
                                                  tmp_path: Path) -> None:
        # Regression: rolling back / rebuilding on an already-closed connection used to raise the
        # driver's raw exception instead of the intended structured error, masking the real cause.
        store = driver(str(tmp_path / filename))
        store.add([Pairing(chunk_id="c1", source="hello world")])
        store.close()
        with pytest.raises(PairingStoreError, match="could not add pairings"):
            store.add([Pairing(chunk_id="c2", source="more")])

    def test_all_ids_spans_multiple_fetch_batches_correctly(
            self, driver: type[Any], filename: str, tmp_path: Path) -> None:
        # Regression: all_ids() fetches in pages internally (so a multi-million-row corpus is
        # never held fully in memory just to list ids); this proves the paging loop itself is
        # correct across a page boundary, not just for a handful of rows in one page.
        store = driver(str(tmp_path / filename))
        expected = {f"c{i}" for i in range(2500)}
        store.add([Pairing(chunk_id=cid, source=f"source {cid}") for cid in expected])
        assert set(store.all_ids()) == expected
        store.close()

    def test_all_ids_on_a_closed_store_is_structured(self, driver: type[Any], filename: str,
                                                      tmp_path: Path) -> None:
        store = driver(str(tmp_path / filename))
        store.add([Pairing(chunk_id="c1", source="hello world")])
        store.close()
        with pytest.raises(PairingStoreError, match="could not list pairing ids"):
            list(store.all_ids())

    def test_all_ids_error_on_a_later_fetch_batch_is_structured(
            self, driver: type[Any], filename: str, tmp_path: Path) -> None:
        # Distinct from the closed-before-iterating case above: this closes the store *after* the
        # first (successful) fetchmany batch, so the failure is the loop's second-and-later
        # iteration, not just its opening query.
        store = driver(str(tmp_path / filename))
        store.add([Pairing(chunk_id="c1", source="hello world")])
        ids = store.all_ids()
        assert next(ids) == "c1"
        store.close()
        with pytest.raises(PairingStoreError, match="could not list pairing ids"):
            next(ids)


class TestSqliteCoLocationInvariant:
    """The property the co-located schema exists for: a write to the row and its search entry
    happen in one transaction, so a failure partway through leaves neither behind."""

    def test_an_aborted_add_leaves_neither_row_nor_search_entry(self) -> None:
        store = SqlitePairings()
        good = Pairing(chunk_id="ok", source="alpha beta")
        bad = Pairing(chunk_id="bad", source=object())  # type: ignore[arg-type]  # unbindable
        with pytest.raises(PairingStoreError, match="could not add pairings"):
            store.add([good, bad])
        assert store.count() == 0
        assert store.get("ok") is None
        assert store.search("alpha", k=5) == []


class TestSqliteConcurrency:
    """WAL + busy_timeout: a concurrent reader must not be locked out by a writer's in-flight
    transaction (an on-disk store is read from many workers at once during a run)."""

    def test_journal_mode_is_wal(self, tmp_path: Path) -> None:
        store = SqlitePairings(str(tmp_path / "p.db"))
        assert store._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    def test_busy_timeout_is_set(self) -> None:
        store = SqlitePairings()
        assert store._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000

    def test_a_reader_is_not_locked_out_by_an_open_writer(self, tmp_path: Path) -> None:
        import sqlite3

        path = str(tmp_path / "p.db")
        store = SqlitePairings(path)
        store.add([Pairing(chunk_id="p1", source="alpha")])

        # A second connection holds a write transaction open -- under the default rollback
        # journal this would make the read below raise "database is locked"; under WAL it must
        # not, because a reader never blocks on a writer's uncommitted transaction.
        writer = sqlite3.connect(path)
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("INSERT INTO pairings(chunk_id, source) VALUES ('p2', 'beta')")
        try:
            assert store.count() == 1
            assert store.search("alpha", k=5)
        finally:
            writer.rollback()
            writer.close()

    def test_busy_timeout_exhausted_is_a_structured_error_not_a_raw_one(
            self, tmp_path: Path) -> None:
        # Unlike the read above (WAL: readers never contend with a writer), two *writers* do
        # genuinely contend -- WAL still allows only one at a time. This holds a competing write
        # lock for longer than the store's busy_timeout, so the retry budget is actually
        # exhausted (not just brushed past), and checks the resulting sqlite3.OperationalError
        # ("database is locked") comes back as PairingStoreError, not the raw driver exception.
        import sqlite3
        import threading
        import time

        path = str(tmp_path / "p.db")
        store = SqlitePairings(path)
        store._conn.execute("PRAGMA busy_timeout=100")  # shrunk so the test stays fast
        store.add([Pairing(chunk_id="p1", source="alpha")])

        lock_acquired = threading.Event()

        def _hold_write_lock_for(seconds: float) -> None:
            writer = sqlite3.connect(path)
            writer.execute("BEGIN IMMEDIATE")
            writer.execute("INSERT INTO pairings(chunk_id, source) VALUES ('p2', 'beta')")
            lock_acquired.set()
            time.sleep(seconds)
            writer.rollback()
            writer.close()

        holder = threading.Thread(target=_hold_write_lock_for, args=(0.4,))
        holder.start()
        lock_acquired.wait(timeout=5)
        try:
            with pytest.raises(PairingStoreError, match="could not add pairings"):
                store.add([Pairing(chunk_id="p3", source="gamma")])
        finally:
            holder.join(timeout=5)


class TestDuckDBBatchAtomicity:
    """Regression: DuckDB does not roll back a failed executemany on its own, so a mid-batch
    failure used to leave a partial write and return an added-count describing it -- breaking
    PairingStore.add's "in one transaction" promise that the SQLite driver honours. An explicit
    BEGIN/COMMIT/ROLLBACK now makes it all-or-nothing on both drivers."""

    def test_a_partial_batch_failure_leaves_nothing_behind(self) -> None:
        store = DuckDBPairings()
        good = Pairing(chunk_id="ok", source="alpha beta")
        bad = Pairing(chunk_id="bad", source=object())  # type: ignore[arg-type]  # unbindable
        with pytest.raises(PairingStoreError, match="could not add pairings"):
            store.add([good, bad])
        assert store.count() == 0
        assert store.get("ok") is None
        assert store.search("alpha", k=5) == []

    def test_the_store_is_still_usable_after_a_rolled_back_batch(self) -> None:
        store = DuckDBPairings()
        bad = Pairing(chunk_id="bad", source=object())  # type: ignore[arg-type]
        with pytest.raises(PairingStoreError):
            store.add([bad])
        assert store.add([Pairing(chunk_id="ok", source="alpha beta")]) == 1
        assert [cid for cid, _ in store.search("alpha", k=5)] == ["ok"]


class TestDuckDBConcurrency:
    """Unlike SQLite (WAL: many readers, one writer, all in one process-shared connection),
    DuckDB has no equivalent of busy_timeout for a genuinely separate OS *process* opening the
    same on-disk file: a second process is refused outright, immediately, not retried. This must
    still surface as a structured PairingStoreError, not a raw duckdb.IOException -- the actual
    real-world incident this guards against is two admin processes (an ingest job and a status
    check, say) pointed at the same on-disk pairings database."""

    def test_a_second_process_opening_the_same_file_is_refused_and_structured(
            self, tmp_path: Path) -> None:
        path = str(tmp_path / "p.duckdb")
        store = DuckDBPairings(path)  # held open for the whole test -- the file lock is live
        store.add([Pairing(chunk_id="c1", source="alpha")])

        ctx = multiprocessing.get_context("spawn")
        queue: multiprocessing.Queue[str] = ctx.Queue()
        proc = ctx.Process(target=_open_duckdb_pairings_in_a_new_process, args=(path, queue))
        proc.start()
        outcome = queue.get(timeout=30)
        proc.join(timeout=30)

        assert outcome.startswith("PairingStoreError:"), outcome
        assert "could not open the pairing store" in outcome
        store.close()


class TestDuckDBDeferredIndex:
    """The FTS index is rebuilt by the first search after a write, not inside every add.

    Rebuilding per add made a batched load quadratic: DuckDB's fts extension re-indexes the whole
    table each time, so a 7M-row corpus at 5k per batch re-indexed a growing table ~1,400 times.
    Deferring keeps `search`'s contract identical -- it never sees a row the index is missing."""

    def test_add_does_not_rebuild_the_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = DuckDBPairings()
        rebuilds = _count_rebuilds(store, monkeypatch)
        for i in range(5):
            store.add([Pairing(chunk_id=f"c{i}", source=f"alpha {i}")])
        assert rebuilds == [], "add() must not rebuild the FTS index"

    def test_many_batches_then_a_search_rebuilds_exactly_once(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = DuckDBPairings()
        rebuilds = _count_rebuilds(store, monkeypatch)
        for i in range(10):
            store.add([Pairing(chunk_id=f"c{i}", source=f"alpha beta {i}")])
        store.search("alpha", k=5)
        store.search("beta", k=5)  # still clean -- no write since the rebuild
        assert len(rebuilds) == 1

    def test_search_still_sees_every_row_ever_added(self) -> None:
        store = DuckDBPairings()
        store.add([Pairing(chunk_id="c1", source="alpha")])
        assert [cid for cid, _ in store.search("alpha", k=5)] == ["c1"]
        store.add([Pairing(chunk_id="c2", source="alpha again")])
        assert {cid for cid, _ in store.search("alpha", k=5)} == {"c1", "c2"}

    def test_a_rebuild_failure_surfaces_from_search_as_a_structured_error(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = DuckDBPairings()
        store.add([Pairing(chunk_id="c1", source="a")])

        def flaky() -> None:
            raise RuntimeError("rebuild boom")

        monkeypatch.setattr(store, "_rebuild_fts", flaky)
        with pytest.raises(PairingStoreError, match="fts query failed"):
            store.search("a", k=1)


def _count_rebuilds(store: DuckDBPairings, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    seen: list[int] = []
    real = store._rebuild_fts

    def spy() -> None:
        seen.append(1)
        real()

    monkeypatch.setattr(store, "_rebuild_fts", spy)
    return seen


class TestDuckDBErrors:
    def test_missing_duckdb_package_is_a_structured_error(self, monkeypatch: pytest.MonkeyPatch
                                                           ) -> None:
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "duckdb":
                raise ImportError("no duckdb")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(PairingStoreError, match="needs the duckdb package"):
            DuckDBPairings()

    def test_a_pairing_store_error_while_opening_propagates_unwrapped(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        # __init__ must not re-wrap a PairingStoreError _load_fts already raised (that would bury
        # the real reason inside a generic "could not open" message).
        from ragkit.store.pairings import duckdb as duckdb_module

        def boom(_conn: object) -> None:
            raise PairingStoreError("fts unavailable")

        monkeypatch.setattr(duckdb_module, "_load_fts", boom)
        with pytest.raises(PairingStoreError, match=r"^fts unavailable$"):
            DuckDBPairings()


class _ScriptedConn:
    """A stub connection whose ``execute`` fails exactly when ``should_fail(sql, call_count)``
    says so -- drives ``_load_fts`` without needing the real extension to actually be absent."""

    def __init__(self, should_fail: Any) -> None:
        self._should_fail = should_fail
        self.calls: list[str] = []

    def execute(self, sql: str) -> None:
        self.calls.append(sql)
        if self._should_fail(sql, self.calls.count(sql)):
            raise RuntimeError(f"boom: {sql}")


class TestLoadFts:
    def test_falls_back_to_install_on_first_load_failure(self) -> None:
        from ragkit.store.pairings.duckdb import _load_fts

        conn = _ScriptedConn(lambda sql, n: sql == "LOAD fts" and n == 1)
        _load_fts(conn)
        assert conn.calls == ["LOAD fts", "INSTALL fts", "LOAD fts"]

    def test_raises_a_structured_error_when_install_also_fails(self) -> None:
        from ragkit.store.pairings.duckdb import _load_fts

        conn = _ScriptedConn(lambda sql, n: True)  # every call fails: offline, nothing cached
        with pytest.raises(PairingStoreError, match="could not be installed or loaded"):
            _load_fts(conn)
