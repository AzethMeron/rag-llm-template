"""The PairingStore drivers: lifecycle over both, the SQLite driver's co-located rows+index
invariant, and the DuckDB driver's documented weaker-atomicity tradeoff."""
from __future__ import annotations

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


class TestDuckDBWeakerAtomicity:
    """Documented tradeoff (see the module docstring): unlike the SQLite driver, DuckDB does not
    roll back a whole batch when one row in it fails -- a partial write can survive."""

    def test_a_partial_batch_failure_can_leave_a_partial_write(self) -> None:
        store = DuckDBPairings()
        good = Pairing(chunk_id="ok", source="alpha beta")
        bad = Pairing(chunk_id="bad", source=object())  # type: ignore[arg-type]  # unbindable
        with pytest.raises(PairingStoreError, match="could not add pairings"):
            store.add([good, bad])
        assert store.count() == 1
        assert store.get("ok") is not None
        # The index is still rebuilt over whatever survived, so it is never stale either.
        assert store.search("alpha", k=5)


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
