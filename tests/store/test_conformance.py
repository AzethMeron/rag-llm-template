"""Driver conformance: the same operations run against every implementation of each port, so
"interchangeable" is a tested claim rather than an assertion. As more drivers land (usearch,
pgvector, sqlite-vec, bm25s), they are added to the factory lists here and must pass unchanged.

The suite runs against the shipped driver AND a small independent in-memory implementation of each
port. Two implementations passing the identical operations is what makes "the port is a real,
independently-implementable seam" a tested fact rather than an assertion — and it is exactly the
shape a third party's own driver takes. The in-memory implementations are pure-Python (no numpy, no
brute-force *shipped* driver — they live here in the test suite, not in ``src/``), so they add no
dependency and are not a production default.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from ragkit.core.lexicon import Entry
from ragkit.core.ports import (
    LexiconStore,
    Pairing,
    PairingStore,
    RunResult,
    RunStore,
    SqlStore,
    VectorIndex,
)
from ragkit.core.records import Record, Status
from ragkit.store.lexicon.sqlite import SqliteLexicon
from ragkit.store.pairings.duckdb import DuckDBPairings
from ragkit.store.pairings.sqlite import SqlitePairings
from ragkit.store.run.sqlite import SqliteRunStore
from ragkit.store.sql.duckdb import DuckDBStore
from ragkit.store.sql.sqlite import SqliteStore, SqlStoreError
from ragkit.store.vector.common import VectorIndexError, validate_upsert
from ragkit.store.vector.lancedb import LanceVectorIndex
from ragkit.store.vector.qdrant import QdrantVectorIndex


class InMemoryVectorIndex:
    """A minimal, dependency-free VectorIndex — a second implementation of the port, purely to
    prove the port is independently implementable and interchangeable with the shipped driver."""

    def __init__(self) -> None:
        self._vectors: dict[str, list[float]] = {}
        self._meta: dict[str, Mapping[str, Any]] = {}

    def upsert(self, ids: Sequence[str], vectors: Sequence[Sequence[float]],
               metas: Sequence[Mapping[str, Any]]) -> None:
        # Through the shared boundary check, exactly as a third party's own driver would: the
        # argument contract is part of the port, not each driver's private business.
        validate_upsert(ids, vectors, metas, dim=3)
        for id_, vector, meta in zip(ids, vectors, metas, strict=True):
            self._vectors[id_] = list(vector)
            self._meta[id_] = meta

    def search(self, vector: Sequence[float], *, k: int,
               where: object = ()) -> list[tuple[str, float]]:
        scored = [(id_, _cosine(vector, vec)) for id_, vec in self._vectors.items()]
        scored.sort(key=lambda pair: pair[1], reverse=True)  # best-first, higher-is-better
        return scored[:k]

    def delete(self, ids: Sequence[str]) -> None:
        for id_ in ids:
            self._vectors.pop(id_, None)
            self._meta.pop(id_, None)

    def count(self) -> int:
        return len(self._vectors)

    def reconcile(self, chunk_ids: Iterable[str]) -> set[str]:
        wanted = set(chunk_ids)
        for orphan in set(self._vectors) - wanted:
            self.delete([orphan])
        return wanted - set(self._vectors)


class InMemoryPairings:
    """A minimal, dependency-free PairingStore — the second implementation of that port, combining
    a simple word-overlap search with a dict-backed row store. ``add`` mirrors the shipped drivers'
    idempotent-on-duplicate contract."""

    def __init__(self) -> None:
        self._rows: dict[str, Pairing] = {}

    def add(self, pairings: Iterable[Pairing]) -> int:
        added = 0
        for pairing in pairings:
            if pairing.chunk_id not in self._rows:
                self._rows[pairing.chunk_id] = pairing
                added += 1
        return added

    def search(self, query: str, *, k: int) -> list[tuple[str, float]]:
        terms = set(query.lower().split())
        if not terms or k <= 0:
            return []
        scored = []
        for chunk_id, pairing in self._rows.items():
            words = set(f"{pairing.source} {pairing.context} {pairing.target}".lower().split())
            overlap = len(terms & words)
            if overlap:
                scored.append((chunk_id, float(overlap)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def document(self, chunk_id: str) -> tuple[str, Mapping[str, Any]] | None:
        pairing = self._rows.get(chunk_id)
        if pairing is None:
            return None
        display = f"{pairing.source} -> {pairing.target}" if pairing.target else pairing.source
        return display, dict(pairing.meta)

    def get(self, chunk_id: str) -> Pairing | None:
        return self._rows.get(chunk_id)

    def all_ids(self) -> Iterator[str]:
        return iter(self._rows)

    def count(self) -> int:
        return len(self._rows)


class InMemoryRunStore:
    """A minimal, dependency-free RunStore — the second implementation of that port. ``_results``
    is append-only in call order, exactly like the SQL driver's ``seq``, so ordering derives from
    it the same way: :meth:`results` returns the latest result per record, ordered by the *index*
    (≈ ``seq``) of that latest write."""

    def __init__(self) -> None:
        self._records: dict[str, Record] = {}
        self._results: list[RunResult] = []

    def add_records(self, records: Iterable[Record]) -> int:
        added = 0
        for record in records:
            if record.record_id not in self._records:
                self._records[record.record_id] = record
                added += 1
        return added

    def append_result(self, result: RunResult) -> None:
        record_id = result.record.record_id
        if record_id not in self._records:
            raise ValueError(f"no record {record_id!r} in the catalogue")
        self._results.append(result)

    def completed_ids(self) -> set[str]:
        return {result.record.record_id for result in self._results}

    def pending(self) -> Iterator[Record]:
        done = self.completed_ids()
        waiting = [r for r in self._records.values()
                  if r.status is Status.PENDING and r.record_id not in done]
        waiting.sort(key=lambda r: (r.rel_path, r.line_no))
        return iter(waiting)

    def results(self) -> Iterator[RunResult]:
        latest_index: dict[str, int] = {}
        for index, result in enumerate(self._results):
            latest_index[result.record.record_id] = index
        return (self._results[i] for i in sorted(latest_index.values()))

    def latest_records(self) -> Iterator[Record]:
        return (result.record for result in self.results())

    def count_records(self) -> int:
        return len(self._records)


class InMemoryLexicon:
    """A minimal, dependency-free LexiconStore -- the second implementation of that port. ``add``
    upserts, keyed on ``(term, category)``, matching the shipped driver's update semantics."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], Entry] = {}

    def entries(self) -> list[Entry]:
        return list(self._rows.values())

    def add(self, entries: Iterable[Entry]) -> int:
        added = 0
        for entry in entries:
            key = (entry.term, entry.category)
            if key not in self._rows:
                added += 1
            self._rows[key] = entry
        return added


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# Each factory builds a fresh, empty driver in the given tmp directory. Both a shipped driver and an
# independent in-memory implementation must pass the identical conformance operations.
VECTOR_FACTORIES: list[Callable[[Path], VectorIndex]] = [
    lambda tmp: LanceVectorIndex(str(tmp / "v.lance"), dim=3),
    lambda tmp: QdrantVectorIndex(str(tmp / "v.qdrant"), dim=3),  # a second REAL vector DB
    lambda tmp: InMemoryVectorIndex(),
]
PAIRING_FACTORIES: list[Callable[[Path], PairingStore]] = [
    lambda tmp: SqlitePairings(str(tmp / "p.sqlite")),
    lambda tmp: DuckDBPairings(str(tmp / "p.duckdb")),  # a second REAL co-located store
    lambda tmp: InMemoryPairings(),
]
RUN_FACTORIES: list[Callable[[Path], RunStore]] = [
    lambda tmp: SqliteRunStore(str(tmp / "run.db")),
    lambda tmp: SqliteRunStore(),  # in-memory SQLite
    lambda tmp: InMemoryRunStore(),
]
LEXICON_FACTORIES: list[Callable[[Path], LexiconStore]] = [
    lambda tmp: SqliteLexicon(str(tmp / "lex.db")),
    lambda tmp: SqliteLexicon(),  # in-memory SQLite
    lambda tmp: InMemoryLexicon(),
]


@pytest.mark.parametrize("factory", VECTOR_FACTORIES)
class TestVectorIndexConformance:
    def test_lifecycle(self, factory: Callable[[Path], VectorIndex], tmp_path: Path) -> None:
        index = factory(tmp_path)
        assert index.count() == 0
        index.upsert(["a", "b", "c"],
                     [[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]],
                     [{"g": "x"}, {"g": "y"}, {"g": "x"}])
        assert index.count() == 3

        results = index.search([1, 0, 0], k=3)
        assert results[0][0] == "a"  # nearest
        scores = [s for _id, s in results]
        assert scores == sorted(scores, reverse=True)  # best-first, higher-is-better

        index.delete(["a"])
        assert index.count() == 2
        assert all(chunk_id != "a" for chunk_id, _ in index.search([1, 0, 0], k=5))

    def test_reconcile_contract(self, factory: Callable[[Path], VectorIndex],
                                tmp_path: Path) -> None:
        index = factory(tmp_path)
        index.upsert(["a", "b"], [[1, 0, 0], [0, 1, 0]], [{}, {}])
        assert index.reconcile(["a", "new"]) == {"new"}  # b dropped, new reported missing

    def test_upsert_refuses_a_duplicate_id_within_one_batch(
            self, factory: Callable[[Path], VectorIndex], tmp_path: Path) -> None:
        """The drivers used to disagree here, which is the kind of divergence a happy-path-only
        conformance suite cannot see: Qdrant's deterministic point id let the last occurrence
        silently win, while LanceDB's merge_insert refuses outright. An id whose vector depends on
        its position in the batch is not reproducible, so every driver refuses."""
        index = factory(tmp_path)
        with pytest.raises(VectorIndexError, match="more than once"):
            index.upsert(["a", "b", "a"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [{}, {}, {}])
        assert index.count() == 0  # refused at the boundary: nothing was written

    def test_upsert_rejections_are_identical_across_drivers(
            self, factory: Callable[[Path], VectorIndex], tmp_path: Path) -> None:
        index = factory(tmp_path)
        with pytest.raises(VectorIndexError, match="mismatched lengths"):
            index.upsert(["a"], [[1, 0, 0], [0, 1, 0]], [{}])
        with pytest.raises(VectorIndexError, match="dimension"):
            index.upsert(["a"], [[1, 0]], [{}])


@pytest.mark.parametrize("factory", PAIRING_FACTORIES)
class TestPairingStoreConformance:
    def test_lifecycle(self, factory: Callable[[Path], PairingStore], tmp_path: Path) -> None:
        store = factory(tmp_path)
        assert store.count() == 0
        added = store.add([
            Pairing(chunk_id="p1", source="the quick brown fox", target="a fast animal"),
            Pairing(chunk_id="p2", source="a slow green turtle", target="not fast at all"),
        ])
        assert added == 2
        assert store.count() == 2
        assert store.document("p1") == ("the quick brown fox -> a fast animal", {})
        assert store.document("missing") is None

        pairing = store.get("p1")
        assert pairing is not None and pairing.target == "a fast animal"
        assert store.get("missing") is None
        assert set(store.all_ids()) == {"p1", "p2"}

    def test_search_ranks_higher_is_better(self, factory: Callable[[Path], PairingStore],
                                           tmp_path: Path) -> None:
        store = factory(tmp_path)
        store.add([Pairing(chunk_id="p1", source="the quick brown fox"),
                  Pairing(chunk_id="p2", source="a slow green turtle")])
        results = store.search("quick fox", k=5)
        assert [chunk_id for chunk_id, _score in results] == ["p1"]
        assert all(score >= 0 for _id, score in results)

    def test_empty_query(self, factory: Callable[[Path], PairingStore], tmp_path: Path) -> None:
        assert factory(tmp_path).search("", k=5) == []

    def test_add_is_idempotent_on_a_duplicate_chunk_id(
            self, factory: Callable[[Path], PairingStore], tmp_path: Path) -> None:
        store = factory(tmp_path)
        store.add([Pairing(chunk_id="p1", source="first")])
        assert store.add([Pairing(chunk_id="p1", source="second")]) == 0
        assert store.count() == 1
        pairing = store.get("p1")
        assert pairing is not None and pairing.source == "first"


# Two real SqlStore engines behind one port: swapping SQLite -> DuckDB is a config edit only.
_SCHEMA = "CREATE TABLE t(id INTEGER, name VARCHAR); INSERT INTO t VALUES (1, 'a'), (2, 'b');"
SQL_DRIVERS: list[tuple[type[SqlStore], str]] = [
    (SqliteStore, "s.sqlite"),
    (DuckDBStore, "s.duckdb"),
]


@pytest.mark.parametrize(("driver", "filename"), SQL_DRIVERS)
class TestSqlStoreConformance:
    def test_query_and_write(self, driver: type[SqlStore], filename: str, tmp_path: Path) -> None:
        store = driver(str(tmp_path / filename), schema_sql=_SCHEMA)  # type: ignore[call-arg]
        assert store.query("SELECT name FROM t WHERE id = ?", [1]) == [{"name": "a"}]
        store.execute("INSERT INTO t VALUES (3, 'c')")
        assert len(store.query("SELECT * FROM t")) == 3
        store.close()  # type: ignore[attr-defined]

    def test_read_only_binding_refuses_a_write(self, driver: type[SqlStore], filename: str,
                                               tmp_path: Path) -> None:
        path = str(tmp_path / filename)
        driver(path, schema_sql=_SCHEMA).close()  # type: ignore[call-arg,attr-defined]
        readonly = driver(path, read_only=True)  # type: ignore[call-arg]
        assert readonly.query("SELECT count(*) AS n FROM t")[0]["n"] == 2
        with pytest.raises(SqlStoreError, match="read_only"):
            readonly.execute("INSERT INTO t VALUES (9, 'z')")
        readonly.close()  # type: ignore[attr-defined]

    def test_bad_sql_is_a_structured_error(self, driver: type[SqlStore], filename: str,
                                          tmp_path: Path) -> None:
        store = driver(str(tmp_path / filename), schema_sql=_SCHEMA)  # type: ignore[call-arg]
        with pytest.raises(SqlStoreError, match="query failed"):
            store.query("SELECT * FROM no_such_table")
        store.close()  # type: ignore[attr-defined]


@pytest.mark.parametrize("factory", RUN_FACTORIES)
class TestRunStoreConformance:
    def test_lifecycle(self, factory: Callable[[Path], RunStore], tmp_path: Path) -> None:
        store = factory(tmp_path)
        assert store.count_records() == 0
        added = store.add_records([Record(record_id="1", source="a"),
                                   Record(record_id="2", source="b")])
        assert added == 2
        assert store.count_records() == 2
        assert {r.record_id for r in store.pending()} == {"1", "2"}

        applied = Record(record_id="1", source="a", status=Status.VERIFIED, output="A")
        store.append_result(RunResult(record=applied))
        assert store.completed_ids() == {"1"}
        assert {r.record_id for r in store.pending()} == {"2"}
        [result] = list(store.results())
        assert result.record.record_id == "1" and result.record.output == "A"

    def test_add_records_is_idempotent_on_a_duplicate_id(
            self, factory: Callable[[Path], RunStore], tmp_path: Path) -> None:
        store = factory(tmp_path)
        store.add_records([Record(record_id="1", source="a")])
        assert store.add_records([Record(record_id="1", source="a")]) == 0
        assert store.count_records() == 1

    def test_results_returns_the_latest_by_write_order(
            self, factory: Callable[[Path], RunStore], tmp_path: Path) -> None:
        store = factory(tmp_path)
        store.add_records([Record(record_id="1", source="a")])
        store.append_result(RunResult(
            record=Record(record_id="1", source="a", status=Status.PRODUCED, output="first")))
        store.append_result(RunResult(
            record=Record(record_id="1", source="a", status=Status.VERIFIED, output="second")))
        [result] = list(store.results())
        assert result.record.output == "second"

    def test_latest_records_matches_results_but_only_the_record(
            self, factory: Callable[[Path], RunStore], tmp_path: Path) -> None:
        store = factory(tmp_path)
        store.add_records([Record(record_id="1", source="a")])
        store.append_result(RunResult(
            record=Record(record_id="1", source="a", status=Status.PRODUCED, output="first")))
        store.append_result(RunResult(
            record=Record(record_id="1", source="a", status=Status.VERIFIED, output="second")))
        [record] = list(store.latest_records())
        assert record.output == "second" and record.status is Status.VERIFIED
        assert [r.record for r in store.results()] == list(store.latest_records())

    def test_pending_excludes_skipped_and_completed_records(
            self, factory: Callable[[Path], RunStore], tmp_path: Path) -> None:
        store = factory(tmp_path)
        store.add_records([
            Record(record_id="1", source="a"),
            Record(record_id="2", source="   ", status=Status.SKIPPED),
        ])
        assert {r.record_id for r in store.pending()} == {"1"}


@pytest.mark.parametrize("factory", LEXICON_FACTORIES)
class TestLexiconStoreConformance:
    def test_lifecycle(self, factory: Callable[[Path], LexiconStore], tmp_path: Path) -> None:
        store = factory(tmp_path)
        assert store.entries() == []
        added = store.add([Entry(term="cat", rendering="kot"),
                           Entry(term="dog", rendering="pies")])
        assert added == 2
        assert {e.term for e in store.entries()} == {"cat", "dog"}

    def test_add_upserts_on_a_duplicate_key(
            self, factory: Callable[[Path], LexiconStore], tmp_path: Path) -> None:
        store = factory(tmp_path)
        store.add([Entry(term="cat", rendering="kot")])
        assert store.add([Entry(term="cat", rendering="KOTEK")]) == 0
        [entry] = store.entries()
        assert entry.rendering == "KOTEK"
