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

from ragkit.core.ports import (
    DocumentStore,
    LexicalIndex,
    Pairing,
    PairingStore,
    SqlStore,
    VectorIndex,
)
from ragkit.store.documents.sqlite import SqliteDocuments
from ragkit.store.lexical.fts5 import Fts5Index
from ragkit.store.pairings.duckdb import DuckDBPairings
from ragkit.store.pairings.sqlite import SqlitePairings
from ragkit.store.sql.duckdb import DuckDBStore
from ragkit.store.sql.sqlite import SqliteStore, SqlStoreError
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


class InMemoryLexicalIndex:
    """A minimal, dependency-free LexicalIndex — the second implementation of that port."""

    def __init__(self) -> None:
        self._docs: dict[str, set[str]] = {}

    def index(self, chunk_id: str, text: str) -> None:
        self._docs[chunk_id] = set(text.lower().split())

    def search(self, query: str, *, k: int) -> list[tuple[str, float]]:
        terms = set(query.lower().split())
        if not terms:
            return []
        scored = [(id_, float(len(terms & words))) for id_, words in self._docs.items()]
        hits = [(id_, score) for id_, score in scored if score > 0]
        hits.sort(key=lambda pair: pair[1], reverse=True)
        return hits[:k]

    def delete(self, chunk_id: str) -> None:
        self._docs.pop(chunk_id, None)


class InMemoryDocuments:
    """A minimal, dependency-free DocumentStore — the second implementation of that port."""

    def __init__(self) -> None:
        self._rows: dict[str, tuple[str, Mapping[str, Any]]] = {}

    def add_documents(self, rows: Iterable[tuple[str, str, Mapping[str, Any]]]) -> None:
        for chunk_id, display, meta in rows:
            self._rows[chunk_id] = (display, dict(meta))

    def document(self, chunk_id: str) -> tuple[str, Mapping[str, Any]] | None:
        return self._rows.get(chunk_id)

    def count(self) -> int:
        return len(self._rows)


class InMemoryPairings:
    """A minimal, dependency-free PairingStore — the second implementation of that port, combining
    a simple word-overlap search (like InMemoryLexicalIndex) with a dict-backed row store (like
    InMemoryDocuments). ``add`` mirrors the shipped drivers' idempotent-on-duplicate contract."""

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
LEXICAL_FACTORIES: list[Callable[[Path], LexicalIndex]] = [
    lambda tmp: Fts5Index(),
    lambda tmp: InMemoryLexicalIndex(),
]
DOCUMENT_FACTORIES: list[Callable[[Path], DocumentStore]] = [
    lambda tmp: SqliteDocuments(str(tmp / "rows.db")),
    lambda tmp: SqliteDocuments(),  # in-memory SQLite
    lambda tmp: InMemoryDocuments(),
]
PAIRING_FACTORIES: list[Callable[[Path], PairingStore]] = [
    lambda tmp: SqlitePairings(str(tmp / "p.sqlite")),
    lambda tmp: DuckDBPairings(str(tmp / "p.duckdb")),  # a second REAL co-located store
    lambda tmp: InMemoryPairings(),
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


@pytest.mark.parametrize("factory", LEXICAL_FACTORIES)
class TestLexicalIndexConformance:
    def test_lifecycle(self, factory: Callable[[Path], LexicalIndex], tmp_path: Path) -> None:
        index = factory(tmp_path)
        index.index("d1", "the quick brown fox")
        index.index("d2", "a lazy dog")

        results = index.search("quick fox", k=5)
        assert [chunk_id for chunk_id, _ in results] == ["d1"]  # only d1 matches
        assert all(score >= 0 for _id, score in results)  # higher-is-better

        index.delete("d1")
        assert index.search("quick", k=5) == []

    def test_empty_query(self, factory: Callable[[Path], LexicalIndex], tmp_path: Path) -> None:
        assert factory(tmp_path).search("", k=5) == []


@pytest.mark.parametrize("factory", DOCUMENT_FACTORIES)
class TestDocumentStoreConformance:
    def test_lifecycle(self, factory: Callable[[Path], DocumentStore], tmp_path: Path) -> None:
        store = factory(tmp_path)
        assert store.count() == 0
        store.add_documents([("d1", "cat -> kot", {"n": 1}), ("d2", "dog -> pies", {})])
        assert store.count() == 2
        assert store.document("d1") == ("cat -> kot", {"n": 1})
        assert store.document("d2") == ("dog -> pies", {})
        assert store.document("missing") is None

    def test_reinsert_replaces(self, factory: Callable[[Path], DocumentStore],
                               tmp_path: Path) -> None:
        store = factory(tmp_path)
        store.add_documents([("d1", "first", {})])
        store.add_documents([("d1", "second", {"v": 2})])
        assert store.count() == 1 and store.document("d1") == ("second", {"v": 2})


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
