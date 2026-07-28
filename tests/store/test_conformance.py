"""Driver conformance: the same operations run against every implementation of each port, so
"interchangeable" is a tested claim rather than an assertion. As more drivers land (usearch,
pgvector, sqlite-vec, bm25s), they are added to the factory lists here and must pass unchanged.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from ragkit.core.ports import LexicalIndex, VectorIndex
from ragkit.store.lexical.fts5 import Fts5Index
from ragkit.store.vector.lancedb import LanceVectorIndex

# Each factory builds a fresh, empty driver in the given tmp directory.
VECTOR_FACTORIES: list[Callable[[Path], VectorIndex]] = [
    lambda tmp: LanceVectorIndex(str(tmp / "v.lance"), dim=3),
]
LEXICAL_FACTORIES: list[Callable[[Path], LexicalIndex]] = [
    lambda tmp: Fts5Index(),
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
