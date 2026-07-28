"""Qdrant VectorIndex driver: the branches the shared conformance suite does not reach
(from_config, :memory:, reopening an existing collection, filter compilation, errors, paging)."""
from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.core.ports import FilterOp, Predicate
from ragkit.store.vector.qdrant import QdrantVectorIndex, VectorIndexError


def _seed(index: QdrantVectorIndex) -> None:
    index.upsert(["a", "b", "c"],
                 [[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]],
                 [{"g": "x", "n": 1}, {"g": "y", "n": 2}, {"g": "x", "n": 3}])


class TestBasics:
    def test_from_config_in_memory(self) -> None:
        index = QdrantVectorIndex.from_config({"path": ":memory:", "dim": 3})
        _seed(index)
        assert index.count() == 3
        assert index.search([1, 0, 0], k=1)[0][0] == "a"

    def test_positive_dim_required(self) -> None:
        with pytest.raises(VectorIndexError, match="positive dim"):
            QdrantVectorIndex(dim=0)

    def test_reopen_existing_collection(self, tmp_path: Path) -> None:
        path = str(tmp_path / "v.qdrant")
        first = QdrantVectorIndex(path, dim=3)
        _seed(first)
        first.close()
        reopened = QdrantVectorIndex(path, dim=3)  # collection already exists -> not re-created
        assert reopened.count() == 3
        reopened.close()

    def test_upsert_length_mismatch_refused(self) -> None:
        with pytest.raises(VectorIndexError, match="mismatched lengths"):
            QdrantVectorIndex(dim=3).upsert(["a"], [[1, 0, 0], [0, 1, 0]], [{}])

    def test_wrong_dimension_refused(self) -> None:
        index = QdrantVectorIndex(dim=3)
        with pytest.raises(VectorIndexError, match="dimension"):
            index.upsert(["a"], [[1, 0]], [{}])
        _seed(index)
        with pytest.raises(VectorIndexError, match="query vector has dimension"):
            index.search([1, 0], k=1)

    def test_empty_upsert_and_delete_are_noops(self) -> None:
        index = QdrantVectorIndex(dim=3)
        index.upsert([], [], [])
        index.delete([])
        assert index.count() == 0

    def test_search_on_empty_index_returns_nothing(self) -> None:
        assert QdrantVectorIndex(dim=3).search([1, 0, 0], k=5) == []

    def test_open_failure_is_a_structured_error(self, tmp_path: Path) -> None:
        # A path that is an existing file, not a directory, cannot host a local Qdrant store.
        clash = tmp_path / "not_a_dir"
        clash.write_text("x", encoding="utf-8")
        with pytest.raises(VectorIndexError, match="could not open the Qdrant collection"):
            QdrantVectorIndex(str(clash), dim=3)

    def test_reconcile_with_no_orphans(self) -> None:
        index = QdrantVectorIndex(dim=3)
        _seed(index)
        assert index.reconcile(["a", "b", "c"]) == set()  # nothing to drop, nothing missing
        assert index.count() == 3


class TestFilters:
    def _index(self) -> QdrantVectorIndex:
        index = QdrantVectorIndex(dim=3)
        _seed(index)
        return index

    def test_eq_filter(self) -> None:
        hits = self._index().search([1, 0, 0], k=5, where=(Predicate("g", FilterOp.EQ, "x"),))
        assert {cid for cid, _ in hits} == {"a", "c"}

    def test_ne_filter(self) -> None:
        hits = self._index().search([1, 0, 0], k=5, where=(Predicate("g", FilterOp.NE, "x"),))
        assert {cid for cid, _ in hits} == {"b"}

    def test_in_filter(self) -> None:
        hits = self._index().search([1, 0, 0], k=5, where=(Predicate("n", FilterOp.IN, [1, 2]),))
        assert {cid for cid, _ in hits} == {"a", "b"}

    def test_range_filters(self) -> None:
        hits = self._index().search([1, 0, 0], k=5, where=(Predicate("n", FilterOp.GE, 2),))
        assert {cid for cid, _ in hits} == {"b", "c"}
        hits = self._index().search([1, 0, 0], k=5, where=(Predicate("n", FilterOp.LT, 2),))
        assert {cid for cid, _ in hits} == {"a"}

    def test_empty_in_filter_refused(self) -> None:
        with pytest.raises(VectorIndexError, match="IN filter"):
            self._index().search([1, 0, 0], k=5, where=(Predicate("n", FilterOp.IN, []),))


class TestReconcilePaging:
    def test_reconcile_over_many_points(self) -> None:
        # More than one scroll page (>256), so indexed_ids exercises the pagination loop.
        index = QdrantVectorIndex(dim=3)
        ids = [f"c{i}" for i in range(300)]
        index.upsert(ids, [[1, 0, 0]] * 300, [{}] * 300)
        assert index.count() == 300
        missing = index.reconcile([*ids[:150], "new"])  # drop 150 orphans, report 1 missing
        assert missing == {"new"} and index.count() == 150
