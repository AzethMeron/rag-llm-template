"""The LanceDB vector index driver."""
from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.core.ports import FilterOp, Predicate
from ragkit.store.vector.lancedb import LanceVectorIndex, VectorIndexError


def _index(tmp_path: Path, dim: int = 3) -> LanceVectorIndex:
    return LanceVectorIndex(str(tmp_path / "vec.lance"), dim=dim)


class TestBasics:
    def test_upsert_search_count(self, tmp_path: Path) -> None:
        index = _index(tmp_path)
        index.upsert(["a", "b"], [[1, 0, 0], [0, 1, 0]], [{"lang": "en"}, {"lang": "pl"}])
        assert index.count() == 2
        results = index.search([1, 0, 0], k=2)
        assert results[0][0] == "a" and results[0][1] > results[1][1]

    def test_score_is_higher_is_better_in_unit_interval(self, tmp_path: Path) -> None:
        index = _index(tmp_path)
        index.upsert(["a"], [[1, 0, 0]], [{}])
        [(chunk_id, score)] = index.search([1, 0, 0], k=1)
        assert chunk_id == "a" and 0.99 <= score <= 1.0  # identical vector -> ~1.0

    def test_upsert_replaces(self, tmp_path: Path) -> None:
        index = _index(tmp_path)
        index.upsert(["a"], [[1, 0, 0]], [{}])
        index.upsert(["a"], [[0, 0, 1]], [{}])  # replace, not duplicate
        assert index.count() == 1

    def test_delete(self, tmp_path: Path) -> None:
        index = _index(tmp_path)
        index.upsert(["a", "b"], [[1, 0, 0], [0, 1, 0]], [{}, {}])
        index.delete(["a"])
        assert index.count() == 1

    def test_metadata_filter(self, tmp_path: Path) -> None:
        index = _index(tmp_path)
        index.upsert(["a", "b"], [[1, 0, 0], [1, 0, 0]], [{}, {}])
        results = index.search([1, 0, 0], k=5, where=(Predicate("id", FilterOp.EQ, "b"),))
        assert [chunk_id for chunk_id, _ in results] == ["b"]

    def test_search_empty_index(self, tmp_path: Path) -> None:
        assert _index(tmp_path).search([1, 0, 0], k=5) == []

    def test_search_k_zero(self, tmp_path: Path) -> None:
        index = _index(tmp_path)
        index.upsert(["a"], [[1, 0, 0]], [{}])
        assert index.search([1, 0, 0], k=0) == []


class TestReconcile:
    def test_drops_orphans_and_reports_missing(self, tmp_path: Path) -> None:
        index = _index(tmp_path)
        index.upsert(["a", "b"], [[1, 0, 0], [0, 1, 0]], [{}, {}])
        missing = index.reconcile(["a", "c"])  # b is an orphan; c is missing
        assert missing == {"c"}
        assert index.indexed_ids() == {"a"}

    def test_no_drift_returns_empty(self, tmp_path: Path) -> None:
        index = _index(tmp_path)
        index.upsert(["a"], [[1, 0, 0]], [{}])
        assert index.reconcile(["a"]) == set()


class TestValidation:
    def test_dim_must_be_positive(self, tmp_path: Path) -> None:
        with pytest.raises(VectorIndexError, match="positive dim"):
            LanceVectorIndex(str(tmp_path / "v"), dim=0)

    def test_mismatched_upsert_lengths(self, tmp_path: Path) -> None:
        with pytest.raises(VectorIndexError, match="mismatched lengths"):
            _index(tmp_path).upsert(["a"], [[1, 0, 0], [0, 1, 0]], [{}])

    def test_wrong_vector_dim_on_upsert(self, tmp_path: Path) -> None:
        with pytest.raises(VectorIndexError, match="dimension"):
            _index(tmp_path).upsert(["a"], [[1, 0]], [{}])

    def test_wrong_query_dim(self, tmp_path: Path) -> None:
        index = _index(tmp_path)
        index.upsert(["a"], [[1, 0, 0]], [{}])
        with pytest.raises(VectorIndexError, match="dimension"):
            index.search([1, 0], k=1)

    def test_empty_upsert_is_a_noop(self, tmp_path: Path) -> None:
        index = _index(tmp_path)
        index.upsert([], [], [])
        assert index.count() == 0

    def test_reopen_existing_table(self, tmp_path: Path) -> None:
        path = str(tmp_path / "v.lance")
        LanceVectorIndex(path, dim=3).upsert(["a"], [[1, 0, 0]], [{}])
        assert LanceVectorIndex(path, dim=3).count() == 1  # reopened, not recreated


class TestConfig:
    def test_from_config(self, tmp_path: Path) -> None:
        index = LanceVectorIndex.from_config({"path": str(tmp_path / "v"), "dim": 4})
        assert index.count() == 0

    def test_from_config_needs_path(self) -> None:
        with pytest.raises(VectorIndexError, match="needs a 'path'"):
            LanceVectorIndex.from_config({"dim": 4})

    def test_close_is_a_noop(self, tmp_path: Path) -> None:
        _index(tmp_path).close()

    def test_empty_delete_is_a_noop(self, tmp_path: Path) -> None:
        index = _index(tmp_path)
        index.upsert(["a"], [[1, 0, 0]], [{}])
        index.delete([])
        assert index.count() == 1

    def test_l2_metric_scoring(self, tmp_path: Path) -> None:
        index = LanceVectorIndex(str(tmp_path / "v"), dim=3, metric="l2")
        index.upsert(["a"], [[1, 0, 0]], [{}])
        [(chunk_id, score)] = index.search([1, 0, 0], k=1)
        assert chunk_id == "a" and 0.0 < score <= 1.0

    def test_opening_an_invalid_path_is_structured(self, tmp_path: Path) -> None:
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("i am a file", encoding="utf-8")
        with pytest.raises(VectorIndexError, match="could not open"):
            LanceVectorIndex(str(blocker), dim=3)

    def test_missing_dependency_is_named(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        from ragkit.store.vector import lancedb as driver
        monkeypatch.setitem(sys.modules, "lancedb", None)
        with pytest.raises(VectorIndexError, match="needs 'lancedb'"):
            driver._require_lancedb()
