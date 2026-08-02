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


class TestUpsertIsOneTransaction:
    """Regression: upsert was `delete(ids)` then `add(rows)` -- two transactions, so a crash
    between them removed the old vectors without adding the new, and each call wrote two
    fragments instead of one. It is now a single native merge_insert."""

    def test_upsert_does_not_delete_first(self, tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
        index = _index(tmp_path)
        index.upsert(["a"], [[1, 0, 0]], [{}])

        def _boom(_ids: object) -> None:
            raise AssertionError("upsert must not delete-then-add; that is the torn-write window")

        monkeypatch.setattr(index, "delete", _boom)
        index.upsert(["a", "b"], [[0, 0, 1], [0, 1, 0]], [{}, {}])
        assert index.count() == 2
        assert index.search([0, 0, 1], k=1)[0][0] == "a"  # the replacement, not the old vector

    def test_matched_and_unmatched_rows_in_one_call(self, tmp_path: Path) -> None:
        index = _index(tmp_path)
        index.upsert(["a", "b"], [[1, 0, 0], [0, 1, 0]], [{"v": 1}, {"v": 1}])
        index.upsert(["a", "c"], [[0, 0, 1], [1, 1, 0]], [{"v": 2}, {"v": 2}])
        assert index.count() == 3
        assert index.indexed_ids() == {"a", "b", "c"}

    def test_a_backend_failure_is_a_structured_error(self, tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
        index = _index(tmp_path)

        def flaky(_on: object) -> None:
            raise RuntimeError("merge boom")

        monkeypatch.setattr(index._table, "merge_insert", flaky)
        with pytest.raises(VectorIndexError, match="could not upsert 1 row"):
            index.upsert(["a"], [[1, 0, 0]], [{}])

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

    def test_indexed_ids_never_materializes_full_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression test: indexed_ids() must project only the "id" column at the LanceDB scan
        level. table.to_arrow() (no column argument) would instead pull every column -- including
        the full-width vector and meta JSON -- for every row just to discard all but "id", which at
        millions of rows is tens of GB for a single call (the cause of a real RAM-exhaustion/disk-
        thrash incident). Failing this test by calling to_arrow() again would reintroduce that."""
        index = _index(tmp_path)
        index.upsert(["a", "b"], [[1, 0, 0], [0, 1, 0]], [{}, {}])

        def _boom() -> None:
            raise AssertionError("indexed_ids() must not call table.to_arrow()")

        monkeypatch.setattr(index._table, "to_arrow", _boom)
        assert index.indexed_ids() == {"a", "b"}


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


class TestCompact:
    def test_compact_reduces_fragment_count_and_preserves_data(self, tmp_path: Path) -> None:
        index = _index(tmp_path)
        vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]]
        for i, vector in enumerate(vectors):  # a separate upsert call each -- its own fragment
            index.upsert([f"c{i}"], [vector], [{"n": i}])
        before_files = list((tmp_path / "vec.lance" / "chunks.lance" / "data").glob("*"))
        assert len(before_files) >= 5

        index.compact()

        after_files = list((tmp_path / "vec.lance" / "chunks.lance" / "data").glob("*"))
        assert len(after_files) < len(before_files)
        assert index.count() == 5
        assert index.indexed_ids() == {f"c{i}" for i in range(5)}
        assert index.search([0, 0, 1], k=1)[0][0] == "c2"

    def test_compact_on_an_empty_table_is_a_noop(self, tmp_path: Path) -> None:
        index = _index(tmp_path)
        index.compact()
        assert index.count() == 0

    def test_missing_pylance_is_a_structured_error(self, tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        index = _index(tmp_path)
        monkeypatch.setitem(sys.modules, "lance", None)
        with pytest.raises(VectorIndexError, match="needs 'pylance'"):
            index.compact()

    def test_optimize_failure_is_a_structured_error(self, tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
        index = _index(tmp_path)
        index.upsert(["a"], [[1, 0, 0]], [{}])

        def flaky(**_kwargs: object) -> None:
            raise RuntimeError("optimize boom")

        monkeypatch.setattr(index._table, "optimize", flaky)
        with pytest.raises(VectorIndexError, match="could not compact"):
            index.compact()


class TestCreateIndex:
    def test_build_index_and_search_still_finds_nearest(self, tmp_path: Path) -> None:
        index = _index(tmp_path)
        vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]]
        index.upsert([f"c{i}" for i in range(5)], vectors, [{} for _ in vectors])
        assert index._table.list_indices() == []

        index.create_index(num_partitions=1)

        assert len(index._table.list_indices()) == 1
        assert index.search([0, 0, 1], k=1)[0][0] == "c2"
        assert index.count() == 5  # index build must not touch row count

    def test_create_index_on_empty_table_is_a_structured_error(self, tmp_path: Path) -> None:
        with pytest.raises(VectorIndexError, match="empty table"):
            _index(tmp_path).create_index()

    def test_default_num_partitions_is_sqrt_row_count(self, tmp_path: Path) -> None:
        index = _index(tmp_path)
        vectors = [[float(i % 2), float((i + 1) % 2), 0.0] for i in range(9)]
        index.upsert([f"c{i}" for i in range(9)], vectors, [{} for _ in vectors])
        index.create_index()  # 9 rows -> sqrt(9) = 3 partitions, no explicit override
        [info] = index._table.list_indices()
        assert info.index_type == "IvfFlat"

    def test_build_failure_is_a_structured_error(self, tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
        index = _index(tmp_path)
        index.upsert(["a"], [[1, 0, 0]], [{}])

        def flaky(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("index boom")

        monkeypatch.setattr(index._table, "create_index", flaky)
        with pytest.raises(VectorIndexError, match="could not build the vector index"):
            index.create_index()


class TestSearchNprobes:
    """Regression for the silent recall drop the ANN index introduced: search() built its query
    without ever setting nprobes, so every query used LanceDB's small fixed default -- under 1% of
    the ~2,650 partitions a 7.1M-row table gets."""

    def test_search_sets_nprobes_on_the_query(self, tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
        index = _index(tmp_path)
        index.upsert(["a"], [[1, 0, 0]], [{}])
        seen: list[int] = []
        build = index._table.search

        def spy(*args: object, **kwargs: object) -> object:
            builder = build(*args, **kwargs)
            set_nprobes = builder.nprobes
            monkeypatch.setattr(builder, "nprobes",
                                lambda value: (seen.append(value), set_nprobes(value))[1])
            return builder

        monkeypatch.setattr(index._table, "search", spy)
        index.search([1, 0, 0], k=1)
        assert seen == [index.search_nprobes(1)]  # 1 row -> 1 partition -> the floor

    @pytest.mark.parametrize(("rows", "expected"), [
        (1, 20),            # 1 partition; the floor keeps it at LanceDB's own default
        (10_000, 20),       # 100 partitions -> 5% = 5, still floored
        (1_000_000, 50),    # 1,000 partitions -> 5%
        (7_100_000, 134),   # legal_procurement's real scale: ~2,664 partitions -> 5%
    ])
    def test_default_scales_with_the_partition_count(self, tmp_path: Path, rows: int,
                                                     expected: int) -> None:
        assert _index(tmp_path).search_nprobes(rows) == expected

    def test_configured_nprobes_wins(self, tmp_path: Path) -> None:
        index = LanceVectorIndex(str(tmp_path / "v"), dim=3, nprobes=7)
        assert index.search_nprobes(7_100_000) == 7

    def test_nprobes_must_be_positive(self, tmp_path: Path) -> None:
        with pytest.raises(VectorIndexError, match="nprobes must be >= 1"):
            LanceVectorIndex(str(tmp_path / "v"), dim=3, nprobes=0)

    def test_probing_more_partitions_recovers_recall(self, tmp_path: Path) -> None:
        """The behavioural half: over one real IVF index, the default nprobes finds neighbours a
        single-partition probe misses. Both readers open the *same* on-disk table, so the k-means
        training is shared and the only variable is how many partitions each query probes."""
        import numpy as np

        rng = np.random.default_rng(20260802)
        dim, rows = 16, 600
        vectors = rng.random((rows, dim), dtype=np.float32)
        path = str(tmp_path / "recall.lance")
        writer = LanceVectorIndex(path, dim=dim)
        writer.upsert([f"c{i}" for i in range(rows)], vectors.tolist(), [{} for _ in range(rows)])
        writer.create_index()  # sqrt(600) -> 24 partitions

        queries = rng.random((40, dim), dtype=np.float32)
        unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        exact = [f"c{i}" for i in
                 (unit @ (queries / np.linalg.norm(queries, axis=1, keepdims=True)).T).argmax(0)]

        def recall(index: LanceVectorIndex) -> float:
            hits = sum(index.search(q.tolist(), k=1)[0][0] == want
                       for q, want in zip(queries, exact, strict=True))
            return hits / len(queries)

        stingy = recall(LanceVectorIndex(path, dim=dim, nprobes=1))
        default = recall(LanceVectorIndex(path, dim=dim))
        assert default > stingy and default >= 0.9


class TestConfig:
    def test_from_config(self, tmp_path: Path) -> None:
        index = LanceVectorIndex.from_config({"path": str(tmp_path / "v"), "dim": 4})
        assert index.count() == 0

    def test_from_config_needs_path(self) -> None:
        with pytest.raises(VectorIndexError, match="needs a 'path'"):
            LanceVectorIndex.from_config({"dim": 4})

    def test_from_config_reads_nprobes(self, tmp_path: Path) -> None:
        index = LanceVectorIndex.from_config({"path": str(tmp_path / "v"), "dim": 4,
                                              "nprobes": 64})
        assert index.search_nprobes(7_100_000) == 64
        assert "nprobes" in LanceVectorIndex.CONFIG_KEYS  # else load_storage would refuse the key

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
