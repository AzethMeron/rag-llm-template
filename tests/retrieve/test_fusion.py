"""The pure fusion math: RRF, the fixed-denominator scale, and MMR — with the non-finite guards."""
from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ragkit.retrieve.fusion import best_possible_rrf, mmr_order, rrf


class TestRrf:
    def test_agreement_outranks_a_single_arm(self) -> None:
        fused = rrf([["a", "b"], ["b", "c"]])
        assert fused["b"] > fused["a"] and fused["b"] > fused["c"]

    def test_rank_matters(self) -> None:
        fused = rrf([["a", "b", "c"]])
        assert fused["a"] > fused["b"] > fused["c"]

    def test_bad_k(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            rrf([["a"]], k=0)

    def test_nan_k_is_refused(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            rrf([["a"]], k=math.nan)


class TestBestPossible:
    def test_scale(self) -> None:
        assert best_possible_rrf(2) == pytest.approx(2 / 61)

    def test_bad_count(self) -> None:
        with pytest.raises(ValueError, match="ranking_count"):
            best_possible_rrf(0)

    def test_bad_k(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            best_possible_rrf(2, k=0)

    def test_nan_count(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            best_possible_rrf(math.nan)


class TestMmr:
    def _sim(self, a: str, b: str) -> float:
        return 1.0 if a == b else 0.5

    def test_pure_relevance_when_lambda_one(self) -> None:
        order = mmr_order(["a", "b", "c"], {"a": 0.5, "b": 0.9, "c": 0.7}, self._sim,
                          lambda_=1.0, k=3)
        assert order == ["b", "c", "a"]

    def test_diversity_penalises_similar(self) -> None:
        # a and b are identical (sim 1.0); with diversity weight, the second pick avoids b.
        def sim(x: str, y: str) -> float:
            return 1.0 if {x, y} <= {"a", "b"} else 0.0
        order = mmr_order(["a", "b", "c"], {"a": 0.9, "b": 0.85, "c": 0.5}, sim,
                          lambda_=0.5, k=2)
        assert order == ["a", "c"]

    def test_k_zero(self) -> None:
        assert mmr_order(["a"], {"a": 1.0}, self._sim, lambda_=0.7, k=0) == []

    def test_bad_lambda(self) -> None:
        with pytest.raises(ValueError, match="lambda_"):
            mmr_order(["a"], {"a": 1.0}, self._sim, lambda_=2.0, k=1)

    def test_missing_relevance(self) -> None:
        with pytest.raises(ValueError, match="no relevance"):
            mmr_order(["a"], {}, self._sim, lambda_=0.7, k=1)

    def test_non_finite_relevance(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            mmr_order(["a"], {"a": math.inf}, self._sim, lambda_=0.7, k=1)

    def test_non_finite_similarity(self) -> None:
        def bad(_a: str, _b: str) -> float:
            return math.nan
        with pytest.raises(ValueError, match="finite"):
            mmr_order(["a", "b"], {"a": 1.0, "b": 1.0}, bad, lambda_=0.5, k=2)


class TestProperties:
    @given(st.lists(st.lists(st.integers(0, 20), max_size=8), max_size=4))
    def test_rrf_scores_are_positive(self, rankings: list[list[int]]) -> None:
        for score in rrf(rankings).values():
            assert score > 0

    @given(st.lists(st.text(min_size=1), min_size=1, max_size=6, unique=True))
    def test_mmr_returns_a_subset_in_bounds(self, items: list[str]) -> None:
        relevance = dict.fromkeys(items, 0.5)
        result = mmr_order(items, relevance, lambda a, b: 1.0 if a == b else 0.0,
                           lambda_=0.7, k=3)
        assert len(result) <= min(3, len(items))
        assert set(result) <= set(items) and len(set(result)) == len(result)
