"""Retrieval metrics (against hand-computed values) and the harness's circularity guard."""
from __future__ import annotations

import math

import pytest

from ragkit.core.ports import Retrieved
from ragkit.eval import (
    CircularEvaluationError,
    IncompleteGroundTruthError,
    Qrels,
    average_precision,
    evaluate_retrieval,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


class _StubRetriever:
    """Returns a fixed ranking of ids per query, ignoring the corpus."""

    def __init__(self, rankings: dict[str, list[str]]) -> None:
        self._rankings = rankings

    def retrieve(self, query: str, *, k: int, min_score: float = 0.0) -> tuple[Retrieved, ...]:
        return tuple(Retrieved(doc_id, doc_id, 1.0) for doc_id in self._rankings[query][:k])


class TestMetrics:
    def test_recall_at_k(self) -> None:
        assert recall_at_k(["a", "b", "c"], frozenset({"a", "x"}), 3) == 0.5
        assert recall_at_k(["a", "b", "c"], frozenset({"z"}), 3) == 0.0
        assert recall_at_k(["a", "b", "c"], frozenset({"a"}), 1) == 1.0

    def test_hit_rate_at_k(self) -> None:
        assert hit_rate_at_k(["a", "b", "c"], frozenset({"a", "x"}), 3) == 1.0
        assert hit_rate_at_k(["a", "b", "c"], frozenset({"z"}), 3) == 0.0
        assert hit_rate_at_k(["a", "b", "c"], frozenset({"c"}), 2) == 0.0  # c is outside top 2
        # unlike recall_at_k, a single hit among several relevant ids still scores 1.0
        assert hit_rate_at_k(["a", "x", "y"], frozenset({"a", "b"}), 3) == 1.0

    def test_reciprocal_rank(self) -> None:
        assert reciprocal_rank(["b", "a"], frozenset({"a"})) == 0.5
        assert reciprocal_rank(["a", "b"], frozenset({"a"})) == 1.0
        assert reciprocal_rank(["x", "y"], frozenset({"a"})) == 0.0

    def test_average_precision(self) -> None:
        # a@1 -> 1/1, b@3 -> 2/3; mean over 2 relevant = (1 + 0.6667) / 2
        assert average_precision(["a", "x", "b"], frozenset({"a", "b"})) == pytest.approx(5 / 6)
        assert average_precision(["x", "y"], frozenset({"a"})) == 0.0

    def test_ndcg_at_k(self) -> None:
        # relevant at rank 2 -> dcg = 1/log2(3); ideal = 1/log2(2) = 1
        assert ndcg_at_k(["a", "b"], frozenset({"b"}), 2) == pytest.approx(1 / math.log2(3))
        assert ndcg_at_k(["b", "a"], frozenset({"b"}), 2) == 1.0  # relevant first = perfect
        assert ndcg_at_k(["x"], frozenset({"b"}), 2) == 0.0


class TestQrels:
    def test_source_is_required(self) -> None:
        with pytest.raises(ValueError, match="non-empty 'source'"):
            Qrels(relevant={"q1": frozenset({"a"})}, source="  ")


class TestEvaluateRetrieval:
    def _setup(self) -> tuple[dict, dict, Qrels]:
        systems = {
            "good": _StubRetriever({"first": ["a", "b"], "second": ["c", "d"]}),
            "bad": _StubRetriever({"first": ["x", "y"], "second": ["z", "w"]}),
        }
        queries = {"q1": "first", "q2": "second"}
        qrels = Qrels(relevant={"q1": frozenset({"a"}), "q2": frozenset({"c"})}, source="gold")
        return systems, queries, qrels

    def test_scores_rank_the_better_system_higher(self) -> None:
        systems, queries, qrels = self._setup()
        scores = evaluate_retrieval(systems, queries, qrels, k=2)
        assert scores["good"].mrr == 1.0 and scores["good"].recall_at_k == 1.0
        assert scores["good"].hit_rate_at_k == 1.0
        assert scores["bad"].mrr == 0.0 and scores["bad"].recall_at_k == 0.0
        assert scores["bad"].hit_rate_at_k == 0.0
        assert scores["good"].queries == 2 and scores["good"].k == 2

    def test_refuses_ground_truth_from_a_system_under_eval(self) -> None:
        systems, queries, _ = self._setup()
        circular = Qrels(relevant={"q1": frozenset({"a"}), "q2": frozenset({"c"})}, source="good")
        with pytest.raises(CircularEvaluationError, match="score perfectly by construction"):
            evaluate_retrieval(systems, queries, circular, k=2)

    def test_refuses_a_query_with_no_ground_truth(self) -> None:
        systems, queries, _ = self._setup()
        holes = Qrels(relevant={"q1": frozenset({"a"})}, source="gold")  # q2 missing
        with pytest.raises(IncompleteGroundTruthError, match="no ground-truth"):
            evaluate_retrieval(systems, queries, holes, k=2)

    def test_incomplete_gold_is_not_reported_as_circularity(self) -> None:
        # Distinct types, on purpose: "the gold set is incomplete" is a data problem with a
        # different fix from "the gold set is not independent", and a caller handling one must
        # not silently absorb the other.
        systems, queries, _ = self._setup()
        holes = Qrels(relevant={"q1": frozenset({"a"})}, source="gold")
        assert not issubclass(IncompleteGroundTruthError, CircularEvaluationError)
        with pytest.raises(IncompleteGroundTruthError):
            evaluate_retrieval(systems, queries, holes, k=2)

    def test_each_metric_can_have_its_own_depth(self) -> None:
        # The reason legal_procurement used to fork the metric loop (and so bypass the guards):
        # Recall@20 alongside a literature-comparable Acc@10 and MRR@10.
        systems = {"s": _StubRetriever({"first": ["x", "y", "a"]})}
        queries, qrels = {"q1": "first"}, Qrels({"q1": frozenset({"a"})}, source="gold")
        scores = evaluate_retrieval(systems, queries, qrels, k=3, hit_rate_k=2, rank_k=2)["s"]
        assert scores.recall_at_k == 1.0    # "a" is within the top 3
        assert scores.hit_rate_at_k == 0.0  # ...but not within the top 2
        assert scores.mrr == 0.0            # nor within the rank window
        assert (scores.k, scores.hit_rate_k, scores.rank_k) == (3, 2, 2)

    def test_the_per_metric_depths_default_to_k(self) -> None:
        systems, queries, qrels = self._setup()
        scores = evaluate_retrieval(systems, queries, qrels, k=2)["good"]
        assert (scores.k, scores.hit_rate_k, scores.rank_k) == (2, 2, 2)

    @pytest.mark.parametrize("depths", [{"hit_rate_k": 0}, {"rank_k": -1}])
    def test_a_non_positive_per_metric_depth_is_refused(self, depths: dict) -> None:
        systems, queries, qrels = self._setup()
        with pytest.raises(ValueError, match="must be >= 1"):
            evaluate_retrieval(systems, queries, qrels, k=2, **depths)

    def test_rejects_bad_k_and_empty_systems(self) -> None:
        _, queries, qrels = self._setup()
        with pytest.raises(ValueError, match="k must be >= 1"):
            evaluate_retrieval({"good": _StubRetriever({})}, {}, qrels, k=0)
        with pytest.raises(ValueError, match="no systems"):
            evaluate_retrieval({}, queries, qrels, k=2)
