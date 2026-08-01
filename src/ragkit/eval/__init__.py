"""Generalised evaluation: retrieval rank-quality metrics and blinded A/B output judging, both
built so a circular configuration (a metric entangled with the thing it ranks) is refused rather
than reported. See :mod:`ragkit.eval.retrieval` and :mod:`ragkit.eval.judge`."""
from __future__ import annotations

from .judge import AbItem, AbSummary, AbVerdict, JudgeError, evaluate_ab
from .retrieval import (
    CircularEvaluationError,
    Qrels,
    RetrievalScores,
    average_precision,
    evaluate_retrieval,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "CircularEvaluationError", "Qrels", "RetrievalScores", "evaluate_retrieval",
    "recall_at_k", "hit_rate_at_k", "reciprocal_rank", "average_precision", "ndcg_at_k",
    "AbItem", "AbSummary", "AbVerdict", "JudgeError", "evaluate_ab",
]
