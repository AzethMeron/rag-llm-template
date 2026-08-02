"""Generalised evaluation: retrieval rank-quality metrics and blinded A/B output judging, both
built so a circular configuration (a metric entangled with the thing it ranks) is refused rather
than reported. See :mod:`ragkit.eval.retrieval` and :mod:`ragkit.eval.judge`."""
from __future__ import annotations

from .classify import ClassificationReport, Labelled, score_labels
from .gold import (
    EvalError,
    Pair,
    gold_rows,
    join_journal_with_gold,
    journal_gold_parser,
    json_field,
    load_fields_gold,
    load_label_gold,
    load_relevance_gold,
    run_report,
)
from .judge import AbItem, AbSummary, AbVerdict, JudgeError, evaluate_ab
from .retrieval import (
    CircularEvaluationError,
    IncompleteGroundTruthError,
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
    "CircularEvaluationError", "IncompleteGroundTruthError", "Qrels", "RetrievalScores",
    "evaluate_retrieval",
    "recall_at_k", "hit_rate_at_k", "reciprocal_rank", "average_precision", "ndcg_at_k",
    "AbItem", "AbSummary", "AbVerdict", "JudgeError", "evaluate_ab",
    # the shared recipe-eval scaffolding
    "EvalError", "Pair", "gold_rows", "load_label_gold", "load_fields_gold",
    "load_relevance_gold", "join_journal_with_gold", "json_field", "journal_gold_parser",
    "run_report",
    "ClassificationReport", "Labelled", "score_labels",
]
