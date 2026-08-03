"""Retrieval quality for the legal-procurement recipe: how well the reference retriever surfaces the
gold-relevant legal passages for each held-out question.

This is the recipe that exercises the framework's retrieval-metrics layer
(:mod:`ragkit.eval.retrieval`). The headline is four rank-quality numbers, each at its
conventional depth, averaged over the evaluated questions:

* **Recall@20** — the fraction of a question's gold passages that appear in the top 20 retrieved;
* **MRR@10** — the mean reciprocal rank of the first gold passage within the top 10;
* **NDCG@10** — the normalised discounted cumulative gain of the top 10 (binary relevance);
* **Acc@10** — 1/0 per question for whether *any* gold passage lands in the top 10, averaged. This
  is NOT the same statistic as Recall@20 (different depth, and binary hit vs. fraction-of-all-
  relevant) — it exists so this recipe's number is directly comparable to papers that report
  "top-k accuracy" for their own (typically task-fine-tuned) retriever, e.g. PolQA's polqa corpus
  (Rybak et al., 2022) reports 51-62% top-10 accuracy for a HerBERT retriever fine-tuned on PolQA's
  own training set — not directly comparable to Recall@20, but directly comparable to Acc@10.

The retriever is built the SAME way the recipe builds it (``assemble(config).retriever``), so the
number reflects the recipe's own configuration, not a re-implementation. Gold judgments come from
polqa (an independent source), so no metric is true by construction.

Optionally, given a run ``--journal``, it also reports the **citation-grounding rate**: the fraction
of produced answers whose citations all fall within the union of the question's gold and the
passages retrieved for it. This needs a model run; the retrieval metrics do not.

Run: ``PYTHONPATH=src:. python -m recipes.legal_procurement.eval \\
        --config recipes/legal_procurement/config \\
        --heldout recipes/legal_procurement/data/heldout.jsonl \\
        --gold recipes/legal_procurement/data/gold.jsonl``
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from ragkit.cli.app import assemble
from ragkit.core.ports import Retriever
from ragkit.core.records import read_journal
from ragkit.eval.gold import EvalError, gold_rows, load_relevance_gold, run_report
from ragkit.eval.retrieval import Qrels, RetrievalScores, evaluate_retrieval

_NDCG_K = 10
_MRR_K = 10
_HIT_RATE_K = 10  # matches PolQA's (Rybak et al., 2022) "top-10 accuracy" retriever metric
_SYSTEM = "recipe-retriever"
_GOLD_SOURCE = "polqa"
"""Where the judgments come from -- an independent dataset, not any system evaluated here. The
guard in ``evaluate_retrieval`` refuses a run whose gold names a system under evaluation."""


def evaluate(retriever: Retriever, queries: Sequence[tuple[str, str]],
             gold: Mapping[str, frozenset[str]], *, k: int = 20) -> RetrievalScores:
    """Score the recipe's retriever, at each metric's conventional depth.

    Routed through ``ragkit.eval.evaluate_retrieval`` rather than reimplementing the metric loop.
    The fork existed only because the shared evaluator scored every metric at one depth; it also
    meant this recipe -- the only one running retrieval eval -- bypassed the circularity and
    missing-gold guards the shared evaluator applies before doing any work.
    """
    asked = {record_id: question for record_id, question in queries if gold.get(record_id)}
    qrels = Qrels(relevant={rid: gold[rid] for rid in asked}, source=_GOLD_SOURCE)
    scores = evaluate_retrieval({_SYSTEM: retriever}, asked, qrels,
                                k=k, hit_rate_k=_HIT_RATE_K, rank_k=_MRR_K)
    return scores[_SYSTEM]


def load_gold(path: Path) -> dict[str, frozenset[str]]:
    return load_relevance_gold(path, field="relevant")


def load_heldout(path: Path) -> list[tuple[str, str]]:
    """The held-out questions, as ``(record_id, question)`` in file order."""
    if not path.is_file():
        raise EvalError(f"heldout file not found: {path}")
    return [(str(row["record_id"]), str(row["source"]))
            for row in gold_rows(path, required=("record_id", "source"))]


def grounding_rate(retriever: Retriever, journal: Path, gold: Mapping[str, frozenset[str]],
                   queries: Mapping[str, str], *, k: int, citations_field: str = "citations"
                   ) -> tuple[int, int]:
    """Of the answers a run produced, how many cite only ids within the question's gold OR the
    passages retrieved for it. Returns ``(grounded, produced)``."""
    grounded = produced = 0
    for record in read_journal(journal):
        if not record.status.is_injectable or record.output is None:
            continue
        try:
            answer = json.loads(record.output)
        except json.JSONDecodeError:
            answer = None
        cited = answer.get(citations_field) if isinstance(answer, dict) else None
        if not isinstance(cited, list):
            continue
        produced += 1
        question = queries.get(record.record_id, record.source)
        allowed = set(gold.get(record.record_id, frozenset()))
        allowed.update(hit.chunk_id for hit in retriever.retrieve(question, k=k))
        if all(str(c) in allowed for c in cited):
            grounded += 1
    return grounded, produced


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Legal-procurement retrieval quality vs gold.")
    parser.add_argument("--config", type=Path, required=True, help="recipe config directory")
    parser.add_argument("--heldout", type=Path, required=True, help="held-out questions (JSONL)")
    parser.add_argument("--gold", type=Path, required=True,
                        help="gold relevant passage ids (JSONL)")
    parser.add_argument("--k", type=int, default=20, help="Recall@k depth (default: 20)")
    parser.add_argument("--journal", type=Path, default=None,
                        help="optional run journal, to also report citation-grounding rate")
    args = parser.parse_args(argv)

    def build() -> str:
        if args.k < 1:
            raise EvalError(f"--k must be >= 1, got {args.k}")
        gold = load_gold(args.gold)
        queries = load_heldout(args.heldout)
        retriever = assemble(args.config).retriever
        if retriever is None:
            raise EvalError(f"the recipe at {args.config} wires no retriever to evaluate")
        report = evaluate(retriever, queries, gold, k=args.k)
        summary = (f"queries {report.queries} | Recall@{report.k} {report.recall_at_k:.3f} | "
                   f"MRR@{_MRR_K} {report.mrr:.3f} | NDCG@{_NDCG_K} {report.ndcg_at_k:.3f} | "
                   f"Acc@{_HIT_RATE_K} {report.hit_rate_at_k:.3f} (literature-comparable "
                   f"top-{_HIT_RATE_K} hit rate, e.g. PolQA)")
        if args.journal is not None:
            grounded, produced = grounding_rate(
                retriever, args.journal, gold, dict(queries), k=args.k)
            rate = grounded / produced if produced else 0.0
            summary += f" | citation-grounding {grounded}/{produced} ({rate:.3f})"
        return summary

    return run_report(build)


if __name__ == "__main__":  # pragma: no cover - exercised via main() in tests
    raise SystemExit(main())
