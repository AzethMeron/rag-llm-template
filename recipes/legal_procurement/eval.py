"""Retrieval quality for the legal-procurement recipe: how well the reference retriever surfaces the
gold-relevant legal passages for each held-out question.

This is the recipe that exercises the framework's retrieval-metrics layer
(:mod:`ragkit.eval.retrieval`). The headline is three standard rank-quality numbers, each at its
conventional depth, averaged over the evaluated questions:

* **Recall@20** — the fraction of a question's gold passages that appear in the top 20 retrieved;
* **MRR@10** — the mean reciprocal rank of the first gold passage within the top 10;
* **NDCG@10** — the normalised discounted cumulative gain of the top 10 (binary relevance).

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

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ragkit.cli.app import assemble
from ragkit.core.ports import Retriever
from ragkit.core.records import read_journal
from ragkit.eval.retrieval import ndcg_at_k, recall_at_k, reciprocal_rank

_NDCG_K = 10
_MRR_K = 10


class EvalError(Exception):
    """The evaluation cannot run as configured (missing/malformed gold, heldout, or retriever)."""


@dataclass(frozen=True, slots=True)
class QueryScore:
    record_id: str
    recall: float
    reciprocal_rank: float
    ndcg: float


@dataclass(frozen=True, slots=True)
class Report:
    scores: tuple[QueryScore, ...]
    k: int

    @property
    def queries(self) -> int:
        return len(self.scores)

    @property
    def recall_at_k(self) -> float:
        return self._mean(lambda s: s.recall)

    @property
    def mrr(self) -> float:
        return self._mean(lambda s: s.reciprocal_rank)

    @property
    def ndcg(self) -> float:
        return self._mean(lambda s: s.ndcg)

    def _mean(self, pick: Callable[[QueryScore], float]) -> float:
        return sum(pick(s) for s in self.scores) / self.queries if self.scores else 0.0


def evaluate(retriever: Retriever, queries: Sequence[tuple[str, str]],
             gold: Mapping[str, frozenset[str]], *, k: int = 20) -> Report:
    """Score each ``(record_id, question)`` whose id has gold judgments, retrieving at a depth that
    covers every reported metric (``max(k, 10)``). Recall is at ``k``; MRR and NDCG at 10."""
    depth = max(k, _MRR_K, _NDCG_K)
    scores: list[QueryScore] = []
    for record_id, question in queries:
        relevant = gold.get(record_id)
        if not relevant:
            continue
        ranked = [hit.chunk_id for hit in retriever.retrieve(question, k=depth)]
        scores.append(QueryScore(
            record_id=record_id,
            recall=recall_at_k(ranked, relevant, k),
            reciprocal_rank=reciprocal_rank(ranked[:_MRR_K], relevant),
            ndcg=ndcg_at_k(ranked, relevant, _NDCG_K)))
    return Report(tuple(scores), k=k)


def load_gold(path: Path) -> dict[str, frozenset[str]]:
    if not path.is_file():
        raise EvalError(f"gold file not found: {path}")
    gold: dict[str, frozenset[str]] = {}
    for line_no, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalError(f"{path}:{line_no}: invalid JSON in gold file: {exc}") from exc
        if "record_id" not in row or "relevant" not in row:
            raise EvalError(f"{path}:{line_no}: a gold row needs 'record_id' and 'relevant'")
        relevant = row["relevant"]
        if not isinstance(relevant, list) or not relevant:
            raise EvalError(f"{path}:{line_no}: 'relevant' must be a non-empty list of passage ids")
        gold[str(row["record_id"])] = frozenset(str(x) for x in relevant)
    if not gold:
        raise EvalError(f"gold file is empty: {path}")
    return gold


def load_heldout(path: Path) -> list[tuple[str, str]]:
    if not path.is_file():
        raise EvalError(f"heldout file not found: {path}")
    queries: list[tuple[str, str]] = []
    for line_no, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalError(f"{path}:{line_no}: invalid JSON in heldout file: {exc}") from exc
        if "record_id" not in row or "source" not in row:
            raise EvalError(f"{path}:{line_no}: a heldout row needs 'record_id' and 'source'")
        queries.append((str(row["record_id"]), str(row["source"])))
    if not queries:
        raise EvalError(f"heldout file is empty: {path}")
    return queries


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
    parser = argparse.ArgumentParser(description="Legal-procurement retrieval quality vs gold.")
    parser.add_argument("--config", type=Path, required=True, help="recipe config directory")
    parser.add_argument("--heldout", type=Path, required=True, help="held-out questions (JSONL)")
    parser.add_argument("--gold", type=Path, required=True,
                        help="gold relevant passage ids (JSONL)")
    parser.add_argument("--k", type=int, default=20, help="Recall@k depth (default: 20)")
    parser.add_argument("--journal", type=Path, default=None,
                        help="optional run journal, to also report citation-grounding rate")
    args = parser.parse_args(argv)
    try:
        if args.k < 1:
            raise EvalError(f"--k must be >= 1, got {args.k}")
        gold = load_gold(args.gold)
        queries = load_heldout(args.heldout)
        retriever = assemble(args.config).retriever
        if retriever is None:
            raise EvalError(f"the recipe at {args.config} wires no retriever to evaluate")
        report = evaluate(retriever, queries, gold, k=args.k)
        summary = (f"queries {report.queries} | Recall@{report.k} {report.recall_at_k:.3f} | "
                   f"MRR@{_MRR_K} {report.mrr:.3f} | NDCG@{_NDCG_K} {report.ndcg:.3f}")
        if args.journal is not None:
            grounded, produced = grounding_rate(
                retriever, args.journal, gold, dict(queries), k=args.k)
            rate = grounded / produced if produced else 0.0
            summary += f" | citation-grounding {grounded}/{produced} ({rate:.3f})"
    except EvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(summary)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via main() in tests
    raise SystemExit(main())
