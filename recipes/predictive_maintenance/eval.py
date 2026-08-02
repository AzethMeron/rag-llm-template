"""Decision accuracy for the predictive-maintenance recipe: compare each decision's severity to the
gold severity derived from the engine's remaining useful life.

Gold severity comes from the held-out RUL (``urgent`` <= 30 cycles, ``watch`` <= 80, else
``normal``). This scores the decision two ways:

* **exact** — the predicted severity equals the gold bucket;
* **actionable** — whether the decision correctly separated "needs attention" (watch/urgent) from
  "normal", which is the call that actually drives maintenance, forgiving a watch/urgent mix-up.

A record that produced no usable decision (rejected for an ungrounded citation, say) counts as a
miss, never a crash.

Run: ``PYTHONPATH=src:. python -m recipes.predictive_maintenance.eval \\
        --journal work/pdm.jsonl --gold recipes/predictive_maintenance/data/gold.jsonl``
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ragkit.eval.classify import ClassificationReport, Labelled, score_labels
from ragkit.eval.gold import (
    join_journal_with_gold,
    journal_gold_parser,
    load_label_gold,
    run_report,
)

_ATTENTION = frozenset({"watch", "urgent"})


def _actionable(outcome: Labelled) -> bool:
    """Did it get the needs-attention-vs-normal call right, even if it confused watch/urgent? A
    record with no decision is a miss, not a lucky "normal" match."""
    return outcome.produced and (outcome.predicted in _ATTENTION) == (outcome.gold in _ATTENTION)


@dataclass(frozen=True, slots=True)
class Report(ClassificationReport):
    """The shared counts, plus the coarser call that actually drives maintenance."""

    @property
    def exact_accuracy(self) -> float:
        return self.accuracy

    @property
    def actionable_accuracy(self) -> float:
        return self.rate(_actionable)


def evaluate(pairs: object, *, severity_field: str = "severity") -> Report:
    """Score ``(record_id, produced_output_or_None, gold_severity)`` triples."""
    return Report(score_labels(pairs, field=severity_field))  # type: ignore[arg-type]


def main(argv: Sequence[str] | None = None) -> int:
    parser = journal_gold_parser("Predictive-maintenance decision accuracy.",
                                 gold_help="gold severities (JSONL)")
    args = parser.parse_args(argv)

    def build() -> str:
        gold = load_label_gold(args.gold, field="severity")
        report = evaluate(join_journal_with_gold(args.journal, gold))
        return (f"decided {report.produced}/{report.total} | "
                f"exact severity {report.exact_accuracy:.3f} | "
                f"actionable (attention vs normal) {report.actionable_accuracy:.3f}")

    return run_report(build)


if __name__ == "__main__":  # pragma: no cover - exercised via main() in tests
    raise SystemExit(main())
