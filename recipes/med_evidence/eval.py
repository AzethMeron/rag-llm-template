"""Decision accuracy for the biomedical evidence recipe: compare each answer's decision to the gold
PubMedQA final decision (yes / no / maybe).

The gold labels are always concrete (yes/no/maybe); the system may additionally answer
``unsupported`` (abstain). This scores the run three ways:

* **accuracy** — the produced decision equals the gold label (after lower-casing). A record that
  produced no usable answer (rejected for an ungrounded citation, say) or that abstained counts as a
  miss, never a crash.
* **abstain_rate** — the fraction of records the system answered ``unsupported`` (declined to
  decide). A high abstain rate trades coverage for safety; read alongside answered_accuracy.
* **answered_accuracy** — accuracy over only the records it actually decided (a concrete
  non-abstained decision was produced). A miss or an abstention is excluded from this denominator,
  so it measures how right the system is *when it commits*.

Run: ``PYTHONPATH=src:. python -m recipes.med_evidence.eval \\
        --journal work/med.jsonl --gold recipes/med_evidence/data/gold.jsonl``
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

_UNSUPPORTED = "unsupported"


def _abstained(outcome: Labelled) -> bool:
    return outcome.produced and outcome.predicted == _UNSUPPORTED


def _answered(outcome: Labelled) -> bool:
    """A concrete decision was produced -- not a miss, not an abstention."""
    return outcome.produced and outcome.predicted is not None and outcome.predicted != _UNSUPPORTED


@dataclass(frozen=True, slots=True)
class Report(ClassificationReport):
    """The shared counts, plus the two rates specific to a task the system may decline."""

    @property
    def answered(self) -> int:
        return sum(1 for outcome in self.outcomes if _answered(outcome))

    @property
    def abstain_rate(self) -> float:
        return self.rate(_abstained)

    @property
    def answered_accuracy(self) -> float:
        # Accuracy over only the records the system committed a concrete decision on.
        return self.rate(lambda outcome: outcome.correct, over=self.answered)


def evaluate(pairs: object, *, decision_field: str = "decision") -> Report:
    """Score ``(record_id, produced_output_or_None, gold_decision)`` triples."""
    return Report(score_labels(pairs, field=decision_field))  # type: ignore[arg-type]


def main(argv: Sequence[str] | None = None) -> int:
    parser = journal_gold_parser("Biomedical-evidence decision accuracy.",
                                 gold_help="gold decisions (JSONL)")
    args = parser.parse_args(argv)

    def build() -> str:
        gold = load_label_gold(args.gold, field="decision")
        report = evaluate(join_journal_with_gold(args.journal, gold))
        return (f"decided {report.produced}/{report.total} | "
                f"accuracy {report.accuracy:.3f} | "
                f"abstain rate {report.abstain_rate:.3f} | "
                f"answered accuracy (over {report.answered}) {report.answered_accuracy:.3f}")

    return run_report(build)


if __name__ == "__main__":  # pragma: no cover - exercised via main() in tests
    raise SystemExit(main())
