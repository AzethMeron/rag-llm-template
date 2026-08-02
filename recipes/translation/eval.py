"""Translation quality against held-out gold references.

Two numbers, because a single gold reference is a harsh yardstick for translation (many renderings
are equally correct, so exact equality is rare even for a good translation):

* **exact** — the produced translation equals the gold reference after case/whitespace
  normalisation; a strict lower bound, not the whole story.
* **consistency** — the mean character-trigram cosine similarity to the gold reference (the same
  metric the framework's retrieval uses), which credits a correct translation that merely phrases
  the reference differently. Read the two together: high consistency with low exact means valid,
  varied wording; low consistency means the meaning drifted.

A record that produced no usable translation (rejected/pending) counts as a miss on both, never a
crash. Run: ``PYTHONPATH=src:. python -m recipes.translation.eval \\
        --journal work/out.jsonl --gold recipes/translation/data/gold.jsonl``
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ragkit.eval.gold import (
    Pair,
    join_journal_with_gold,
    journal_gold_parser,
    load_label_gold,
    run_report,
)
from ragkit.retrieve import trigram_similarity


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


@dataclass(frozen=True, slots=True)
class Outcome:
    record_id: str
    produced: bool
    exact: bool
    similarity: float


@dataclass(frozen=True, slots=True)
class Report:
    """Not a ClassificationReport: translation scores *text* against a reference, so there is no
    predicted label to compare -- only exact match after normalisation, and a graded similarity."""

    outcomes: tuple[Outcome, ...]

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def produced(self) -> int:
        return sum(1 for o in self.outcomes if o.produced)

    @property
    def exact_match(self) -> float:
        return sum(1 for o in self.outcomes if o.exact) / self.total if self.total else 0.0

    @property
    def mean_similarity(self) -> float:
        return sum(o.similarity for o in self.outcomes) / self.total if self.total else 0.0


def evaluate(pairs: Iterable[Pair]) -> Report:
    """Score ``(record_id, produced_or_None, gold_target)`` triples."""
    outcomes = []
    for record_id, produced, gold in pairs:
        if produced is None:
            outcomes.append(Outcome(record_id, produced=False, exact=False, similarity=0.0))
            continue
        reference = _norm(str(gold))
        outcomes.append(Outcome(
            record_id, produced=True, exact=_norm(produced) == reference,
            similarity=trigram_similarity(_norm(produced), reference)))
    return Report(tuple(outcomes))


def main(argv: Sequence[str] | None = None) -> int:
    parser = journal_gold_parser("Translation quality vs held-out gold.",
                                 gold_help="gold translations (JSONL)")
    args = parser.parse_args(argv)

    def build() -> str:
        gold = load_label_gold(args.gold, field="target")
        report = evaluate(join_journal_with_gold(args.journal, gold))
        return (f"translated {report.produced}/{report.total} | "
                f"exact match {report.exact_match:.3f} | "
                f"mean trigram similarity to gold {report.mean_similarity:.3f}")

    return run_report(build)


if __name__ == "__main__":  # pragma: no cover - exercised via main() in tests
    raise SystemExit(main())
