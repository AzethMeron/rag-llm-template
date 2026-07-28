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

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ragkit.core.records import read_journal
from ragkit.retrieve import trigram_similarity


class EvalError(Exception):
    """The evaluation cannot run as configured (missing/malformed gold)."""


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


def evaluate(pairs: Iterable[tuple[str, str | None, str]]) -> Report:
    """Score ``(record_id, produced_or_None, gold_target)`` triples."""
    outcomes = []
    for record_id, produced, gold in pairs:
        if produced is None:
            outcomes.append(Outcome(record_id, produced=False, exact=False, similarity=0.0))
            continue
        outcomes.append(Outcome(
            record_id, produced=True, exact=_norm(produced) == _norm(gold),
            similarity=trigram_similarity(_norm(produced), _norm(gold))))
    return Report(tuple(outcomes))


def load_gold(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise EvalError(f"gold file not found: {path}")
    gold: dict[str, str] = {}
    for line_no, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalError(f"{path}:{line_no}: invalid JSON in gold file: {exc}") from exc
        if "record_id" not in row or "target" not in row:
            raise EvalError(f"{path}:{line_no}: a gold row needs 'record_id' and 'target'")
        gold[str(row["record_id"])] = str(row["target"])
    if not gold:
        raise EvalError(f"gold file is empty: {path}")
    return gold


def _pairs(journal: Path, gold: Mapping[str, str]) -> list[tuple[str, str | None, str]]:
    produced: dict[str, str | None] = {}
    for record in read_journal(journal):
        produced[record.record_id] = record.output if record.status.is_injectable else None
    return [(rid, produced.get(rid), target) for rid, target in gold.items()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Translation quality vs held-out gold.")
    parser.add_argument("--journal", type=Path, required=True, help="run journal (JSONL)")
    parser.add_argument("--gold", type=Path, required=True, help="gold translations (JSONL)")
    args = parser.parse_args(argv)
    try:
        report = evaluate(_pairs(args.journal, load_gold(args.gold)))
    except EvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"translated {report.produced}/{report.total} | exact match {report.exact_match:.3f} | "
          f"mean trigram similarity to gold {report.mean_similarity:.3f}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via main() in tests
    raise SystemExit(main())
