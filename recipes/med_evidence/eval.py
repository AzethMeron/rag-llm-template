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
  non-abstained decision was produced). A miss or an abstention is excluded from this denominator, so
  it measures how right the system is *when it commits*.

Run: ``PYTHONPATH=src:. python -m recipes.med_evidence.eval \\
        --journal work/med.jsonl --gold recipes/med_evidence/data/gold.jsonl``
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ragkit.core.records import read_journal

_UNSUPPORTED = "unsupported"


class EvalError(Exception):
    """The evaluation cannot run as configured (missing/malformed gold)."""


@dataclass(frozen=True, slots=True)
class Outcome:
    record_id: str
    produced: bool
    predicted: str | None
    gold: str

    @property
    def correct(self) -> bool:
        # Gold is always a concrete yes/no/maybe, so an abstention or a miss (predicted None or
        # 'unsupported') can never be correct -- exactly the intended behaviour.
        return self.predicted == self.gold

    @property
    def abstained(self) -> bool:
        return self.produced and self.predicted == _UNSUPPORTED

    @property
    def answered(self) -> bool:
        # A concrete decision was produced -- not a miss, not an abstention.
        return self.produced and self.predicted is not None and self.predicted != _UNSUPPORTED


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
    def answered(self) -> int:
        return sum(1 for o in self.outcomes if o.answered)

    @property
    def accuracy(self) -> float:
        return self._rate(lambda o: o.correct, self.total)

    @property
    def abstain_rate(self) -> float:
        return self._rate(lambda o: o.abstained, self.total)

    @property
    def answered_accuracy(self) -> float:
        # Accuracy over only the records the system committed a concrete decision on.
        return self._rate(lambda o: o.correct, self.answered)

    def _rate(self, predicate: Callable[[Outcome], bool], denominator: int) -> float:
        return sum(1 for o in self.outcomes if predicate(o)) / denominator if denominator else 0.0


def _decision(produced: str | None, field: str) -> str | None:
    if produced is None:
        return None
    try:
        answer = json.loads(produced)
    except json.JSONDecodeError:
        return None
    value = answer.get(field) if isinstance(answer, dict) else None
    return value.strip().lower() if isinstance(value, str) else None


def evaluate(pairs: Iterable[tuple[str, str | None, str]], *,
             decision_field: str = "decision") -> Report:
    """Score ``(record_id, produced_output_or_None, gold_decision)`` triples."""
    outcomes = []
    for record_id, produced, gold in pairs:
        predicted = _decision(produced, decision_field)
        outcomes.append(Outcome(record_id, produced=produced is not None,
                                predicted=predicted, gold=gold.strip().lower()))
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
        if "record_id" not in row or "decision" not in row:
            raise EvalError(f"{path}:{line_no}: a gold row needs 'record_id' and 'decision'")
        gold[str(row["record_id"])] = str(row["decision"])
    if not gold:
        raise EvalError(f"gold file is empty: {path}")
    return gold


def _pairs(journal: Path, gold: Mapping[str, str]) -> list[tuple[str, str | None, str]]:
    produced: dict[str, str | None] = {}
    for record in read_journal(journal):
        produced[record.record_id] = record.output if record.status.is_injectable else None
    return [(rid, produced.get(rid), decision) for rid, decision in gold.items()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Biomedical-evidence decision accuracy.")
    parser.add_argument("--journal", type=Path, required=True, help="run journal (JSONL)")
    parser.add_argument("--gold", type=Path, required=True, help="gold decisions (JSONL)")
    args = parser.parse_args(argv)
    try:
        report = evaluate(_pairs(args.journal, load_gold(args.gold)))
    except EvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"decided {report.produced}/{report.total} | "
          f"accuracy {report.accuracy:.3f} | "
          f"abstain rate {report.abstain_rate:.3f} | "
          f"answered accuracy (over {report.answered}) {report.answered_accuracy:.3f}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via main() in tests
    raise SystemExit(main())
