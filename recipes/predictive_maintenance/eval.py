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

import argparse
import json
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ragkit.core.records import read_journal

_ATTENTION = frozenset({"watch", "urgent"})


class EvalError(Exception):
    """The evaluation cannot run as configured (missing/malformed gold)."""


@dataclass(frozen=True, slots=True)
class Outcome:
    record_id: str
    produced: bool
    predicted: str | None
    gold: str

    @property
    def exact(self) -> bool:
        return self.predicted == self.gold

    @property
    def actionable(self) -> bool:
        # Did it get the needs-attention-vs-normal call right, even if it confused watch/urgent? A
        # record with no decision is a miss, not a lucky "normal" match.
        return (self.produced
                and (self.predicted in _ATTENTION) == (self.gold in _ATTENTION))


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
    def exact_accuracy(self) -> float:
        return self._rate(lambda o: o.exact)

    @property
    def actionable_accuracy(self) -> float:
        return self._rate(lambda o: o.actionable)

    def _rate(self, predicate: Callable[[Outcome], bool]) -> float:
        return sum(1 for o in self.outcomes if predicate(o)) / self.total if self.total else 0.0


def _severity(produced: str | None, field: str) -> str | None:
    if produced is None:
        return None
    try:
        decision = json.loads(produced)
    except json.JSONDecodeError:
        return None
    value = decision.get(field) if isinstance(decision, dict) else None
    return value.strip().lower() if isinstance(value, str) else None


def evaluate(pairs: Iterable[tuple[str, str | None, str]], *,
             severity_field: str = "severity") -> Report:
    """Score ``(record_id, produced_output_or_None, gold_severity)`` triples."""
    outcomes = []
    for record_id, produced, gold in pairs:
        predicted = _severity(produced, severity_field)
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
        if "record_id" not in row or "severity" not in row:
            raise EvalError(f"{path}:{line_no}: a gold row needs 'record_id' and 'severity'")
        gold[str(row["record_id"])] = str(row["severity"])
    if not gold:
        raise EvalError(f"gold file is empty: {path}")
    return gold


def _pairs(journal: Path, gold: Mapping[str, str]) -> list[tuple[str, str | None, str]]:
    produced: dict[str, str | None] = {}
    for record in read_journal(journal):
        produced[record.record_id] = record.output if record.status.is_injectable else None
    return [(rid, produced.get(rid), sev) for rid, sev in gold.items()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predictive-maintenance decision accuracy.")
    parser.add_argument("--journal", type=Path, required=True, help="run journal (JSONL)")
    parser.add_argument("--gold", type=Path, required=True, help="gold severities (JSONL)")
    args = parser.parse_args(argv)
    try:
        report = evaluate(_pairs(args.journal, load_gold(args.gold)))
    except EvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"decided {report.produced}/{report.total} | "
          f"exact severity {report.exact_accuracy:.3f} | "
          f"actionable (attention vs normal) {report.actionable_accuracy:.3f}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via main() in tests
    raise SystemExit(main())
