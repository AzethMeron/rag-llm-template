"""Held-out-field fill accuracy for the form-autofill recipe: compare each filled form to the gold
values that were held out of the record.

For every gold track we hold out its genre and unit price; the recipe fills them back from the
album's sibling tracks. This scores the fill:

* **genre** — matched case-insensitively after trimming (``"rock"`` == ``"Rock "``);
* **unit_price** — matched numerically within a small tolerance (prices like ``0.99`` should not
  fail on a float representation).

A record that produced no usable output (rejected/pending) counts as a miss on every field, never a
crash. Reports per-field accuracy and the stricter "both fields correct" rate.

Run: ``PYTHONPATH=src:. python -m recipes.form_autofill.eval \\
        --journal work/form.jsonl --gold recipes/form_autofill/data/gold.jsonl``
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ragkit.core.records import read_journal

_PRICE_TOLERANCE = 0.005


class EvalError(Exception):
    """The evaluation cannot run as configured (missing/malformed gold)."""


@dataclass(frozen=True, slots=True)
class Outcome:
    """One evaluated track: whether a form was produced and which fields matched gold."""

    record_id: str
    produced: bool
    genre_ok: bool
    price_ok: bool

    @property
    def both_ok(self) -> bool:
        return self.genre_ok and self.price_ok


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
    def genre_accuracy(self) -> float:
        return self._rate(lambda o: o.genre_ok)

    @property
    def price_accuracy(self) -> float:
        return self._rate(lambda o: o.price_ok)

    @property
    def both_accuracy(self) -> float:
        return self._rate(lambda o: o.both_ok)

    def _rate(self, predicate: Callable[[Outcome], bool]) -> float:
        return sum(1 for o in self.outcomes if predicate(o)) / self.total if self.total else 0.0


def _genre_matches(produced: object, gold: object) -> bool:
    return (isinstance(produced, str) and isinstance(gold, str)
            and produced.strip().lower() == gold.strip().lower())


def _price_matches(produced: object, gold: object) -> bool:
    return (isinstance(produced, (int, float)) and not isinstance(produced, bool)
            and isinstance(gold, (int, float)) and not isinstance(gold, bool)
            and abs(float(produced) - float(gold)) <= _PRICE_TOLERANCE)


def evaluate(pairs: Iterable[tuple[str, str | None, Mapping[str, object]]]) -> Report:
    """Score ``(record_id, produced_output_or_None, gold_fields)`` triples. ``produced_output`` is
    the record's output string (canonical JSON from the form schema), or ``None`` when nothing
    usable was produced."""
    outcomes: list[Outcome] = []
    for record_id, produced, gold in pairs:
        form = _parse(produced)
        genre_ok = _genre_matches(form.get("genre"), gold["genre"])
        price_ok = _price_matches(form.get("unit_price"), gold["unit_price"])
        outcomes.append(Outcome(record_id, produced=produced is not None,
                                genre_ok=genre_ok, price_ok=price_ok))
    return Report(tuple(outcomes))


def _parse(produced: str | None) -> Mapping[str, object]:
    if produced is None:
        return {}
    try:
        form = json.loads(produced)
    except json.JSONDecodeError:
        return {}
    return form if isinstance(form, dict) else {}


def load_gold(path: Path) -> dict[str, dict[str, object]]:
    if not path.is_file():
        raise EvalError(f"gold file not found: {path}")
    gold: dict[str, dict[str, object]] = {}
    for line_no, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalError(f"{path}:{line_no}: invalid JSON in gold file: {exc}") from exc
        if not {"record_id", "genre", "unit_price"} <= row.keys():
            raise EvalError(f"{path}:{line_no}: a gold row needs 'record_id', 'genre', "
                            f"'unit_price'")
        gold[str(row["record_id"])] = {"genre": row["genre"], "unit_price": row["unit_price"]}
    if not gold:
        raise EvalError(f"gold file is empty: {path}")
    return gold


_Pair = tuple[str, str | None, Mapping[str, object]]


def _pairs(journal: Path, gold: Mapping[str, dict[str, object]]) -> list[_Pair]:
    produced: dict[str, str | None] = {}
    for record in read_journal(journal):
        produced[record.record_id] = record.output if record.status.is_injectable else None
    return [(rid, produced.get(rid), fields) for rid, fields in gold.items()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Form-autofill held-out-field accuracy.")
    parser.add_argument("--journal", type=Path, required=True, help="run journal (JSONL)")
    parser.add_argument("--gold", type=Path, required=True, help="gold fields (JSONL)")
    args = parser.parse_args(argv)
    try:
        report = evaluate(_pairs(args.journal, load_gold(args.gold)))
    except EvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"filled {report.produced}/{report.total} | "
          f"genre {report.genre_accuracy:.3f} | price {report.price_accuracy:.3f} | "
          f"both {report.both_accuracy:.3f}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via main() in tests
    raise SystemExit(main())
