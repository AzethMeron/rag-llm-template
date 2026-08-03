"""Gold data and the scaffolding every recipe's ``eval.py`` needs: read the judgments, line them
up with a run journal, and report a failure the same way.

Each recipe scores something different — a decision label, a severity bucket, execution-equivalent
SQL, retrieval rank quality — but the frame around that is identical: load a JSONL gold file
(validating each row and naming ``path:line`` when it is wrong), pair each gold id with whatever
the journal produced for it (``None`` when the record was rejected or never ran), and turn an
``EvalError`` into one stderr line and exit 1. Six copies of that frame had already drifted: the
required-key check differed per recipe, and a fix to one copy's error message reached none of the
others.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

from ragkit.core.errors import RagkitError
from ragkit.core.records import read_journal

# (record_id, produced output or None, gold value) -- what every scorer consumes.
Pair = tuple[str, str | None, object]


class EvalError(RagkitError):
    """A recipe's evaluation cannot run as configured: missing or malformed gold, or a journal
    that cannot be read. A :class:`~ragkit.core.errors.RagkitError`, so a host embedding a recipe
    eval catches it with everything else rather than a bare ``Exception``."""


def gold_rows(path: Path, *, required: Sequence[str]) -> Iterator[Mapping[str, object]]:
    """Each non-blank line of a JSONL gold file, parsed and checked for ``required`` keys.

    One home for the checks each recipe was writing itself, so the "a gold row needs ..." message
    and the ``path:line`` prefix are the same everywhere.
    """
    if not path.is_file():
        raise EvalError(f"gold file not found: {path}")
    empty = True
    for line_no, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalError(f"{path}:{line_no}: invalid JSON in gold file: {exc}") from exc
        if not isinstance(row, Mapping):
            raise EvalError(f"{path}:{line_no}: a gold row must be a JSON object, got "
                            f"{type(row).__name__}")
        missing = [key for key in required if key not in row]
        if missing:
            raise EvalError(f"{path}:{line_no}: a gold row needs "
                            f"{', '.join(repr(k) for k in required)} (missing {missing})")
        empty = False
        yield row
    if empty:
        raise EvalError(f"gold file is empty: {path}")


def load_label_gold(path: Path, *, field: str) -> dict[str, str]:
    """``{record_id: label}`` from a JSONL gold file — the single-label shape (a decision, a
    severity bucket, a reference translation, a gold SQL statement)."""
    return {str(row["record_id"]): str(row[field])
            for row in gold_rows(path, required=("record_id", field))}


def load_fields_gold(path: Path, *, fields: Sequence[str]) -> dict[str, dict[str, object]]:
    """``{record_id: {field: value, ...}}`` — the several-fields-per-record shape (a form's
    held-out columns). Values keep their JSON type, since a numeric field is compared numerically.
    """
    return {str(row["record_id"]): {field: row[field] for field in fields}
            for row in gold_rows(path, required=("record_id", *fields))}


def load_relevance_gold(path: Path, *, field: str = "relevant") -> dict[str, frozenset[str]]:
    """``{record_id: {relevant_chunk_id, ...}}`` — the retrieval shape. An empty list is refused
    here rather than becoming a query that scores 0 by construction."""
    gold: dict[str, frozenset[str]] = {}
    for row in gold_rows(path, required=("record_id", field)):
        relevant = row[field]
        if not isinstance(relevant, list) or not relevant:
            raise EvalError(f"{path}: {field!r} must be a non-empty list of ids for record "
                            f"{row['record_id']!r}")
        gold[str(row["record_id"])] = frozenset(str(item) for item in relevant)
    return gold


def join_journal_with_gold(journal: Path, gold: Mapping[str, object]) -> list[Pair]:
    """One ``(record_id, produced, gold_value)`` per gold entry, in gold order.

    ``produced`` is the record's output when the journal has an *injectable* result for it, and
    ``None`` otherwise — a record that was rejected, skipped, or never ran. That distinction is
    the reason this is shared: scoring a rejection as an empty-string answer rather than a miss
    would quietly flatter every recipe's numbers.
    """
    produced: dict[str, str | None] = {
        record.record_id: (record.output if record.status.is_injectable else None)
        for record in read_journal(journal)}
    return [(record_id, produced.get(record_id), value) for record_id, value in gold.items()]


def json_field(produced: str | None, field: str) -> str | None:
    """``field`` from a produced JSON object, lower-cased and stripped — or ``None`` when nothing
    was produced, the output is not JSON, or the field is absent or not a string. Every one of
    those is a miss, never a crash: an eval must survive whatever a model returned."""
    if produced is None:
        return None
    try:
        parsed = json.loads(produced)
    except json.JSONDecodeError:
        return None
    value = parsed.get(field) if isinstance(parsed, dict) else None
    return value.strip().lower() if isinstance(value, str) else None


def journal_gold_parser(description: str, *, gold_help: str) -> argparse.ArgumentParser:
    """The ``--journal``/``--gold`` argument parser every recipe eval starts from."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--journal", type=Path, required=True, help="run journal (JSONL)")
    parser.add_argument("--gold", type=Path, required=True, help=gold_help)
    return parser


def run_report(build: Callable[[], str]) -> int:
    """Print what ``build`` returns and exit 0, or report an :class:`EvalError` on stderr and
    exit 1. The shared tail of every recipe's ``main()``."""
    try:
        line = build()
    except EvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(line)
    return 0
