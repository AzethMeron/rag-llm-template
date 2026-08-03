"""Execution accuracy for the NL->SQL recipe: run each produced query and its gold query against
the real (read-only) database and compare result sets, the standard Spider metric.

Why execution match rather than string match: two different SQL strings can be equally correct
(``COUNT(*)`` vs ``COUNT(id)``, a different join order, an alias), and string equality would score
a correct answer wrong. Comparison follows the Spider convention:

* the produced and gold result sets are compared **positionally by row tuple** (column order and
  count must agree, as they must for the answer to be usable);
* order between rows is compared **only when the gold query has an ``ORDER BY``** — otherwise the
  rows are compared as a multiset, since an unordered query makes no promise about row order.

A record that was not produced (rejected/skipped/pending) or whose SQL fails to execute counts as a
miss, never a crash — the point of the metric is to be reported, not to abort. Execution happens
through the read-only :class:`~ragkit.core.ports.SqlStore` binding, so eval cannot mutate the data.

Run:  ``PYTHONPATH=src:. python -m recipes.nl_to_sql.eval --config recipes/nl_to_sql/config \\
        --journal work/nl_to_sql.jsonl --gold recipes/nl_to_sql/data/gold.jsonl``
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ragkit.core.ports import SqlStore
from ragkit.eval.gold import (
    EvalError,
    Pair,
    join_journal_with_gold,
    journal_gold_parser,
    load_label_gold,
    run_report,
)
from ragkit.store import load_storage

_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Outcome:
    """One evaluated question: whether the produced query executed and whether it matched gold."""

    record_id: str
    produced: bool  # a usable (injectable) output existed to evaluate
    executed: bool  # the produced query ran without error
    matched: bool  # its result set equals the gold query's
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Report:
    """Aggregate execution accuracy over every gold question."""

    outcomes: tuple[Outcome, ...]

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def matched(self) -> int:
        return sum(1 for o in self.outcomes if o.matched)

    @property
    def executed(self) -> int:
        return sum(1 for o in self.outcomes if o.executed)

    @property
    def accuracy(self) -> float:
        # Accuracy over every gold question (a missing or failing query is a miss), so an empty
        # gold set is a configuration error rather than a vacuous 1.0.
        return self.matched / self.total if self.total else 0.0


def _result_signature(rows: Sequence[Mapping[str, object]], *, ordered: bool) -> object:
    tuples = [tuple(row.values()) for row in rows]
    return tuples if ordered else Counter(tuples)


def _run(store: SqlStore, sql: str, *, ordered: bool) -> object:
    return _result_signature(store.query(sql), ordered=ordered)


def evaluate(pairs: Iterable[Pair], store: SqlStore) -> Report:
    """Score ``(record_id, produced_sql_or_None, gold_sql)`` triples against ``store``.

    ``produced_sql`` is ``None`` when the record produced no usable output (it was rejected or never
    run); that is a miss. The gold query is assumed correct, so a gold query that fails to execute
    is a configuration error (wrong database wired) and is raised rather than silently scored.
    """
    outcomes: list[Outcome] = []
    for record_id, produced, raw_gold in pairs:
        gold = str(raw_gold)
        ordered = bool(_ORDER_BY.search(gold))
        try:
            expected = _run(store, gold, ordered=ordered)
        except Exception as exc:  # a failing gold means the wrong DB is wired -- re-raised below
            raise EvalError(f"gold SQL for {record_id!r} failed to execute against the wired "
                            f"database: {exc}") from exc
        if produced is None:
            outcomes.append(Outcome(record_id, produced=False, executed=False, matched=False,
                                    detail="no usable output was produced"))
            continue
        try:
            got = _run(store, produced, ordered=ordered)
        except Exception as exc:  # noqa: BLE001 -- a produced query that will not run is a miss
            outcomes.append(Outcome(record_id, produced=True, executed=False, matched=False,
                                    detail=f"produced SQL failed to execute: {exc}"))
            continue
        matched = got == expected
        outcomes.append(Outcome(record_id, produced=True, executed=True, matched=matched,
                                detail="" if matched else "result set differs from gold"))
    return Report(tuple(outcomes))


def _external_store(config_dir: Path) -> SqlStore:
    storage = load_storage(config_dir / "storage.toml")
    if storage.sql is None:
        raise EvalError("storage.toml wires no [sql] store to evaluate against")
    if not storage.sql.read_only:
        raise EvalError("the [sql] store must be read_only for evaluation (it is the external "
                        "data source, never written)")
    return storage.sql


def main(argv: Sequence[str] | None = None) -> int:
    parser = journal_gold_parser("NL->SQL execution-accuracy evaluation.",
                                 gold_help="gold SQL (JSONL)")
    parser.add_argument("--config", type=Path, required=True, help="recipe config directory")
    args = parser.parse_args(argv)

    def build() -> str:
        store = _external_store(args.config)
        gold = load_label_gold(args.gold, field="sql")
        report = evaluate(join_journal_with_gold(args.journal, gold), store)
        return (f"execution accuracy: {report.matched}/{report.total} = {report.accuracy:.3f} "
                f"({report.executed} executed)")

    return run_report(build)


if __name__ == "__main__":  # pragma: no cover - exercised via main() in tests
    raise SystemExit(main())
