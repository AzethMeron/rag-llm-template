"""Command-line entry point: three subcommands over a run store — ``import`` loads a JSONL
catalogue, ``run`` assembles a config and executes pending records, ``export`` writes the store's
results back out as a ``journal.jsonl``-compatible file.

Splitting execution from the JSONL catalogue this way is what makes the run durable in a real
database (see :mod:`ragkit.store.run.sqlite`) rather than a flat file: ``import``/``export`` are the
bridge to and from the format a fetch script or a recipe's ``eval.py`` still speaks, and ``run``
itself knows nothing about JSONL at all. ``run`` knows nothing about any particular task either —
the output schema, validators, context blocks and reference corpus all come from the config
directory, resolved through registries.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ragkit.core.errors import RagkitError
from ragkit.core.ports import RunStore
from ragkit.core.records import export_jsonl, import_jsonl
from ragkit.harness import Harness, OutputMemory, run_batch
from ragkit.store.run.sqlite import SqliteRunStore

from .app import assemble

logger = logging.getLogger("ragkit.cli")


class _ConsoleFilter(logging.Filter):
    """Keeps a record off the console while leaving it in the log file: a reviewer's leniency asked
    to suppress it, or the CLI already printed it."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (getattr(record, "leniency_suppress_console", False)
                    or getattr(record, "console_shown", False))


def _configure_logging(log_file: Path | None) -> None:
    log = logging.getLogger("ragkit")
    log.setLevel(logging.WARNING)
    log.propagate = False
    for handler in list(log.handlers):
        log.removeHandler(handler)
        handler.close()
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    console.addFilter(_ConsoleFilter())
    log.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        full = logging.FileHandler(log_file, encoding="utf-8", delay=True)
        full.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
        log.addHandler(full)  # unfiltered on purpose


def _substitutions(pairs: list[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        result[key] = value
    return result


def _seed_memory(harness: Harness, store: RunStore) -> None:
    """Seed the harness's output memory from already-completed results, so a resumed run's
    "already-produced neighbouring lines" context (:class:`~ragkit.harness.context.blocks.
    EstablishedBlock`) is reproducible rather than starting empty on every restart. A no-op when
    the recipe has no memory wired, or the store has no results yet (a fresh run)."""
    if harness.memory is not None:
        harness.memory = OutputMemory.from_records(result.record for result in store.results())


def cmd_import(args: argparse.Namespace) -> int:
    store = SqliteRunStore(str(args.run_db))
    added = import_jsonl(store, args.catalog)
    print(f"{added:,} record(s) added to {args.run_db} ({store.count_records():,} total)")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    import time
    substitutions = _substitutions(args.set)
    assembled = assemble(args.config, substitutions=substitutions)
    store = SqliteRunStore(str(args.run_db))
    _seed_memory(assembled.harness, store)
    records = list(store.pending())
    already_done = len(store.completed_ids())
    if args.limit is not None:
        records = records[:args.limit]
    if not records:
        print("nothing to do: no pending records in the run store")
        return 0
    with assembled.pool:
        print(f"{len(records):,} records queued -> {args.run_db} "
              f"({args.concurrency} at a time)")
        progress = run_batch(
            assembled.harness, records, store, concurrency=args.concurrency,
            already_done=already_done, clock=time.monotonic,
            on_progress=lambda p: print(f"  {p.summary()}", flush=True))
    print(f"\n{progress.summary()}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    store = SqliteRunStore(str(args.run_db))
    written = export_jsonl(store, args.journal)
    print(f"{written:,} result(s) exported to {args.journal}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    logging_args = argparse.ArgumentParser(add_help=False)
    logging_args.add_argument("--log-file", type=Path, default=Path("work/ragkit.log"))
    logging_args.add_argument("--no-log-file", action="store_true")

    parser = argparse.ArgumentParser(prog="ragkit", description="Run a configured RAG+LLM task")
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser(
        "import", parents=[logging_args], help="load a JSONL catalogue into a run store")
    import_parser.add_argument("-c", "--catalog", type=Path, default=Path("work/records.jsonl"))
    import_parser.add_argument("--run-db", type=Path, default=Path("work/run.db"))
    import_parser.set_defaults(handler=cmd_import)

    run_parser = subparsers.add_parser(
        "run", parents=[logging_args], help="execute pending records from a run store")
    run_parser.add_argument(
        "-C", "--config", type=Path, required=True,
        help="the config directory (models/personas/rules/context/recipe .toml)")
    run_parser.add_argument("--run-db", type=Path, default=Path("work/run.db"))
    run_parser.add_argument("--concurrency", type=int, default=2,
                            help="records produced at once against the same pool")
    run_parser.add_argument("--limit", type=int, help="stop after N records (for trials)")
    run_parser.add_argument("--set", action="append",
                            help="a key=value substitution for persona instructions (repeatable), "
                                 "e.g. --set source_language=English")
    run_parser.set_defaults(handler=cmd_run)

    export_parser = subparsers.add_parser(
        "export", parents=[logging_args],
        help="write a run store's results as a JSONL journal")
    export_parser.add_argument("--run-db", type=Path, default=Path("work/run.db"))
    export_parser.add_argument("-j", "--journal", type=Path, default=Path("work/journal.jsonl"))
    export_parser.set_defaults(handler=cmd_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(None if args.no_log_file else args.log_file)
    try:
        return int(args.handler(args))
    except RagkitError as exc:
        print(f"ragkit: {exc}", file=sys.stderr)
        logger.error("run aborted: %s", exc, extra={"console_shown": True})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
