"""Command-line entry point: assemble a run from a config directory and execute it over a catalogue.

Reads the intermediate record format and writes results to a journal, resuming what an earlier run
finished. Knows nothing about any particular task — the output schema, validators, context blocks
and reference corpus all come from the config directory, resolved through registries.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ragkit.core.errors import RagkitError
from ragkit.harness import completed_ids, pending_records, run_batch

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


def cmd_run(args: argparse.Namespace) -> int:
    import time
    substitutions = _substitutions(args.set)
    assembled = assemble(args.config, substitutions=substitutions)
    records = pending_records(args.catalog, args.journal)
    already_done = len(completed_ids(args.journal))
    if args.limit is not None:
        records = records[:args.limit]
    if not records:
        print("nothing to do: no pending records outside the journal")
        return 0
    with assembled.pool:
        print(f"{len(records):,} records queued -> {args.journal} "
              f"({args.concurrency} at a time)")
        progress = run_batch(
            assembled.harness, records, args.journal, concurrency=args.concurrency,
            already_done=already_done, clock=time.monotonic,
            on_progress=lambda p: print(f"  {p.summary()}", flush=True))
    print(f"\n{progress.summary()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragkit", description="Run a configured RAG+LLM task")
    parser.set_defaults(handler=cmd_run)
    parser.add_argument("-C", "--config", type=Path, required=True,
                        help="the config directory (models/personas/rules/context/recipe .toml)")
    parser.add_argument("-c", "--catalog", type=Path, default=Path("work/records.jsonl"))
    parser.add_argument("-j", "--journal", type=Path, default=Path("work/journal.jsonl"))
    parser.add_argument("--concurrency", type=int, default=2,
                        help="records produced at once against the same pool")
    parser.add_argument("--limit", type=int, help="stop after N records (for trials)")
    parser.add_argument("--set", action="append",
                        help="a key=value substitution for persona instructions (repeatable), "
                             "e.g. --set source_language=English")
    parser.add_argument("--log-file", type=Path, default=Path("work/ragkit.log"))
    parser.add_argument("--no-log-file", action="store_true")
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
