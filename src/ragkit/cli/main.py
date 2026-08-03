"""Command-line entry point: four subcommands over a run store — ``import`` loads a JSONL
catalogue, ``run`` assembles a config and executes pending records, ``export`` writes the store's
results back out as a ``journal.jsonl``-compatible file, ``writeback`` folds a finished run's
verified outputs into the reference memory as new pairings.

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
from ragkit.core.ports import RunStore, VectorIndex
from ragkit.core.records import export_jsonl, import_jsonl
from ragkit.harness import Harness, OutputMemory, run_batch
from ragkit.ingest.writeback import write_back
from ragkit.retrieve.embedding import EmbeddingClient
from ragkit.store import PAIRING_STORES, VECTOR_INDEXES
from ragkit.store.pairings.sink import PairingSink
from ragkit.store.run.sqlite import SqliteRunStore

from .app import CliError, assemble

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
            raise CliError(f"--set expects key=value, got {pair!r}")
        if not key.strip():
            # `--set =x` used to produce a substitution under the empty key, which matches no
            # {placeholder} and so silently did nothing.
            raise CliError(f"--set needs a non-empty key before '=', got {pair!r}")
        result[key] = value
    return result


def _positive(value: str) -> int:
    """An ``argparse`` type for a count that must be at least 1.

    ``--limit -1`` used to reach ``records[:-1]``, quietly dropping the *last* record and running
    everything else -- a silent wrong answer rather than an error. ``--concurrency 0`` reached
    ``run_batch``'s own check, but only after the config had been assembled and a server
    contacted. Both are caught here, by argparse, before any work starts.
    """
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {number}")
    return number


def _seed_memory(harness: Harness, store: RunStore) -> None:
    """Seed the harness's output memory from already-completed results, so a resumed run's
    "already-produced neighbouring lines" context (:class:`~ragkit.harness.context.blocks.
    EstablishedBlock`) is reproducible rather than starting empty on every restart. A no-op when
    the recipe has no memory wired, or the store has no results yet (a fresh run)."""
    if harness.memory is not None:
        harness.memory = OutputMemory.from_records(store.latest_records())


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


def _writeback_vector(
        args: argparse.Namespace) -> tuple[VectorIndex | None, EmbeddingClient | None]:
    """Build the optional vector index + embedder write-back reconciles against, from the raw
    endpoint flags (writeback is a standalone post-run step -- it does not load a recipe's
    models.toml, so the embedding endpoint is named directly rather than resolved by model name)."""
    if args.vector_path is None:
        return None, None
    if not args.embedding_url:
        raise CliError("writeback: --vector-path needs --embedding-url too")
    options: dict[str, object] = {"path": str(args.vector_path)}
    if args.vector_dim is not None:
        options["dim"] = args.vector_dim
    vector = VECTOR_INDEXES.create(args.vector_driver, options)
    embedder = EmbeddingClient(base_url=args.embedding_url, model=args.embedding_model)
    return vector, embedder


def cmd_writeback(args: argparse.Namespace) -> int:
    run_store = SqliteRunStore(str(args.run_db))
    pairing_store = PAIRING_STORES.create(args.pairings_driver, {"path": str(args.pairings_db)})
    vector, embedder = _writeback_vector(args)
    sink = PairingSink(pairing_store)
    added = write_back(run_store, sink, pairing_store, vector=vector, embedder=embedder)
    print(f"{added:,} pairing(s) written back to {args.pairings_db}")
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
    run_parser.add_argument("--concurrency", type=_positive, default=2,
                            help="records produced at once against the same pool")
    run_parser.add_argument("--limit", type=_positive,
                        help="stop after N records (for trials)")
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

    writeback_parser = subparsers.add_parser(
        "writeback", parents=[logging_args],
        help="fold a finished run's verified outputs into the reference memory as new pairings")
    writeback_parser.add_argument("--run-db", type=Path, default=Path("work/run.db"))
    writeback_parser.add_argument("--pairings-db", type=Path, default=Path("work/pairings.db"))
    writeback_parser.add_argument("--pairings-driver", default="sqlite")
    writeback_parser.add_argument(
        "--vector-path", type=Path,
        help="reconcile this vector index after write-back (needs --embedding-url too)")
    writeback_parser.add_argument("--vector-driver", default="lancedb")
    writeback_parser.add_argument("--vector-dim", type=int,
                                  help="the vector index's dimension (with --vector-path)")
    writeback_parser.add_argument("--embedding-url", help="the embedding endpoint's base URL")
    writeback_parser.add_argument("--embedding-model", default="local")
    writeback_parser.set_defaults(handler=cmd_writeback)

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
