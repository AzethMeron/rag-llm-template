"""Resumable batch execution over a catalogue of records.

A full run can be tens of thousands of records, each several sequential model calls, so it spans
hours and will be interrupted. Durability is a design requirement: results are appended to a JSONL
journal as each record completes, restarting reads the journal and skips what is recorded, and the
journal is separate from the catalogue so a crashed run can never truncate the catalogue itself.

Work order is injectable — the default is catalogue order, but a caller can translate the most
valuable records first — because "most valuable first" depends on the task, and baking one task's
answer in is what makes a runner task-specific. Duplicate inputs are grouped so identical text is
produced once and shared, and only this thread touches the journal or the progress counters, so
the append-per-result durability needs no locking even under concurrent workers.
"""
from __future__ import annotations

import os
import signal
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragkit.core.errors import RagkitError
from ragkit.core.records import Record, Status, read_catalog, read_journal

from .agents import Harness, Outcome, learn_memory

PriorityKey = Callable[[Record], Any]

# Injected so a test can drive time deterministically; defaults to the monotonic wall clock.
Clock = Callable[[], float]


def catalog_order(record: Record) -> tuple[str, int]:
    """Stable default sort key: the record's position in its source."""
    return (record.rel_path, record.line_no)


class RunnerError(RagkitError):
    """The run cannot start or continue."""


@dataclass
class Progress:
    """Live counters for a job that may span many runs. ``done`` counts what *this* run produced
    (the rate is measured from it); ``already_done`` is what earlier runs finished, so the reported
    position describes the whole job rather than this session."""

    total: int
    already_done: int = 0
    done: int = 0
    verified: int = 0
    produced: int = 0
    rejected: int = 0
    skipped: int = 0
    started_at: float = 0.0
    _clock: Clock = field(default=lambda: 0.0, repr=False, compare=False)

    def record(self, outcome: Outcome) -> None:
        # Counted last so a status with no bucket leaves every counter untouched rather than a
        # ``done`` that outruns its categories.
        match outcome.status:
            case Status.VERIFIED:
                self.verified += 1
            case Status.PRODUCED:
                self.produced += 1
            case Status.REJECTED:
                self.rejected += 1
            case Status.SKIPPED:
                self.skipped += 1
            case unhandled:
                raise RunnerError(
                    f"Progress.record: outcome for record {outcome.record.record_id!r} carries "
                    f"non-terminal status {unhandled!r}, which has no counter")
        self.done += 1

    @property
    def elapsed(self) -> float:
        return self._clock() - self.started_at

    @property
    def completed(self) -> int:
        return self.already_done + self.done

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.completed)

    def summary(self) -> str:
        return (f"{self.completed:,}/{self.total:,} done, {self.remaining:,} left "
                f"(verified {self.verified:,} | needs review {self.produced:,} | "
                f"rejected {self.rejected:,})")


def completed_ids(journal: Path) -> set[str]:
    """Record ids already recorded in the journal."""
    return {record.record_id for record in read_journal(journal)}


def pending_records(catalog: Path, journal: Path, *,
                    key: PriorityKey = catalog_order) -> list[Record]:
    """Records still needing work, in ``key`` order, excluding journalled results."""
    done = completed_ids(journal)
    records = [r for r in read_catalog(catalog)
               if r.status is Status.PENDING and r.record_id not in done]
    records.sort(key=key)
    return records


def group_duplicates(records: Iterable[Record]) -> list[tuple[Record, tuple[Record, ...]]]:
    """Group records that should receive one output, keeping input order. Keyed on
    ``(source, discriminator)`` where the discriminator is ``meta['speaker']`` if present, so two
    speakers of the same line can still diverge; members share the first member's output, and each
    is journalled individually so resume and reinjection stay per-record."""
    groups: dict[tuple[str, Any], list[Record]] = {}
    for record in records:
        key = (record.source, record.meta.get("speaker"))
        groups.setdefault(key, []).append(record)
    return [(members[0], tuple(members)) for members in groups.values()]


class _Interruptible:
    """Turn SIGINT/SIGTERM into a cooperative stop flag: the first signal asks the loop to finish
    the current record and exit cleanly, rather than tearing down an open journal mid-write."""

    def __init__(self) -> None:
        self.stop = False
        self._previous: dict[int, Any] = {}

    def __enter__(self) -> _Interruptible:
        for number in (signal.SIGINT, signal.SIGTERM):
            self._previous[number] = signal.getsignal(number)
            signal.signal(number, self._handle)
        return self

    def _handle(self, *_: object) -> None:
        self.stop = True

    def __exit__(self, *_: object) -> None:
        for number, handler in self._previous.items():
            signal.signal(number, handler)  # type: ignore[arg-type]


def run_batch(harness: Harness, records: Iterable[Record], journal: Path, *,
              total: int | None = None, already_done: int = 0,
              on_progress: Callable[[Progress], None] | None = None, report_every: int = 25,
              concurrency: int = 1, clock: Clock = lambda: 0.0,
              install_signal_handlers: bool = True) -> Progress:
    """Produce for ``records``, appending each result to ``journal`` as it completes.

    ``concurrency`` records are produced at once against the same pool; the personas within one
    record still run in sequence. Only this thread touches the journal or the counters. Raises
    whatever a worker raised, once results already in flight have been journalled, so an
    infrastructure failure never discards completed work. ``clock`` is injectable for
    deterministic tests; ``install_signal_handlers`` is off in a worker thread where signals
    cannot be caught.
    """
    if concurrency < 1:
        raise RunnerError(f"concurrency must be >= 1, got {concurrency}")
    if report_every < 1:
        raise RunnerError(f"report_every must be >= 1, got {report_every}")

    records = list(records)
    progress = Progress(total=total if total is not None else len(records) + already_done,
                        already_done=already_done, started_at=clock(), _clock=clock)
    journal.parent.mkdir(parents=True, exist_ok=True)
    queued = iter(group_duplicates(records))
    failure: BaseException | None = None

    interrupt = _Interruptible() if install_signal_handlers else _NullInterrupt()
    with interrupt, journal.open("a", encoding="utf-8") as handle, \
            ThreadPoolExecutor(max_workers=concurrency) as pool:

        def submit_next() -> bool:
            group = next(queued, None)
            if group is None:
                return False
            representative, members = group
            in_flight.add(pool.submit(lambda: (harness.process(representative), members)))
            return True

        in_flight: set[Future[tuple[Outcome, tuple[Record, ...]]]] = set()
        for _ in range(concurrency):
            if not submit_next():
                break

        while in_flight:
            finished, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in finished:
                try:
                    outcome, members = future.result()
                except BaseException as exc:  # noqa: BLE001  -- re-raised below
                    failure = failure or exc
                    continue
                for member in members:
                    handle.write(outcome.applied_to(member).to_json())
                    handle.write("\n")
                    progress.record(outcome)
                handle.flush()
                os.fsync(handle.fileno())
                learn_memory(harness.memory, outcome)
                if on_progress and progress.done % report_every == 0:
                    on_progress(progress)

            if failure is None and not interrupt.stop:
                for _ in range(len(finished)):
                    if not submit_next():
                        break

    if failure is not None:
        raise failure
    if on_progress:
        on_progress(progress)
    return progress


class _NullInterrupt:
    """A no-op stand-in for :class:`_Interruptible` when signal handlers must not be installed
    (a non-main thread cannot register them)."""

    stop = False

    def __enter__(self) -> _NullInterrupt:
        return self

    def __exit__(self, *_: object) -> None:
        return None
