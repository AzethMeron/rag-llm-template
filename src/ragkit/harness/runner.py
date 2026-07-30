"""Resumable batch execution over a run store.

A full run can be tens of thousands of records, each several sequential model calls, so it spans
hours and will be interrupted. Durability is a design requirement: each result is appended to the
injected :class:`~ragkit.core.ports.RunStore` as it completes, in one transaction — restarting
reads the store's own :meth:`~ragkit.core.ports.RunStore.pending` and skips what already has a
result, and a crash mid-write leaves a complete result or none at all (the store's job, not this
module's — see :mod:`ragkit.store.run.sqlite`).

Duplicate inputs are grouped so identical text is produced once and shared, and only this thread
appends results or touches the progress counters, so the append-per-result durability needs no
locking even under concurrent workers.
"""
from __future__ import annotations

import signal
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from typing import Any

from ragkit.core.errors import RagkitError
from ragkit.core.ports import RetrievedRef, RunResult, RunStore
from ragkit.core.records import Record, Status

from .agents import Harness, Outcome, learn_memory

# Injected so a test can drive time deterministically; defaults to the monotonic wall clock.
Clock = Callable[[], float]


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


def _as_run_result(outcome: Outcome, record: Record) -> RunResult:
    """The RunStore-persisted projection of an Outcome applied to one member of its duplicate
    group: the record carrying its verdict (see ``Outcome.applied_to``), plus the structured
    reviews/violations/capture the legacy JSONL journal could not hold. ``reviews`` is flattened to
    plain dicts (via ``dataclasses.asdict``) because ``Review`` is a harness type and the run-store
    port must not depend on this layer."""
    return RunResult(
        record=outcome.applied_to(record),
        context_passage=outcome.context_passage,
        retrieved=tuple(RetrievedRef(r.chunk_id, r.text, r.score) for r in outcome.retrieved),
        reviews=tuple(asdict(review) for review in outcome.reviews),
        violations=outcome.violations,
        rounds=outcome.rounds,
        error=outcome.error)


class _Interruptible:
    """Turn SIGINT/SIGTERM into a cooperative stop flag: the first signal asks the loop to finish
    the current record and exit cleanly, rather than tearing down mid-write."""

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


def run_batch(harness: Harness, records: Iterable[Record], run_store: RunStore, *,
              total: int | None = None, already_done: int = 0,
              on_progress: Callable[[Progress], None] | None = None, report_every: int = 25,
              concurrency: int = 1, clock: Clock = lambda: 0.0,
              install_signal_handlers: bool = True) -> Progress:
    """Produce for ``records``, appending each result to ``run_store`` as it completes.

    ``concurrency`` records are produced at once against the same pool; the personas within one
    record still run in sequence. Only this thread appends results or touches the counters. Raises
    whatever a worker raised, once results already in flight have been appended, so an
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
    queued = iter(group_duplicates(records))
    failure: BaseException | None = None

    interrupt = _Interruptible() if install_signal_handlers else _NullInterrupt()
    with interrupt, ThreadPoolExecutor(max_workers=concurrency) as pool:

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
                    run_store.append_result(_as_run_result(outcome, member))
                    progress.record(outcome)
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
