"""The run-state store over SQLite — the record catalogue and the append-only result history that
replace the JSONL catalogue+journal pair (see ``docs/storage-overhaul-plan.md``).

WAL journal mode lets a long-lived :meth:`~SqliteRunStore.pending` read run concurrently with the
single writer thread's :meth:`~SqliteRunStore.append_result` calls, without either blocking the
other. ``synchronous`` controls the fsync discipline for a commit: ``FULL`` (default) fsyncs before
returning — the same durability guarantee the JSONL journal's per-line fsync gave; ``NORMAL`` is
faster and still crash-safe under WAL (a commit can be *lost* on power failure, never *corrupted*)
— a documented throughput option, not the default. Foreign keys are enforced (``PRAGMA
foreign_keys=ON``), so :meth:`append_result` for a record never added is refused **at the
database**, rather than discovered later the way ``merge_journal``'s orphan check used to.
"""
from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

from ragkit.core.errors import RagkitError
from ragkit.core.ports import RetrievedRef, RunResult
from ragkit.core.records import Record, Status
from ragkit.core.rules import Severity, Violation

_SYNCHRONOUS_MODES = frozenset({"FULL", "NORMAL"})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records(
    record_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    status TEXT NOT NULL,
    rel_path TEXT NOT NULL DEFAULT '',
    line_no INTEGER NOT NULL DEFAULT 0,
    span_start INTEGER NOT NULL DEFAULT 0,
    span_end INTEGER NOT NULL DEFAULT 0,
    meta TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS records_order ON records(rel_path, line_no);
CREATE TABLE IF NOT EXISTS results(
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id TEXT NOT NULL REFERENCES records(record_id),
    status TEXT NOT NULL,
    output TEXT,
    notes TEXT NOT NULL DEFAULT '[]',
    violations TEXT NOT NULL DEFAULT '[]',
    reviews TEXT NOT NULL DEFAULT '[]',
    rounds INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    context_passage TEXT NOT NULL DEFAULT '',
    retrieved TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS results_record ON results(record_id);
"""

_PENDING_QUERY = (
    "SELECT record_id, source, status, rel_path, line_no, span_start, span_end, meta FROM records "
    "WHERE status = ? AND NOT EXISTS "
    "(SELECT 1 FROM results WHERE results.record_id = records.record_id) "
    "ORDER BY rel_path, line_no")

_LATEST_RESULTS_QUERY = (
    "SELECT r.record_id, r.source, r.rel_path, r.line_no, r.span_start, r.span_end, r.meta, "
    "res.status, res.output, res.notes, res.violations, res.reviews, res.rounds, res.error, "
    "res.context_passage, res.retrieved "
    "FROM results res "
    "JOIN records r ON r.record_id = res.record_id "
    "JOIN (SELECT record_id, max(seq) AS seq FROM results GROUP BY record_id) latest "
    "ON latest.record_id = res.record_id AND latest.seq = res.seq "
    "ORDER BY res.seq")


class RunStoreError(RagkitError):
    """A run-store operation failed, the run database is misconfigured, or ``append_result`` was
    called for a record that was never added to the catalogue."""


def _row_to_record(row: tuple[Any, ...]) -> Record:
    record_id, source, status, rel_path, line_no, span_start, span_end, meta = row
    return Record(record_id=record_id, source=source, status=Status(status), rel_path=rel_path,
                 line_no=line_no, span_start=span_start, span_end=span_end,
                 meta=json.loads(meta))


def _row_to_result(row: tuple[Any, ...]) -> RunResult:
    (record_id, source, rel_path, line_no, span_start, span_end, meta, status, output, notes,
     violations, reviews, rounds, error, context_passage, retrieved) = row
    record = Record(record_id=record_id, source=source, status=Status(status), output=output,
                    rel_path=rel_path, line_no=line_no, span_start=span_start, span_end=span_end,
                    notes=tuple(json.loads(notes)), meta=json.loads(meta))
    return RunResult(
        record=record, context_passage=context_passage,
        retrieved=tuple(RetrievedRef(**ref) for ref in json.loads(retrieved)),
        reviews=tuple(json.loads(reviews)),
        violations=tuple(Violation(rule_id=v["rule_id"], severity=Severity(v["severity"]),
                                   message=v["message"]) for v in json.loads(violations)),
        rounds=rounds, error=error)


class SqliteRunStore:
    """A :class:`~ragkit.core.ports.RunStore` over one SQLite database (WAL, FK-enforced)."""

    CONFIG_KEYS = frozenset({"path", "synchronous"})

    def __init__(self, path: str = ":memory:", *, synchronous: str = "FULL",
                clock: Callable[[], float] = time.time) -> None:
        mode = synchronous.upper()
        if mode not in _SYNCHRONOUS_MODES:
            raise RunStoreError(
                f"synchronous must be one of {sorted(_SYNCHRONOUS_MODES)}, got {synchronous!r}")
        self._clock = clock
        self._lock = threading.Lock()
        try:
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            # A concurrent reader retries instead of failing outright on the brief window WAL
            # doesn't cover (e.g. its own checkpoint) -- see SqlitePairings.__init__ for the fuller
            # rationale (this store already had WAL; busy_timeout closes the remaining gap).
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute(f"PRAGMA synchronous={mode}")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        except sqlite3.Error as exc:
            raise RunStoreError(f"could not open the run store at {path!r}: {exc}") from exc

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> SqliteRunStore:
        return cls(path=str(options.get("path", ":memory:")),
                   synchronous=str(options.get("synchronous", "FULL")))

    def add_records(self, records: Iterable[Record]) -> int:
        rows = [(r.record_id, r.source, r.status.value, r.rel_path, r.line_no, r.span_start,
                 r.span_end, json.dumps(dict(r.meta), ensure_ascii=False)) for r in records]
        if not rows:
            return 0
        with self._lock:
            try:
                before = self._count_records_locked()
                self._conn.executemany(
                    "INSERT OR IGNORE INTO records"
                    "(record_id, source, status, rel_path, line_no, span_start, span_end, meta) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
                self._conn.commit()
                return self._count_records_locked() - before
            except sqlite3.Error as exc:
                self._safe_rollback()
                raise RunStoreError(f"could not add records: {exc}") from exc

    def append_result(self, result: RunResult) -> None:
        record = result.record
        notes = json.dumps(list(record.notes), ensure_ascii=False)
        violations = json.dumps(
            [{"rule_id": v.rule_id, "severity": v.severity.value, "message": v.message}
             for v in result.violations], ensure_ascii=False)
        reviews = json.dumps([dict(r) for r in result.reviews], ensure_ascii=False)
        retrieved = json.dumps(
            [{"chunk_id": r.chunk_id, "text": r.text, "score": r.score}
             for r in result.retrieved], ensure_ascii=False)
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO results(record_id, status, output, notes, violations, reviews, "
                    "rounds, error, context_passage, retrieved, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (record.record_id, record.status.value, record.output, notes, violations,
                     reviews, result.rounds, result.error, result.context_passage, retrieved,
                     self._clock()))
                self._conn.commit()
            except sqlite3.IntegrityError as exc:
                self._safe_rollback()
                raise RunStoreError(
                    f"append_result: record {record.record_id!r} was never added to the "
                    f"catalogue (call add_records first): {exc}") from exc
            except sqlite3.Error as exc:
                self._safe_rollback()
                raise RunStoreError(f"could not append a result: {exc}") from exc

    def _safe_rollback(self) -> None:
        """Roll back, unless the connection itself is unusable (already closed) -- in which case
        there is nothing to roll back, and letting that failure replace the real one would mask
        the actual cause behind a confusing 'closed database' error."""
        with contextlib.suppress(sqlite3.Error):
            self._conn.rollback()

    def completed_ids(self) -> set[str]:
        with self._lock:
            rows = self._conn.execute("SELECT DISTINCT record_id FROM results").fetchall()
        return {row[0] for row in rows}

    def pending(self) -> Iterator[Record]:
        with self._lock:
            rows = self._conn.execute(_PENDING_QUERY, (Status.PENDING.value,)).fetchall()
        return (_row_to_record(row) for row in rows)

    def results(self) -> Iterator[RunResult]:
        with self._lock:
            rows = self._conn.execute(_LATEST_RESULTS_QUERY).fetchall()
        return (_row_to_result(row) for row in rows)

    def _count_records_locked(self) -> int:
        return int(self._conn.execute("SELECT count(*) FROM records").fetchone()[0])

    def count_records(self) -> int:
        with self._lock:
            return self._count_records_locked()

    def close(self) -> None:
        self._conn.close()
