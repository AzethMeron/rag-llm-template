"""The intermediate work item — the :class:`Record` — and its durable catalogue and journal.

A record is the framework's single source of truth for one task instance: a line to
translate, a natural-language question to turn into SQL, a form to fill. It is deliberately
*provenance-carrying* — it stores the exact span it came from — so the produced output can be
put back where it belongs rather than re-derived. What a span measures is the source's
business, not this module's; only ordering matters, which is why the invariant is
``span_end >= span_start`` and nothing more.

Everything task-specific (placeholders, surrounding context, a form schema, a column budget)
lives in the free-form :attr:`Record.meta` mapping, so this type stays task-agnostic while a
recipe carries whatever it needs. ``meta`` is frozen at construction and must be
JSON-serialisable, because a catalogue round-trips through JSON Lines.

Serialised as JSON Lines: streamable, diffable, appendable, and resumable after an
interrupted run without holding the whole catalogue in memory. The durability machinery here
— atomic catalogue writes and a torn-record-tolerant journal reader — is what lets a run of
tens of thousands of records survive a crash and resume without repeating finished work.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from .errors import RagkitError

if TYPE_CHECKING:
    # Deferred: ports.py imports Record from this module, so a module-level import here would be
    # circular. Only needed for the type hint below, never evaluated at runtime.
    from .ports import RunStore


class Status(str, Enum):
    """Lifecycle of a record. The string value *is* the member, so it serialises as itself."""

    PENDING = "pending"

    PRODUCED = "produced"
    """Mechanically sound, but the review panel still objected when the budget ran out.

    Injectable like a verified record, and kept under this status so the items a human might
    want to revisit stay findable in the catalogue.
    """

    VERIFIED = "verified"
    """Passed the mechanical checks and every reviewer."""

    REJECTED = "rejected"
    """Failed the mechanical checks, or the model produced nothing usable.

    Never injected: the output may be empty or wrong, so surfacing it would put a defect in
    front of the consumer.
    """

    SKIPPED = "skipped"
    """The input needs no production (already in the desired form, or empty)."""

    @property
    def is_injectable(self) -> bool:
        """Whether a record in this state may have its output rendered back into a sink."""
        return self in (Status.VERIFIED, Status.PRODUCED)

    @property
    def is_done(self) -> bool:
        """Whether this record needs no further work (settled and acceptable, or skipped)."""
        return self in (Status.VERIFIED, Status.SKIPPED)


class CatalogError(RagkitError):
    """A catalogue or journal is malformed or internally inconsistent."""

    def __init__(self, reason: str, *, path: Path | None = None,
                 line_no: int | None = None) -> None:
        location = f"{path}:{line_no}" if path and line_no else (str(path) if path else None)
        super().__init__(reason, location=location)
        self.path = path
        self.line_no = line_no


_EMPTY_META: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class Record:
    """One task instance, with everything needed to produce its output and reinject it."""

    record_id: str
    source: str
    """The input payload the task operates on — the text to translate, the question to answer,
    the form to fill. A self-contained unit, since a fragment cannot be handled well in
    isolation."""
    output: str | None = None
    """What the harness produced, once it has. ``None`` until then, and required to be
    present for any injectable status (enforced below)."""
    status: Status = Status.PENDING
    rel_path: str = ""
    """Where this record came from, named for reinjection. Opaque to this module."""
    line_no: int = 0
    span_start: int = 0
    """Where the payload begins in its source, in whatever unit the source uses (a byte
    offset, a millisecond along a timeline, a row id). Only ordering is interpreted here."""
    span_end: int = 0
    """Where it ends. Half-open, so a payload's end is its successor's start."""
    notes: tuple[str, ...] = ()
    meta: Mapping[str, Any] = _EMPTY_META
    """Task-specific extras, frozen and JSON-serialisable. The core never interprets these;
    recipes and their components read the keys they put here (placeholders, context, a form
    schema, a column budget, a speaker label, ...)."""

    def __post_init__(self) -> None:
        if self.span_end < self.span_start:
            raise CatalogError(
                f"record {self.record_id}: span_end {self.span_end} precedes "
                f"span_start {self.span_start}")
        if self.status.is_injectable and self.output is None:
            # Both VERIFIED and PRODUCED are injectable, so both must carry the text a sink
            # would render -- otherwise a reinjector would write None into the output.
            raise CatalogError(
                f"record {self.record_id}: status {self.status.value!r} is injectable but "
                f"output is None")
        # Freeze meta defensively: copy the caller's mapping and wrap it read-only, so a
        # record's state cannot be mutated through a reference the caller kept. A non-mapping
        # is a construction error, caught here rather than at some later access.
        if not isinstance(self.meta, Mapping):
            raise CatalogError(
                f"record {self.record_id}: meta must be a mapping, got "
                f"{type(self.meta).__name__}")
        if not isinstance(self.meta, MappingProxyType):
            object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))

    def with_output(self, output: str, status: Status) -> Record:
        """A copy carrying a produced ``output`` and its ``status``; everything else unchanged."""
        return replace(self, output=output, status=status)

    def to_json(self) -> str:
        # Built field by field rather than via dataclasses.asdict, which deep-copies every
        # field and cannot copy the read-only mappingproxy that guards meta.
        payload = {
            "record_id": self.record_id,
            "source": self.source,
            "output": self.output,
            "status": self.status.value,
            "rel_path": self.rel_path,
            "line_no": self.line_no,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "notes": list(self.notes),
            "meta": dict(self.meta),
        }
        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str, *, path: Path | None = None,
                  line_no: int | None = None) -> Record:
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CatalogError(f"invalid JSON: {exc}", path=path, line_no=line_no) from exc
        if not isinstance(raw, dict):
            raise CatalogError(
                f"a record must be a JSON object, got {type(raw).__name__}",
                path=path, line_no=line_no)
        missing = {"record_id", "source"} - raw.keys()
        if missing:
            raise CatalogError(f"missing fields {sorted(missing)}", path=path, line_no=line_no)
        meta = raw.get("meta", {})
        if not isinstance(meta, dict):
            raise CatalogError(
                f"meta must be a JSON object, got {type(meta).__name__}",
                path=path, line_no=line_no)
        try:
            return cls(
                record_id=raw["record_id"],
                source=raw["source"],
                output=raw.get("output"),
                status=Status(raw.get("status", Status.PENDING.value)),
                rel_path=raw.get("rel_path", ""),
                line_no=raw.get("line_no", 0),
                span_start=raw.get("span_start", 0),
                span_end=raw.get("span_end", 0),
                notes=tuple(raw.get("notes", ())),
                meta=meta,
            )
        except ValueError as exc:
            # An unknown status value, mostly -- Status(...) raises ValueError. Reported with
            # its location rather than as a bare traceback.
            raise CatalogError(f"bad field value: {exc}", path=path, line_no=line_no) from exc


def make_record_id(rel_path: str, source: str, ordinal: int) -> str:
    """Deterministic id that survives the record moving within its source.

    Keyed on (source name, payload, occurrence index) rather than position, so re-running
    extraction with different settings keeps the ids of every payload that came out the same,
    and their outputs survive with them. The occurrence index is what keeps a payload repeated
    verbatim from collapsing into one id.
    """
    digest = hashlib.sha1(f"{rel_path}\0{source}\0{ordinal}".encode("utf-8")).hexdigest()
    return digest[:16]


def fsync_directory(path: Path) -> None:
    """Flush a directory entry, so a rename into it survives power loss.

    Renaming is atomic, but atomicity is not durability: without this the new name can be lost
    while the old one is already gone. Not every filesystem supports opening a directory, and
    where it is unsupported the rename is durable anyway, so a refusal is not an error.
    """
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def write_catalog(records: Iterable[Record], path: Path) -> int:
    """Write records as JSON Lines, atomically. Returns the number written.

    Writes a temporary sibling and renames it over the target, so a failure part-way through
    cannot leave a half-written catalogue where a good one used to be. The data is fsynced
    before the rename, because a rename that lands before its contents do would publish a
    truncated file under the real name.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(record.to_json())
                handle.write("\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        fsync_directory(path.parent)
    except BaseException:
        # The source iterable can raise mid-write; leaving the temporary behind would
        # accumulate stale ``.partial`` files that look like interrupted runs.
        temporary.unlink(missing_ok=True)
        raise
    return count


def read_catalog(path: Path) -> Iterator[Record]:
    """Stream records from a JSON Lines catalogue, validating each line."""
    if not path.is_file():
        raise CatalogError("catalogue not found", path=path)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if stripped:
                yield Record.from_json(stripped, path=path, line_no=line_no)


def read_journal(journal: Path) -> Iterator[Record]:
    """Yield every result recorded in ``journal``, in write order.

    A malformed *final* line without a trailing newline is tolerated and dropped: that is the
    expected shape of a journal left behind by a process killed mid-write, and the record it
    describes simply stays pending. A malformed line anywhere else means the journal is
    corrupt, and skipping it would discard a completed result while still reporting success, so
    it raises instead. Holds the journal in memory, which is bounded by the catalogue size.
    """
    if not journal.is_file():
        return
    with journal.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    for index, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        try:
            yield Record.from_json(line)
        except CatalogError as exc:
            if index == len(lines) - 1 and not raw.endswith("\n"):
                return
            # CatalogError already renders its own location, so take the bare reason rather
            # than the formatted message, or the path prints twice.
            raise CatalogError(f"corrupt journal record: {exc.reason}",
                               path=journal, line_no=index + 1) from exc


def merge_journal(catalog: Path, journal: Path, output: Path) -> tuple[int, int]:
    """Fold journalled results into the catalogue. Returns ``(records_written, records_updated)``.

    Later journal entries win, so re-running a record supersedes its earlier result. A
    journalled result whose id is not in the catalogue is refused rather than dropped:
    silently discarding a completed result while reporting success is exactly the failure
    :func:`read_journal` guards against, and it usually means the journal and the catalogue
    have drifted apart — which the operator needs to know.
    """
    if not journal.is_file():
        raise CatalogError("journal not found", path=journal)
    results: dict[str, Record] = {record.record_id: record for record in read_journal(journal)}

    updated = 0
    used: set[str] = set()
    merged: list[Record] = []
    for record in read_catalog(catalog):
        replacement = results.get(record.record_id)
        if replacement is not None:
            merged.append(replacement)
            used.add(record.record_id)
            updated += 1
        else:
            merged.append(record)

    orphans = results.keys() - used
    if orphans:
        sample = ", ".join(sorted(orphans)[:5])
        raise CatalogError(
            f"journal holds {len(orphans)} result(s) with no matching record in the catalogue "
            f"(e.g. {sample}); merging would silently discard them", path=journal)
    return write_catalog(merged, output), updated


def import_jsonl(store: RunStore, path: Path) -> int:
    """Import a JSON Lines catalogue into a :class:`~ragkit.core.ports.RunStore`'s record
    catalogue — the bridge from a fetch script's unchanged JSONL output to a DB-native run.
    Idempotent on a ``record_id`` already present (see ``RunStore.add_records``), so re-running an
    import is safe. Returns the number of records actually added."""
    return store.add_records(read_catalog(path))


def export_jsonl(store: RunStore, path: Path) -> int:
    """Export a :class:`~ragkit.core.ports.RunStore`'s results as a ``journal.jsonl``-compatible
    file, atomically (reuses :func:`write_catalog`'s temp-then-rename write) — so a script written
    against the legacy journal format (:func:`read_journal`, a recipe's ``eval.py``) keeps working
    unchanged over a DB-native run. Returns the number of results written."""
    return write_catalog(store.latest_records(), path)
