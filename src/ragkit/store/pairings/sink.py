"""The reinjection seam: a :class:`~ragkit.core.ports.Sink` that writes accepted, produced records
into a :class:`~ragkit.core.ports.PairingStore` as new reference pairings — the mechanism that lets
a run's outputs become a later run's worked examples (write-back; see
:mod:`ragkit.ingest.writeback` for the RunStore-reading orchestration that is its first caller).
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable

from ragkit.core.ports import Pairing, PairingStore
from ragkit.core.records import Record


def pairing_chunk_id(source: str, target: str, context: str) -> str:
    """A deterministic id from a pairing's content, so writing back the identical
    ``(source, target, context)`` twice is a no-op (``PairingStore.add`` is idempotent on a
    duplicate ``chunk_id``) rather than a growing duplicate — the same guarantee
    :func:`~ragkit.core.records.make_record_id` gives a catalogue entry, for the same reason."""
    digest = hashlib.sha1(f"{source}\0{target}\0{context}".encode("utf-8")).hexdigest()
    return f"wb-{digest[:16]}"


class PairingSink:
    """A :class:`~ragkit.core.ports.Sink` that turns each injectable record into a
    :class:`~ragkit.core.ports.Pairing` and adds it to a :class:`~ragkit.core.ports.PairingStore`.

    ``Sink.write`` is a task-agnostic core port (``Iterable[Record]`` only), so the context a
    record was produced with — needed for the pairing's ``context`` field — travels via
    ``record.meta["context_passage"]`` rather than a wider signature; a caller that has it (write-
    back, reading a :class:`~ragkit.core.ports.RunResult`) stashes it there before calling
    :meth:`write`. A record missing it, or one that is not injectable or carries no output, is
    skipped rather than written as a hollow pairing.
    """

    CONTEXT_META_KEY = "context_passage"

    def __init__(self, pairing_store: PairingStore, *,
                clock: Callable[[], float] = time.time) -> None:
        self._pairings = pairing_store
        self._clock = clock

    def write(self, records: Iterable[Record]) -> None:
        pairings = [self._as_pairing(record) for record in records
                   if record.status.is_injectable and record.output]
        if pairings:
            self._pairings.add(pairings)

    def _as_pairing(self, record: Record) -> Pairing:
        context = str(record.meta.get(self.CONTEXT_META_KEY, ""))
        target = record.output or ""
        return Pairing(
            chunk_id=pairing_chunk_id(record.source, target, context), source=record.source,
            target=target, context=context, verified=True, created_at=self._clock())
