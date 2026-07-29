"""Outputs already produced this run, looked up by the input that produced them.

Two things need this. Duplicate inputs are produced once and shared, and a neighbouring record
that has already been produced can be shown to the model *with* its output — not only in the
input it was extracted as — so terminology and phrasing stay continuous across a scene.

The masked input text is a sound key: two records share a key only when they present the model
with the identical problem.
"""
from __future__ import annotations

import threading
from collections.abc import Iterable

from ragkit.core.records import Record


class OutputMemory:
    """Accepted outputs, keyed by the input that produced them. Safe to share across worker
    threads: the runner records from its own thread while workers read, both through the lock."""

    def __init__(self, initial: dict[str, str] | None = None) -> None:
        self._by_source: dict[str, str] = dict(initial or {})
        self._lock = threading.Lock()

    @classmethod
    def from_records(cls, records: Iterable[Record]) -> OutputMemory:
        """Build from previously journalled results. Only injectable records (verified/produced)
        with an output contribute; a rejected record's output failed its checks, so offering it as
        context would spread a known-bad rendering."""
        memory = cls()
        for record in records:
            if record.output and record.status.is_injectable:
                memory.record(record.source, record.output)
        return memory

    def get(self, source: str) -> str | None:
        with self._lock:
            return self._by_source.get(source)

    def record(self, source: str, output: str) -> None:
        with self._lock:
            self._by_source[source] = output

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_source)
