"""Run-local output memory."""
from __future__ import annotations

from ragkit.core.records import Record, Status
from ragkit.harness.memory import OutputMemory


def test_get_and_record() -> None:
    memory = OutputMemory()
    memory.record("src", "out")
    assert memory.get("src") == "out"
    assert memory.get("absent") is None
    assert len(memory) == 1


def test_from_records_keeps_only_injectable() -> None:
    records = [
        Record(record_id="1", source="a", output="A", status=Status.VERIFIED),
        Record(record_id="2", source="b", output="B", status=Status.PRODUCED),
        Record(record_id="3", source="c", output="C", status=Status.REJECTED),
        Record(record_id="4", source="d", status=Status.PENDING),
    ]
    memory = OutputMemory.from_records(records)
    assert memory.get("a") == "A" and memory.get("b") == "B"
    assert memory.get("c") is None  # rejected excluded
    assert len(memory) == 2


def test_initial_seed() -> None:
    assert OutputMemory({"k": "v"}).get("k") == "v"
