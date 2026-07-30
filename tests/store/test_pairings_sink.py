"""PairingSink: the Sink port implementation that turns produced records into new pairings --
the filter-injectable, idempotency, and context-travels-via-meta contracts."""
from __future__ import annotations

from ragkit.core.ports import Sink
from ragkit.core.records import Record, Status
from ragkit.store.pairings.sink import PairingSink, pairing_chunk_id
from ragkit.store.pairings.sqlite import SqlitePairings


def _record(record_id: str, source: str, *, status: Status = Status.VERIFIED,
           output: str | None = "out", **meta: object) -> Record:
    return Record(record_id=record_id, source=source, status=status, output=output, meta=meta)


class TestPairingSink:
    def test_satisfies_the_sink_port(self) -> None:
        assert isinstance(PairingSink(SqlitePairings()), Sink)

    def test_writes_injectable_records_as_pairings(self) -> None:
        store = SqlitePairings()
        sink = PairingSink(store, clock=lambda: 1.0)
        sink.write([_record("1", "q1", output="a1")])
        assert store.count() == 1
        [chunk_id] = list(store.all_ids())
        pairing = store.get(chunk_id)
        assert pairing is not None
        assert pairing.source == "q1" and pairing.target == "a1" and pairing.verified is True
        assert pairing.created_at == 1.0

    def test_context_travels_via_meta(self) -> None:
        store = SqlitePairings()
        sink = PairingSink(store)
        sink.write([_record("1", "q1", output="a1", context_passage="the assembled passage")])
        [chunk_id] = list(store.all_ids())
        pairing = store.get(chunk_id)
        assert pairing is not None and pairing.context == "the assembled passage"

    def test_no_context_defaults_to_empty(self) -> None:
        store = SqlitePairings()
        PairingSink(store).write([_record("1", "q1", output="a1")])
        [chunk_id] = list(store.all_ids())
        assert store.get(chunk_id).context == ""  # type: ignore[union-attr]

    def test_skips_rejected_records(self) -> None:
        store = SqlitePairings()
        PairingSink(store).write([_record("1", "q1", status=Status.REJECTED, output=None)])
        assert store.count() == 0

    def test_skips_a_record_with_no_output(self) -> None:
        # PRODUCED/VERIFIED always carry output per Record's own invariant in real use, but the
        # sink must not blow up (or write a hollow pairing) if it somehow gets one without.
        store = SqlitePairings()
        record = Record(record_id="1", source="q1", status=Status.SKIPPED, output=None)
        PairingSink(store).write([record])
        assert store.count() == 0

    def test_writing_the_same_content_twice_is_idempotent(self) -> None:
        store = SqlitePairings()
        sink = PairingSink(store)
        sink.write([_record("1", "q1", output="a1")])
        sink.write([_record("1", "q1", output="a1")])  # same source/target/context
        assert store.count() == 1

    def test_empty_batch_is_a_noop(self) -> None:
        store = SqlitePairings()
        PairingSink(store).write([])
        assert store.count() == 0


class TestPairingChunkId:
    def test_deterministic(self) -> None:
        assert pairing_chunk_id("s", "t", "c") == pairing_chunk_id("s", "t", "c")

    def test_sensitive_to_each_component(self) -> None:
        base = pairing_chunk_id("s", "t", "c")
        assert pairing_chunk_id("s2", "t", "c") != base
        assert pairing_chunk_id("s", "t2", "c") != base
        assert pairing_chunk_id("s", "t", "c2") != base

    def test_prefixed_for_readability(self) -> None:
        assert pairing_chunk_id("s", "t", "c").startswith("wb-")
