"""The capture sink: a context block or validator reports what it already retrieved/assembled,
never triggers a retrieval of its own."""
from __future__ import annotations

from ragkit.core.ports import Retrieved
from ragkit.harness.capture import Capture, capture_retrieved


class TestCapture:
    def test_starts_empty(self) -> None:
        capture = Capture()
        assert capture.passage == "" and capture.retrieved == {}

    def test_note_passage_overwrites(self) -> None:
        capture = Capture()
        capture.note_passage("first")
        capture.note_passage("second")
        assert capture.passage == "second"

    def test_note_retrieved_keys_by_chunk_id(self) -> None:
        capture = Capture()
        hit = Retrieved("c1", "text", 0.5)
        capture.note_retrieved([hit])
        assert capture.retrieved == {"c1": hit}

    def test_note_retrieved_merges_across_calls(self) -> None:
        capture = Capture()
        first = Retrieved("c1", "one", 0.5)
        second = Retrieved("c2", "two", 0.6)
        capture.note_retrieved([first])
        capture.note_retrieved([second])
        assert capture.retrieved == {"c1": first, "c2": second}

    def test_note_retrieved_a_repeated_id_is_overwritten_by_the_later_call(self) -> None:
        capture = Capture()
        capture.note_retrieved([Retrieved("c1", "stale", 0.1)])
        fresh = Retrieved("c1", "fresh", 0.9)
        capture.note_retrieved([fresh])
        assert capture.retrieved == {"c1": fresh}


class TestCaptureRetrieved:
    def test_records_hits_when_a_capture_is_wired(self) -> None:
        capture = Capture()
        hit = Retrieved("c1", "text", 0.5)
        capture_retrieved({"capture": capture}, [hit])
        assert capture.retrieved == {"c1": hit}

    def test_no_op_without_a_capture(self) -> None:
        # No exception, nothing to assert on -- just must not raise.
        capture_retrieved({}, [Retrieved("c1", "text", 0.5)])
        capture_retrieved({"capture": None}, [Retrieved("c1", "text", 0.5)])

    def test_no_op_for_a_wrong_typed_capture_value(self) -> None:
        capture_retrieved({"capture": object()}, [Retrieved("c1", "text", 0.5)])

    def test_no_op_for_empty_hits(self) -> None:
        capture = Capture()
        capture_retrieved({"capture": capture}, [])
        assert capture.retrieved == {}
