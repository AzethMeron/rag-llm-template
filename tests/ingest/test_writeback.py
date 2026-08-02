"""write_back: the RunStore -> Sink orchestration -- the verified-only filter, idempotency, and
the vector reconcile it drives when a vector index/embedder are given."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from ragkit.core.ports import RunResult
from ragkit.core.records import Record, Status
from ragkit.ingest.writeback import write_back
from ragkit.retrieve.embedding import EmbeddingClient
from ragkit.store.pairings.sink import PairingSink
from ragkit.store.pairings.sqlite import SqlitePairings
from ragkit.store.run.sqlite import SqliteRunStore
from ragkit.store.vector.lancedb import LanceVectorIndex


def _embedder(vector_of: Callable[[str], list[float]]) -> EmbeddingClient:
    def handler(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        return httpx.Response(200, json={
            "data": [{"index": i, "embedding": vector_of(t)} for i, t in enumerate(inputs)]})
    return EmbeddingClient(base_url="http://x/v1",
                           client=httpx.Client(transport=httpx.MockTransport(handler)))


def _run_store_with(results: list[tuple[str, str, Status, str | None, str]]) -> SqliteRunStore:
    """Build a RunStore with one record + result per (record_id, source, status, output,
    context_passage) tuple."""
    store = SqliteRunStore()
    store.add_records([Record(record_id=rid, source=source) for rid, source, *_ in results])
    for record_id, source, status, output, context in results:
        store.append_result(RunResult(
            record=Record(record_id=record_id, source=source, status=status, output=output),
            context_passage=context))
    return store


class TestWriteBack:
    def test_writes_only_verified_results(self) -> None:
        run_store = _run_store_with([
            ("1", "q1", Status.VERIFIED, "a1", "ctx1"),
            ("2", "q2", Status.PRODUCED, "a2", "ctx2"),
            ("3", "q3", Status.REJECTED, None, ""),
        ])
        pairing_store = SqlitePairings()
        sink = PairingSink(pairing_store)
        added = write_back(run_store, sink, pairing_store)
        assert added == 1
        [chunk_id] = list(pairing_store.all_ids())
        pairing = pairing_store.get(chunk_id)
        assert pairing is not None and pairing.source == "q1" and pairing.target == "a1"

    def test_context_passage_reaches_the_pairing(self) -> None:
        run_store = _run_store_with([("1", "q1", Status.VERIFIED, "a1", "the assembled passage")])
        pairing_store = SqlitePairings()
        write_back(run_store, PairingSink(pairing_store), pairing_store)
        [chunk_id] = list(pairing_store.all_ids())
        pairing = pairing_store.get(chunk_id)
        assert pairing is not None and pairing.context == "the assembled passage"

    def test_rerunning_is_idempotent(self) -> None:
        run_store = _run_store_with([("1", "q1", Status.VERIFIED, "a1", "ctx")])
        pairing_store = SqlitePairings()
        sink = PairingSink(pairing_store)
        first = write_back(run_store, sink, pairing_store)
        second = write_back(run_store, sink, pairing_store)
        assert first == 1 and second == 0
        assert pairing_store.count() == 1

    def test_no_verified_results_writes_nothing(self) -> None:
        run_store = _run_store_with([("1", "q1", Status.REJECTED, None, "")])
        pairing_store = SqlitePairings()
        added = write_back(run_store, PairingSink(pairing_store), pairing_store)
        assert added == 0 and pairing_store.count() == 0

    def test_vector_without_embedder_is_refused(self, tmp_path: Path) -> None:
        run_store = _run_store_with([])
        pairing_store = SqlitePairings()
        vector = LanceVectorIndex(str(tmp_path / "v"), dim=2)
        with pytest.raises(ValueError, match="needs an embedder"):
            write_back(run_store, PairingSink(pairing_store), pairing_store, vector=vector)

    def test_no_vector_means_no_reconcile_attempted(self) -> None:
        # Must not raise or otherwise require a vector when none is configured.
        run_store = _run_store_with([("1", "q1", Status.VERIFIED, "a1", "")])
        pairing_store = SqlitePairings()
        added = write_back(run_store, PairingSink(pairing_store), pairing_store)
        assert added == 1

    def test_reconcile_embeds_the_new_pairing_into_the_vector_index(self, tmp_path: Path) -> None:
        run_store = _run_store_with([("1", "cat text", Status.VERIFIED, "a1", "")])
        pairing_store = SqlitePairings()
        vector = LanceVectorIndex(str(tmp_path / "v"), dim=2)
        embedder = _embedder(lambda t: [1.0, 0.0] if "cat" in t else [0.0, 1.0])
        write_back(run_store, PairingSink(pairing_store), pairing_store, vector=vector,
                  embedder=embedder)
        assert vector.count() == 1
        [chunk_id] = list(pairing_store.all_ids())
        assert vector.search([1.0, 0.0], k=1)[0][0] == chunk_id

    def test_reconcile_runs_even_when_nothing_new_was_written(self, tmp_path: Path) -> None:
        # A gap left by a previous crash must still be closed on a write-back call that itself
        # adds nothing new (mirrors ingest.reference's "reconcile on a no-op resume" contract).
        run_store = _run_store_with([("1", "cat text", Status.VERIFIED, "a1", "")])
        pairing_store = SqlitePairings()
        vector = LanceVectorIndex(str(tmp_path / "v"), dim=2)
        embedder = _embedder(lambda t: [1.0, 0.0])
        write_back(run_store, PairingSink(pairing_store), pairing_store, vector=vector,
                  embedder=embedder)
        [chunk_id] = list(pairing_store.all_ids())
        vector.delete([chunk_id])
        assert vector.count() == 0

        added = write_back(run_store, PairingSink(pairing_store), pairing_store, vector=vector,
                           embedder=embedder)
        assert added == 0  # nothing new in the pairing store
        assert vector.count() == 1  # but the gap was closed
