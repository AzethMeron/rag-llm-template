"""Importing a reference JSONL corpus into a PairingStore, and the retrievers built over it."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from ragkit.core.ports import Pairing
from ragkit.ingest.reference import (
    PairingRetrievers,
    ReferenceImportError,
    import_reference,
    reference_pairings,
)
from ragkit.retrieve.embedding import EmbeddingClient, EmbeddingError
from ragkit.store.pairings.sqlite import SqlitePairings
from ragkit.store.vector.lancedb import LanceVectorIndex


def _embedder(vector_of: Callable[[str], list[float]]) -> EmbeddingClient:
    def handler(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        return httpx.Response(200, json={
            "data": [{"index": i, "embedding": vector_of(t)} for i, t in enumerate(inputs)]})
    return EmbeddingClient(base_url="http://x/v1",
                           client=httpx.Client(transport=httpx.MockTransport(handler)))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


class TestReferencePairings:
    def test_yields_one_pairing_per_line_numbered_ref_n(self, tmp_path: Path) -> None:
        path = tmp_path / "ref.jsonl"
        _write_jsonl(path, [{"source": "the cat sat", "target": "kot"},
                            {"source": "a dog ran", "target": "pies"}])
        pairings = list(reference_pairings(path))
        assert [p.chunk_id for p in pairings] == ["ref-1", "ref-2"]
        assert pairings[0].source == "the cat sat" and pairings[0].target == "kot"
        assert pairings[0].meta == {"source": "the cat sat", "target": "kot"}

    def test_blank_lines_and_missing_index_field_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "ref.jsonl"
        path.write_text('{"source": "the cat"}\n\n{"target": "no source"}\n', encoding="utf-8")
        pairings = list(reference_pairings(path))
        assert len(pairings) == 1 and pairings[0].source == "the cat"

    def test_no_target_field_leaves_target_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "ref.jsonl"
        _write_jsonl(path, [{"source": "just a passage, no target"}])
        [pairing] = list(reference_pairings(path))
        assert pairing.target == ""

    def test_custom_index_and_target_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "ref.jsonl"
        _write_jsonl(path, [{"q": "question", "a": "answer"}])
        [pairing] = list(reference_pairings(path, index_field="q", target_field="a"))
        assert pairing.source == "question" and pairing.target == "answer"

    def test_invalid_json_is_a_structured_error(self, tmp_path: Path) -> None:
        path = tmp_path / "ref.jsonl"
        path.write_text('{"source": "ok"}\n{not json', encoding="utf-8")
        with pytest.raises(ReferenceImportError, match="invalid JSON"):
            list(reference_pairings(path))

    def test_skip_fast_forwards_without_parsing(self, tmp_path: Path) -> None:
        path = tmp_path / "ref.jsonl"
        # A malformed early line would raise if parsed; skip must never reach it.
        path.write_text('{not json\n{"source": "kept"}\n', encoding="utf-8")
        [pairing] = list(reference_pairings(path, skip=1))
        assert pairing.chunk_id == "ref-2" and pairing.source == "kept"


class TestImportReference:
    def test_imports_all_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "ref.jsonl"
        _write_jsonl(path, [{"source": "the cat sat", "target": "kot"},
                            {"source": "a dog ran", "target": "pies"}])
        store = SqlitePairings()
        assert import_reference(path, store) == 2
        assert store.count() == 2
        assert store.document("ref-1") == (
            "the cat sat -> kot", {"source": "the cat sat", "target": "kot"})

    def test_resumes_without_reimporting(self, tmp_path: Path) -> None:
        path = tmp_path / "ref.jsonl"
        _write_jsonl(path, [{"source": "a"}, {"source": "b"}])
        store = SqlitePairings()
        assert import_reference(path, store) == 2
        assert import_reference(path, store) == 0  # nothing left past the floor
        assert store.count() == 2

    def test_resume_continues_from_a_grown_source(self, tmp_path: Path) -> None:
        path = tmp_path / "ref.jsonl"
        _write_jsonl(path, [{"source": "a"}, {"source": "b"}])
        store = SqlitePairings()
        import_reference(path, store)
        _write_jsonl(path, [{"source": "a"}, {"source": "b"}, {"source": "c"}])
        assert import_reference(path, store) == 1
        assert store.count() == 3 and store.get("ref-3") is not None

    def test_streams_in_batches(self, tmp_path: Path) -> None:
        path = tmp_path / "ref.jsonl"
        _write_jsonl(path, [{"source": f"passage {i} cat"} for i in range(4)])
        store = SqlitePairings()
        assert import_reference(path, store, batch_size=2) == 4
        assert store.count() == 4

    def test_bad_batch_size_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            import_reference(tmp_path / "ref.jsonl", SqlitePairings(), batch_size=0)

    def test_vector_without_embedder_is_refused(self, tmp_path: Path) -> None:
        vector = LanceVectorIndex(str(tmp_path / "v"), dim=2)
        with pytest.raises(ValueError, match="needs an embedder"):
            import_reference(tmp_path / "ref.jsonl", SqlitePairings(), vector=vector)

    def test_embeds_and_upserts_each_batch(self, tmp_path: Path) -> None:
        path = tmp_path / "ref.jsonl"
        _write_jsonl(path, [{"source": "cat text"}, {"source": "dog text"}])
        store = SqlitePairings()
        vector = LanceVectorIndex(str(tmp_path / "v"), dim=2)
        embedder = _embedder(lambda t: [1.0, 0.0] if "cat" in t else [0.0, 1.0])
        import_reference(path, store, vector=vector, embedder=embedder)
        assert vector.count() == 2
        hits = vector.search([1.0, 0.0], k=1)
        assert hits[0][0] == "ref-1"

    def test_a_real_embedder_failure_mid_import_is_durable_and_resumable(
            self, tmp_path: Path) -> None:
        """Unlike the test below (which fakes the crashed state directly), this drives a real
        embedding-server failure partway through a multi-batch import: the exception must
        propagate (not be swallowed), whatever pairings were durably committed before the failure
        must survive, and a plain resume (fresh embedder, no injected failure) must both finish
        the corpus and heal the vector gap the crash left -- the actual multi-hour-job
        crash/restart scenario, not just its already-healed end state."""
        path = tmp_path / "ref.jsonl"
        _write_jsonl(path, [{"source": f"text {i}"} for i in range(6)])
        store = SqlitePairings()
        vector = LanceVectorIndex(str(tmp_path / "v"), dim=2)

        call_count = 0

        def flaky_handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise httpx.ConnectError("simulated embedding-server crash")
            inputs = json.loads(request.content)["input"]
            return httpx.Response(200, json={
                "data": [{"index": i, "embedding": [1.0, 0.0]} for i, _ in enumerate(inputs)]})

        flaky_embedder = EmbeddingClient(
            base_url="http://x/v1", max_retries=0,
            client=httpx.Client(transport=httpx.MockTransport(flaky_handler)))

        with pytest.raises(EmbeddingError):
            import_reference(path, store, vector=vector, embedder=flaky_embedder, batch_size=2)

        # The pairing store commits a batch before embedding it (see _flush), so the crashed
        # batch's two pairings are durably present even though their vectors never landed.
        assert store.count() == 4
        assert vector.count() == 2

        # A plain resume (no injected failure) picks up where the pairing store left off, and its
        # unconditional final reconcile_vector() re-embeds the batch the crash orphaned.
        good_embedder = _embedder(lambda t: [1.0, 0.0])
        added = import_reference(path, store, vector=vector, embedder=good_embedder, batch_size=2)
        assert added == 2  # only ref-5/ref-6, the two lines never reached before the crash
        assert store.count() == 6
        assert vector.count() == 6

    def test_reconcile_re_embeds_a_gap_left_by_a_previous_crash(self, tmp_path: Path) -> None:
        path = tmp_path / "ref.jsonl"
        _write_jsonl(path, [{"source": "cat text"}, {"source": "dog text"}])
        store = SqlitePairings()
        vector = LanceVectorIndex(str(tmp_path / "v"), dim=2)
        embedder = _embedder(lambda t: [1.0, 0.0] if "cat" in t else [0.0, 1.0])
        import_reference(path, store, vector=vector, embedder=embedder)
        vector.delete(["ref-1"])  # simulate an upsert that never landed
        assert vector.count() == 1

        # Resume: no new pairings (floor == count), but reconcile must still run and close the gap.
        assert import_reference(path, store, vector=vector, embedder=embedder) == 0
        assert vector.count() == 2
        assert vector.search([1.0, 0.0], k=1)[0][0] == "ref-1"

    def test_reconcile_is_a_noop_with_nothing_missing(self, tmp_path: Path) -> None:
        path = tmp_path / "ref.jsonl"
        _write_jsonl(path, [{"source": "cat text"}])
        store = SqlitePairings()
        vector = LanceVectorIndex(str(tmp_path / "v"), dim=2)
        embedder = _embedder(lambda t: [1.0, 0.0])
        import_reference(path, store, vector=vector, embedder=embedder)
        import_reference(path, store, vector=vector, embedder=embedder)  # nothing to reconcile
        assert vector.count() == 1

    def test_reconcile_skips_an_id_that_vanished_before_get(self) -> None:
        # Defensive: reconcile() reports an id from all_ids() as missing, but get() no longer finds
        # it (a delete raced in between) -- must not crash and must not upsert an empty batch.
        from ragkit.ingest.reference import reconcile_vector

        class _StubPairingStore:
            def all_ids(self) -> list[str]:
                return ["ref-1"]

            def get(self, _chunk_id: str) -> None:
                return None

        class _StubVectorIndex:
            def reconcile(self, chunk_ids: object) -> set[str]:
                return set(chunk_ids)  # type: ignore[arg-type]

            def upsert(self, *_args: object, **_kwargs: object) -> None:
                raise AssertionError("must not upsert when nothing resolved")

        reconcile_vector(_StubPairingStore(), _StubVectorIndex(),  # type: ignore[arg-type]
                         embedder=None)  # type: ignore[arg-type]

    def test_reconcile_rejects_a_bad_batch_size(self, tmp_path: Path) -> None:
        from ragkit.ingest.reference import reconcile_vector

        store = SqlitePairings()
        vector = LanceVectorIndex(str(tmp_path / "v"), dim=2)
        embedder = _embedder(lambda t: [1.0, 0.0])
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            reconcile_vector(store, vector, embedder, batch_size=0)

    def test_reconcile_embeds_missing_ids_in_chunks_of_batch_size(self, tmp_path: Path) -> None:
        """Regression test: a large gap must be re-embedded in bounded batches, not as one call
        that holds every missing row's vector in RAM at once (the same shape of bug the LanceDB
        indexed_ids() fix addressed on the read side)."""
        from ragkit.ingest.reference import reconcile_vector

        path = tmp_path / "ref.jsonl"
        _write_jsonl(path, [{"source": f"text {i}"} for i in range(10)])
        store = SqlitePairings()
        vector = LanceVectorIndex(str(tmp_path / "v"), dim=2)
        embedder = _embedder(lambda t: [1.0, 0.0])
        import_reference(path, store)  # pairings only, no vector -- so all 10 are "missing"

        upsert_call_sizes: list[int] = []
        real_upsert = vector.upsert

        def _tracking_upsert(ids: object, vectors: object, metas: object) -> None:
            upsert_call_sizes.append(len(vectors))  # type: ignore[arg-type]
            real_upsert(ids, vectors, metas)  # type: ignore[arg-type]

        vector.upsert = _tracking_upsert  # type: ignore[method-assign]

        missing = reconcile_vector(store, vector, embedder, batch_size=3)

        assert missing == 10
        assert upsert_call_sizes == [3, 3, 3, 1]  # 10 ids in batches of 3, never all at once
        assert vector.count() == 10

    def test_reconcile_on_batch_reports_total_upfront_then_progress(self, tmp_path: Path) -> None:
        from ragkit.ingest.reference import reconcile_vector

        path = tmp_path / "ref.jsonl"
        _write_jsonl(path, [{"source": f"text {i}"} for i in range(5)])
        store = SqlitePairings()
        vector = LanceVectorIndex(str(tmp_path / "v"), dim=2)
        embedder = _embedder(lambda t: [1.0, 0.0])
        import_reference(path, store)

        calls: list[tuple[int, int]] = []
        reconcile_vector(store, vector, embedder, batch_size=2,
                         on_batch=lambda done, total: calls.append((done, total)))

        assert calls[0] == (0, 5)  # scope announced before any batch runs
        assert calls[-1] == (5, 5)  # final call reports completion
        assert all(done <= total for done, total in calls)

    def test_reconcile_compacts_every_n_batches(self, tmp_path: Path) -> None:
        from ragkit.ingest.reference import reconcile_vector

        path = tmp_path / "ref.jsonl"
        _write_jsonl(path, [{"source": f"text {i}"} for i in range(10)])
        store = SqlitePairings()
        vector = LanceVectorIndex(str(tmp_path / "v"), dim=2)
        embedder = _embedder(lambda t: [1.0, 0.0])
        import_reference(path, store)

        compact_calls = 0
        real_compact = vector.compact

        def _tracking_compact() -> None:
            nonlocal compact_calls
            compact_calls += 1
            real_compact()

        vector.compact = _tracking_compact  # type: ignore[method-assign]

        announced = 0
        def _on_compact() -> None:
            nonlocal announced
            announced += 1

        reconcile_vector(store, vector, embedder, batch_size=2, compact_every=3,
                         on_compact=_on_compact)

        assert compact_calls == 1  # 5 batches of 2 -> compacts once, after the 3rd batch
        assert announced == 1
        assert vector.count() == 10  # compaction never loses data

    def test_reconcile_compacts_without_an_on_compact_callback(self, tmp_path: Path) -> None:
        from ragkit.ingest.reference import reconcile_vector

        path = tmp_path / "ref.jsonl"
        _write_jsonl(path, [{"source": f"text {i}"} for i in range(4)])
        store = SqlitePairings()
        vector = LanceVectorIndex(str(tmp_path / "v"), dim=2)
        embedder = _embedder(lambda t: [1.0, 0.0])
        import_reference(path, store)

        # compact_every=1 with no on_compact -- must compact every batch without needing the
        # announce callback.
        reconcile_vector(store, vector, embedder, batch_size=2, compact_every=1)
        assert vector.count() == 4

    def test_reconcile_propagates_a_compact_failure_without_losing_prior_batches(
            self, tmp_path: Path) -> None:
        # A failed compaction must not be silently swallowed (the caller needs to know a
        # multi-hour job aborted), but everything embedded before the failing compact() call is
        # already committed (embed-then-upsert happens before the compaction check) and must
        # survive so a rerun only re-does the remaining gap.
        from ragkit.ingest.reference import reconcile_vector

        path = tmp_path / "ref.jsonl"
        _write_jsonl(path, [{"source": f"text {i}"} for i in range(6)])
        store = SqlitePairings()
        vector = LanceVectorIndex(str(tmp_path / "v"), dim=2)
        embedder = _embedder(lambda t: [1.0, 0.0])
        import_reference(path, store)

        def _boom() -> None:
            raise RuntimeError("simulated compaction failure")

        vector.compact = _boom  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="simulated compaction failure"):
            # compact_every=1 with batch_size=2 -- fails right after the first batch commits.
            reconcile_vector(store, vector, embedder, batch_size=2, compact_every=1)

        assert vector.count() == 2  # the one batch that committed before compact() blew up

        # Resume with compaction disabled -- the rest of the corpus still gets embedded.
        added = reconcile_vector(store, vector, embedder, batch_size=2, compact_every=None)
        assert added == 4
        assert vector.count() == 6

    def test_reconcile_compact_every_zero_or_none_disables_it(self, tmp_path: Path) -> None:
        from ragkit.ingest.reference import reconcile_vector

        path = tmp_path / "ref.jsonl"
        _write_jsonl(path, [{"source": f"text {i}"} for i in range(4)])
        store = SqlitePairings()
        vector = LanceVectorIndex(str(tmp_path / "v"), dim=2)
        embedder = _embedder(lambda t: [1.0, 0.0])
        import_reference(path, store)

        def _boom() -> None:
            raise AssertionError("must not compact when compact_every is None")

        vector.compact = _boom  # type: ignore[method-assign]
        reconcile_vector(store, vector, embedder, batch_size=2, compact_every=None)
        assert vector.count() == 4

    def test_reconcile_rejects_a_bad_compact_every(self, tmp_path: Path) -> None:
        from ragkit.ingest.reference import reconcile_vector

        store = SqlitePairings()
        vector = LanceVectorIndex(str(tmp_path / "v"), dim=2)
        embedder = _embedder(lambda t: [1.0, 0.0])
        with pytest.raises(ValueError, match="compact_every must be >= 1"):
            reconcile_vector(store, vector, embedder, compact_every=0)

    def test_reconcile_skips_compact_on_a_vector_index_without_one(self, tmp_path: Path) -> None:
        # Qdrant (and any driver without compact()) must not blow up just because compact_every
        # was passed -- getattr(vector, "compact", None) is None and nothing is called.
        from ragkit.ingest.reference import reconcile_vector

        path = tmp_path / "ref.jsonl"
        _write_jsonl(path, [{"source": f"text {i}"} for i in range(4)])
        store = SqlitePairings()
        import_reference(path, store)
        embedder = _embedder(lambda t: [1.0, 0.0])

        class _StubVectorIndexNoCompact:
            def __init__(self) -> None:
                self._ids: dict[str, list[float]] = {}

            def reconcile(self, chunk_ids: object) -> set[str]:
                return set(chunk_ids) - self._ids.keys()  # type: ignore[arg-type]

            def upsert(self, ids: object, vectors: object, _metas: object) -> None:
                self._ids.update(zip(ids, vectors, strict=True))  # type: ignore[arg-type]

        vector = _StubVectorIndexNoCompact()
        reconcile_vector(store, vector, embedder, batch_size=2,  # type: ignore[arg-type]
                         compact_every=1)
        assert len(vector._ids) == 4


class TestPairingRetrievers:
    def test_lexical_retriever(self) -> None:
        store = SqlitePairings()
        store.add([Pairing(chunk_id="c1", source="the cat sat", target="kot")])
        hits = PairingRetrievers(store).lexical_retriever().retrieve("cat", k=1)
        assert hits[0].text == "the cat sat -> kot"

    def test_construction_needs_an_embedder_with_a_vector(self, tmp_path: Path) -> None:
        vector = LanceVectorIndex(str(tmp_path / "v"), dim=2)
        with pytest.raises(ValueError, match="needs an embedder"):
            PairingRetrievers(SqlitePairings(), vector=vector)

    def test_dense_retriever_without_wiring_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no vector index"):
            PairingRetrievers(SqlitePairings()).dense_retriever()

    def test_dense_and_hybrid(self, tmp_path: Path) -> None:
        path = tmp_path / "ref.jsonl"
        _write_jsonl(path, [{"source": "the cat sat"}, {"source": "a dog ran"}])
        store = SqlitePairings()
        vector = LanceVectorIndex(str(tmp_path / "v"), dim=2)
        embedder = _embedder(lambda t: [1.0, 0.0] if "cat" in t else [0.0, 1.0])
        import_reference(path, store, vector=vector, embedder=embedder)

        retrievers = PairingRetrievers(store, vector=vector, embedder=embedder)
        dense_hits = retrievers.dense_retriever().retrieve("cat query", k=1)
        assert dense_hits[0].chunk_id == "ref-1"

        from ragkit.retrieve.hybrid import HybridRetriever
        hybrid = retrievers.hybrid_retriever()
        assert isinstance(hybrid, HybridRetriever)
        assert hybrid.retrieve("cat", k=2, min_score=0.0)
