"""The resumable batch runner: appending to a RunStore, resume, duplicate grouping, progress,
failure."""
from __future__ import annotations

import pytest

from ragkit.core.ports import Retrieved, RunResult
from ragkit.core.records import Record, Status
from ragkit.harness import (
    Harness,
    JsonFieldSchema,
    Outcome,
    Panel,
    Persona,
    Progress,
    RuleSet,
    RunnerError,
    ValidatorPipeline,
    group_duplicates,
    run_batch,
)
from ragkit.harness.context import ContextAssembler, RetrievedBlock
from ragkit.harness.context.assembler import _PlacedBlock
from ragkit.llm.errors import LlmError
from ragkit.store.run.sqlite import SqliteRunStore

from .conftest import ACCEPT, build_harness, build_pool, ok


def _records(n: int) -> list[Record]:
    return [Record(record_id=str(i), source=f"line {i}", line_no=i) for i in range(n)]


def _store(records: list[Record]) -> SqliteRunStore:
    store = SqliteRunStore()
    store.add_records(records)
    return store


class _FailOn:
    """A pluggable validator that raises for one record's source and passes every other — the
    "one bad record in a big batch" shape."""

    def __init__(self, source: str, exc: BaseException) -> None:
        self._source = source
        self._exc = exc

    def validate(self, record: Record, _output: str, _context: dict) -> list:
        if record.source == self._source:
            raise self._exc
        return []


class TestRunBatch:
    def test_appends_each_result(self) -> None:
        harness = build_harness(produce=[ok({"output": "R"})] * 3, review=[ACCEPT] * 3)
        store = _store(_records(3))
        progress = run_batch(harness, store.pending(), store, install_signal_handlers=False)
        assert progress.verified == 3 and progress.done == 3
        assert store.completed_ids() == {"0", "1", "2"}

    def test_shares_output_across_duplicates(self) -> None:
        # Two records with identical source: one production, both get a result.
        records = [Record(record_id="a", source="same"), Record(record_id="b", source="same")]
        harness = build_harness(produce=[ok({"output": "R"})], review=[ACCEPT])
        store = _store(records)
        run_batch(harness, store.pending(), store, install_signal_handlers=False)
        results = {r.record.record_id: r for r in store.results()}
        assert results["a"].record.output == "R" and results["b"].record.output == "R"

    def test_worker_failure_propagates_after_appending(self) -> None:
        # The second record's server dies (a bare LlmError); the run stops, but the first record's
        # completed result is appended first and stays completed for the resume.
        harness = build_harness(produce=[ok({"output": "R"})] * 2, review=[ACCEPT],
                                extra_validators=[_FailOn("line 1", LlmError("server is down"))])
        store = _store(_records(2))
        with pytest.raises(LlmError, match="server is down"):
            run_batch(harness, store.pending(), store, concurrency=1,
                      install_signal_handlers=False)
        assert store.completed_ids() == {"0"}  # record 1 stays PENDING for the resume

    def test_a_poison_record_does_not_block_the_batch(self) -> None:
        # Regression: a plugin defect on one record used to leave it PENDING and abort the run, so
        # every resume re-hit the same deterministic exception -- a batch that could never finish.
        harness = build_harness(produce=[ok({"output": "R"})] * 3, review=[ACCEPT] * 3,
                                extra_validators=[_FailOn("line 1", ValueError("plugin bug"))])
        store = _store(_records(3))
        progress = run_batch(harness, store.pending(), store, install_signal_handlers=False)
        assert progress.done == 3 and progress.verified == 2 and progress.rejected == 1
        assert store.completed_ids() == {"0", "1", "2"}
        poisoned = next(r for r in store.results() if r.record.record_id == "1")
        assert poisoned.record.status is Status.REJECTED
        assert "unexpected ValueError in validate()" in (poisoned.error or "")
        assert "plugin bug" in (poisoned.error or "")

    def test_concurrency_completes_all(self) -> None:
        harness = build_harness(produce=[ok({"output": "R"})] * 5, review=[ACCEPT] * 5)
        store = _store(_records(5))
        progress = run_batch(harness, store.pending(), store, concurrency=4,
                             install_signal_handlers=False)
        assert progress.done == 5

    def test_bad_concurrency(self) -> None:
        harness = build_harness(produce=[], review=[])
        with pytest.raises(RunnerError, match="concurrency"):
            run_batch(harness, [], SqliteRunStore(), concurrency=0,
                      install_signal_handlers=False)

    def test_bad_report_every(self) -> None:
        harness = build_harness(produce=[], review=[])
        with pytest.raises(RunnerError, match="report_every"):
            run_batch(harness, [], SqliteRunStore(), report_every=0,
                      install_signal_handlers=False)

    def test_progress_callback_and_clock(self) -> None:
        ticks = iter([0.0, 10.0, 20.0, 30.0])
        seen: list[int] = []
        harness = build_harness(produce=[ok({"output": "R"})] * 2, review=[ACCEPT] * 2)
        store = _store(_records(2))
        run_batch(harness, store.pending(), store, report_every=1,
                  on_progress=lambda p: seen.append(p.done), clock=lambda: next(ticks),
                  install_signal_handlers=False)
        assert seen and seen[-1] == 2

    def test_failure_after_a_partial_append_leaves_completed_results_intact(self) -> None:
        # A worker failure must not discard results already appended for other members of the
        # same batch -- exercised with concurrency=2 so one record can succeed while the other
        # hits infrastructure failure.
        harness = build_harness(produce=[ok({"output": "R"})] * 2, review=[ACCEPT],
                                extra_validators=[_FailOn("line 1", LlmError("server is down"))])
        store = _store(_records(2))
        with pytest.raises(LlmError, match="server is down"):
            run_batch(harness, store.pending(), store, concurrency=2,
                      install_signal_handlers=False)
        # Record 0 completes and is appended; record 1's infrastructure failure stops the run
        # without burning it, so it stays PENDING for the resume.
        assert store.completed_ids() == {"0"}

    def test_the_captured_context_reaches_the_run_store(self) -> None:
        # End-to-end: a wired retriever's hits and the assembled passage flow from the harness's
        # capture sink (see harness.capture) through Outcome, run_batch's RunResult projection,
        # and the store's own JSON round-trip -- not just the store's own serialisation in
        # isolation (see TestCaptureRoundTrip in test_run_store.py).
        class _StubRetriever:
            def retrieve(self, query: str, *, k: int,
                         min_score: float = 0.0) -> tuple[Retrieved, ...]:
                return (Retrieved("c1", "an example", 0.9),)[:k]

        pool = build_pool([ok({"output": "R"})], [ACCEPT])
        panel = Panel(producer=Persona("p", "producer", "prod", instructions="produce"),
                     reviewers=(Persona("r", "reviewer", "rev", instructions="review"),))
        ruleset = RuleSet()
        context = ContextAssembler([_PlacedBlock("retrieved", RetrievedBlock())])
        harness = Harness(pool, panel, ruleset, JsonFieldSchema("output"),
                          ValidatorPipeline(ruleset), context, retriever=_StubRetriever())
        store = _store([Record(record_id="1", source="q")])
        run_batch(harness, store.pending(), store, install_signal_handlers=False)

        [result] = list(store.results())
        assert "an example" in result.context_passage
        assert result.retrieved and result.retrieved[0].chunk_id == "c1"


class TestResume:
    def test_pending_excludes_records_with_a_result(self) -> None:
        store = _store(_records(3))
        store.append_result(RunResult(
            record=Record(record_id="1", source="line 1", line_no=1, status=Status.VERIFIED,
                          output="x")))
        assert {r.record_id for r in store.pending()} == {"0", "2"}

    def test_completed_ids(self) -> None:
        store = _store([Record(record_id="7", source="s")])
        store.append_result(RunResult(
            record=Record(record_id="7", source="s", status=Status.VERIFIED, output="x")))
        assert store.completed_ids() == {"7"}

    def test_pending_sorts_by_provenance(self) -> None:
        store = _store([Record(record_id="b", source="s", line_no=2),
                        Record(record_id="a", source="s", line_no=1)])
        assert [r.record_id for r in store.pending()] == ["a", "b"]

    def test_a_resumed_run_does_not_repeat_completed_records(self) -> None:
        # Simulates a crashed-and-restarted run: record "0" already has a result: a resumed
        # run_batch over store.pending() must not re-produce it.
        store = _store(_records(2))
        store.append_result(RunResult(
            record=Record(record_id="0", source="line 0", status=Status.VERIFIED, output="OLD")))
        harness = build_harness(produce=[ok({"output": "NEW"})], review=[ACCEPT])
        progress = run_batch(harness, store.pending(), store, already_done=1,
                             install_signal_handlers=False)
        assert progress.done == 1 and progress.already_done == 1
        results = {r.record.record_id: r.record.output for r in store.results()}
        assert results == {"0": "OLD", "1": "NEW"}  # the old result was never touched


class TestGroupDuplicates:
    def test_groups_by_source_and_speaker(self) -> None:
        records = [Record(record_id="1", source="hi", meta={"speaker": "A"}),
                   Record(record_id="2", source="hi", meta={"speaker": "A"}),
                   Record(record_id="3", source="hi", meta={"speaker": "B"})]
        groups = group_duplicates(records)
        sizes = sorted(len(members) for _rep, members in groups)
        assert sizes == [1, 2]  # A's two share, B's one apart


class TestProgress:
    def test_counts_by_status(self) -> None:
        progress = Progress(total=4)
        for status in (Status.VERIFIED, Status.PRODUCED, Status.REJECTED, Status.SKIPPED):
            progress.record(Outcome(record=Record(record_id="x", source="s"), status=status,
                                    output="o" if status.is_injectable else None))
        assert (progress.verified, progress.produced, progress.rejected, progress.skipped) \
            == (1, 1, 1, 1)
        assert progress.done == 4 and progress.remaining == 0

    def test_non_terminal_status_has_no_bucket(self) -> None:
        progress = Progress(total=1)
        with pytest.raises(RunnerError, match="no counter"):
            progress.record(Outcome(record=Record(record_id="x", source="s"),
                                    status=Status.PENDING, output=None))

    def test_summary_is_readable(self) -> None:
        progress = Progress(total=10, already_done=5)
        progress.record(Outcome(record=Record(record_id="x", source="s"),
                                status=Status.VERIFIED, output="o"))
        assert "6/10 done" in progress.summary()
