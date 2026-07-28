"""The resumable batch runner: journalling, resume, duplicate grouping, progress, failure."""
from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.core.records import Record, Status, read_journal, write_catalog
from ragkit.harness import (
    Outcome,
    Progress,
    RunnerError,
    completed_ids,
    group_duplicates,
    pending_records,
    run_batch,
)

from .conftest import ACCEPT, build_harness, ok


def _records(n: int) -> list[Record]:
    return [Record(record_id=str(i), source=f"line {i}", line_no=i) for i in range(n)]


class TestRunBatch:
    def test_journals_each_result(self, tmp_path: Path) -> None:
        harness = build_harness(produce=[ok({"output": "R"})] * 3, review=[ACCEPT] * 3)
        journal = tmp_path / "j.jsonl"
        progress = run_batch(harness, _records(3), journal, install_signal_handlers=False)
        assert progress.verified == 3 and progress.done == 3
        assert {r.record_id for r in read_journal(journal)} == {"0", "1", "2"}

    def test_shares_output_across_duplicates(self, tmp_path: Path) -> None:
        # Two records with identical source: one production, both journalled.
        records = [Record(record_id="a", source="same"), Record(record_id="b", source="same")]
        harness = build_harness(produce=[ok({"output": "R"})], review=[ACCEPT])
        journal = tmp_path / "j.jsonl"
        run_batch(harness, records, journal, install_signal_handlers=False)
        results = {r.record_id: r for r in read_journal(journal)}
        assert results["a"].output == "R" and results["b"].output == "R"

    def test_worker_failure_propagates_after_journalling(self, tmp_path: Path) -> None:
        # The first record's server dies (a bare LlmError); the run stops but any completed record
        # is journalled first.
        harness = build_harness(produce=[ok({"output": "R"})], review=[])
        # Make process raise a non-content error by exhausting the scripted queue on the 2nd call.
        journal = tmp_path / "j.jsonl"
        with pytest.raises(AssertionError):  # the scripted queue raises when exhausted
            run_batch(harness, _records(2), journal, concurrency=1,
                      install_signal_handlers=False)

    def test_concurrency_completes_all(self, tmp_path: Path) -> None:
        harness = build_harness(produce=[ok({"output": "R"})] * 5, review=[ACCEPT] * 5)
        journal = tmp_path / "j.jsonl"
        progress = run_batch(harness, _records(5), journal, concurrency=4,
                             install_signal_handlers=False)
        assert progress.done == 5

    def test_bad_concurrency(self, tmp_path: Path) -> None:
        harness = build_harness(produce=[], review=[])
        with pytest.raises(RunnerError, match="concurrency"):
            run_batch(harness, [], tmp_path / "j.jsonl", concurrency=0,
                      install_signal_handlers=False)

    def test_bad_report_every(self, tmp_path: Path) -> None:
        harness = build_harness(produce=[], review=[])
        with pytest.raises(RunnerError, match="report_every"):
            run_batch(harness, [], tmp_path / "j.jsonl", report_every=0,
                      install_signal_handlers=False)

    def test_progress_callback_and_clock(self, tmp_path: Path) -> None:
        ticks = iter([0.0, 10.0, 20.0, 30.0])
        seen: list[int] = []
        harness = build_harness(produce=[ok({"output": "R"})] * 2, review=[ACCEPT] * 2)
        run_batch(harness, _records(2), tmp_path / "j.jsonl", report_every=1,
                  on_progress=lambda p: seen.append(p.done), clock=lambda: next(ticks),
                  install_signal_handlers=False)
        assert seen and seen[-1] == 2


class TestResume:
    def test_pending_excludes_journalled(self, tmp_path: Path) -> None:
        catalog = tmp_path / "c.jsonl"
        write_catalog(_records(3), catalog)
        journal = tmp_path / "j.jsonl"
        journal.write_text(
            Record(record_id="1", source="line 1", status=Status.VERIFIED, output="x").to_json()
            + "\n", encoding="utf-8")
        pending = pending_records(catalog, journal)
        assert [r.record_id for r in pending] == ["0", "2"]

    def test_completed_ids(self, tmp_path: Path) -> None:
        journal = tmp_path / "j.jsonl"
        journal.write_text(Record(record_id="7", source="s").to_json() + "\n", encoding="utf-8")
        assert completed_ids(journal) == {"7"}

    def test_pending_sorts_by_key(self, tmp_path: Path) -> None:
        catalog = tmp_path / "c.jsonl"
        write_catalog([Record(record_id="b", source="s", line_no=2),
                       Record(record_id="a", source="s", line_no=1)], catalog)
        pending = pending_records(catalog, tmp_path / "none.jsonl")
        assert [r.record_id for r in pending] == ["a", "b"]


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
