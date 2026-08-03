"""The shared recipe-eval scaffolding: gold loading, the journal join, and the report tail.

This was six near-identical copies across the recipes -- and six near-identical copies of these
tests. The required-key check had already drifted between them; a fix to one copy's error message
reached none of the others. Tested once here, so each recipe's own tests cover only its scoring.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragkit.core.records import Record, Status
from ragkit.eval.gold import (
    EvalError,
    join_journal_with_gold,
    journal_gold_parser,
    json_field,
    load_label_gold,
    load_relevance_gold,
    run_report,
)


def _gold(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "gold.jsonl"
    path.write_text(text, encoding="utf-8")
    return path


class TestLoadLabelGold:
    def test_reads_rows_and_skips_blank_lines(self, tmp_path: Path) -> None:
        path = _gold(tmp_path, '{"record_id":"q1","decision":"yes"}\n\n'
                               '{"record_id":"q2","decision":"no"}\n')
        assert load_label_gold(path, field="decision") == {"q1": "yes", "q2": "no"}

    def test_the_field_name_is_the_caller_s(self, tmp_path: Path) -> None:
        path = _gold(tmp_path, '{"record_id":"e1","severity":"urgent"}\n')
        assert load_label_gold(path, field="severity") == {"e1": "urgent"}

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(EvalError, match="not found"):
            load_label_gold(tmp_path / "no.jsonl", field="decision")

    def test_invalid_json_names_the_line(self, tmp_path: Path) -> None:
        with pytest.raises(EvalError, match=r"gold\.jsonl:1: invalid JSON"):
            load_label_gold(_gold(tmp_path, "{bad\n"), field="decision")

    def test_a_missing_required_key_names_it(self, tmp_path: Path) -> None:
        with pytest.raises(EvalError, match=r"needs 'record_id', 'decision'"):
            load_label_gold(_gold(tmp_path, '{"record_id":"q1"}\n'), field="decision")

    def test_a_non_object_row_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(EvalError, match="must be a JSON object"):
            load_label_gold(_gold(tmp_path, "[1, 2]\n"), field="decision")

    def test_empty_file(self, tmp_path: Path) -> None:
        with pytest.raises(EvalError, match="empty"):
            load_label_gold(_gold(tmp_path, "\n"), field="decision")


class TestLoadRelevanceGold:
    def test_reads_relevant_id_sets(self, tmp_path: Path) -> None:
        path = _gold(tmp_path, '{"record_id":"q1","relevant":["p1","p2"]}\n')
        assert load_relevance_gold(path) == {"q1": frozenset({"p1", "p2"})}

    def test_an_empty_relevant_list_is_refused(self, tmp_path: Path) -> None:
        # Otherwise it becomes a query that scores 0 by construction.
        with pytest.raises(EvalError, match="non-empty list"):
            load_relevance_gold(_gold(tmp_path, '{"record_id":"q1","relevant":[]}\n'))

    def test_a_non_list_relevant_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(EvalError, match="non-empty list"):
            load_relevance_gold(_gold(tmp_path, '{"record_id":"q1","relevant":"p1"}\n'))


class TestJoinJournalWithGold:
    def _journal(self, tmp_path: Path, records: list[Record]) -> Path:
        path = tmp_path / "j.jsonl"
        path.write_text("".join(r.to_json() + "\n" for r in records), encoding="utf-8")
        return path

    def test_pairs_gold_with_what_the_journal_produced(self, tmp_path: Path) -> None:
        journal = self._journal(tmp_path, [
            Record(record_id="r1", source="s", output="OUT", status=Status.VERIFIED)])
        assert join_journal_with_gold(journal, {"r1": "yes"}) == [("r1", "OUT", "yes")]

    def test_a_non_injectable_result_pairs_as_a_miss(self, tmp_path: Path) -> None:
        # A rejected record must score as a miss, not as an empty-string answer -- the latter
        # would quietly flatter every recipe's numbers.
        journal = self._journal(tmp_path, [
            Record(record_id="r1", source="s", status=Status.REJECTED)])
        assert join_journal_with_gold(journal, {"r1": "yes"}) == [("r1", None, "yes")]

    def test_a_gold_id_absent_from_the_journal_pairs_as_a_miss(self, tmp_path: Path) -> None:
        journal = self._journal(tmp_path, [])
        assert join_journal_with_gold(journal, {"never_ran": "yes"}) == [("never_ran", None, "yes")]

    def test_the_result_follows_gold_order_not_journal_order(self, tmp_path: Path) -> None:
        journal = self._journal(tmp_path, [
            Record(record_id="b", source="s", output="B", status=Status.VERIFIED),
            Record(record_id="a", source="s", output="A", status=Status.VERIFIED)])
        assert [rid for rid, _, _ in join_journal_with_gold(journal, {"a": "1", "b": "2"})] == [
            "a", "b"]


class TestJsonField:
    def test_reads_and_normalises_the_field(self) -> None:
        assert json_field(json.dumps({"decision": "  YES "}), "decision") == "yes"

    @pytest.mark.parametrize("produced", [
        None,                       # nothing produced
        "not json at all",          # unparseable
        '{"other": "yes"}',         # field absent
        '{"decision": 7}',          # field not a string
        '["a list"]',               # not an object
    ])
    def test_anything_unusable_is_a_miss_not_a_crash(self, produced: str | None) -> None:
        assert json_field(produced, "decision") is None


class TestReportTail:
    def test_prints_and_returns_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert run_report(lambda: "accuracy 1.000") == 0
        assert capsys.readouterr().out.strip() == "accuracy 1.000"

    def test_an_eval_error_goes_to_stderr_with_exit_one(
            self, capsys: pytest.CaptureFixture[str]) -> None:
        def boom() -> str:
            raise EvalError("gold file is empty: g.jsonl")

        assert run_report(boom) == 1
        captured = capsys.readouterr()
        assert captured.out == "" and "gold file is empty" in captured.err

    def test_the_parser_takes_journal_and_gold(self) -> None:
        parser = journal_gold_parser("desc", gold_help="gold things (JSONL)")
        args = parser.parse_args(["--journal", "j.jsonl", "--gold", "g.jsonl"])
        assert args.journal == Path("j.jsonl") and args.gold == Path("g.jsonl")
