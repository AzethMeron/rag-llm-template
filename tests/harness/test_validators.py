"""The mechanical checks, the validator pipeline, and the on-exhaustion partition."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ragkit.core.lexicon import Entry
from ragkit.core.records import Record
from ragkit.core.rules import Severity, Violation
from ragkit.harness.rules import RuleSet
from ragkit.harness.validators import (
    ValidatorPipeline,
    blocking,
    check_mechanical,
    partition_on_exhaustion,
)


def _rec(source: str = "hello", **meta: object) -> Record:
    return Record(record_id="1", source=source, meta=meta)


def _ids(violations: list[Violation]) -> set[str]:
    return {v.rule_id for v in violations}


class TestCheckMechanical:
    def test_clean_output_has_no_violations(self) -> None:
        assert check_mechanical(_rec(), "a fine output", RuleSet()) == []

    def test_empty_output_when_input_is_not(self) -> None:
        assert "nonempty" in _ids(check_mechanical(_rec(), "   ", RuleSet()))

    def test_empty_allowed_when_input_also_empty(self) -> None:
        assert check_mechanical(_rec("   "), "", RuleSet()) == []

    def test_placeholder_set_changed(self) -> None:
        assert "placeholders" in _ids(
            check_mechanical(_rec("[[0]] x"), "no placeholder", RuleSet()))

    def test_placeholder_repeated(self) -> None:
        v = check_mechanical(_rec("[[0]]"), "[[0]] [[0]]", RuleSet())
        assert any("repeated" in x.message for x in v)

    def test_placeholders_preserved_is_clean(self) -> None:
        assert check_mechanical(_rec("[[0]] [[1]]"), "[[1]] [[0]]", RuleSet()) == []

    def test_control_character(self) -> None:
        assert "control_character" in _ids(check_mechanical(_rec(), "bad\x00char", RuleSet()))

    def test_newline_is_allowed(self) -> None:
        # A raw newline is task-dependent, so the core control-char check permits it.
        assert "control_character" not in _ids(check_mechanical(_rec(), "line1\nline2", RuleSet()))

    def test_forbidden_pattern(self) -> None:
        rs = RuleSet(forbidden_patterns=(("^here is", "preamble"),))
        v = check_mechanical(_rec(), "here is the output", rs)
        assert any(x.rule_id == "forbidden" and "preamble" in x.message for x in v)

    def test_required_term_absent(self) -> None:
        v = check_mechanical(_rec(), "missing it", RuleSet(), required_terms=("Kraków",))
        assert v[0].rule_id == "lexicon" and v[0].severity is Severity.WARNING

    def test_required_term_present_is_clean(self) -> None:
        assert check_mechanical(_rec(), "in kraków today", RuleSet(),
                                required_terms=("Kraków",)) == []

    def test_hard_column_budget_blocks_past_tolerance(self) -> None:
        v = check_mechanical(_rec(), "x" * 20, RuleSet(max_columns_tolerance=0.1), max_columns=10)
        assert v[0].rule_id == "line_width" and v[0].severity is Severity.ERROR

    def test_hard_column_budget_warns_within_tolerance(self) -> None:
        v = check_mechanical(_rec(), "x" * 11, RuleSet(max_columns_tolerance=0.2), max_columns=10)
        assert v[0].severity is Severity.WARNING

    def test_within_hard_budget_is_clean(self) -> None:
        assert check_mechanical(_rec(), "short", RuleSet(), max_columns=100) == []

    def test_project_guideline_only_warns(self) -> None:
        v = check_mechanical(_rec(), "y" * 200, RuleSet(max_line_columns=50))
        assert v and all(x.severity is Severity.WARNING for x in v)


class TestPipeline:
    def test_requires_lexicon_terms_that_occur(self) -> None:
        pipeline = ValidatorPipeline(RuleSet(), [Entry(term="cat", rendering="kot")])
        v = pipeline.check(_rec("the cat"), "translated without the term")
        assert "lexicon" in _ids(v)

    def test_reads_max_columns_from_meta(self) -> None:
        pipeline = ValidatorPipeline(RuleSet(max_columns_tolerance=0.0))
        assert "line_width" in _ids(pipeline.check(_rec("hi", max_columns=3), "far too long"))

    def test_runs_extra_pluggable_validators(self) -> None:
        class Shouty:
            def validate(self, record: Record, output: str,
                         context: Mapping[str, Any]) -> list[Violation]:
                return [Violation("shout", Severity.ERROR, "no shouting")] if output.isupper() \
                    else []

        pipeline = ValidatorPipeline(RuleSet(), extra=[Shouty()])
        assert "shout" in _ids(pipeline.check(_rec(), "LOUD"))
        assert "shout" not in _ids(pipeline.check(_rec(), "quiet"))


class TestPartition:
    def test_blocking_filters_errors(self) -> None:
        vs = [Violation("a", Severity.ERROR, "m"), Violation("b", Severity.WARNING, "m")]
        assert [v.rule_id for v in blocking(vs)] == ["a"]

    def test_partition_splits_by_keep_flagged(self) -> None:
        rs = RuleSet(keep_flagged_rules=frozenset({"line_width"}))
        vs = [Violation("nonempty", Severity.ERROR, "m"),
              Violation("line_width", Severity.ERROR, "m"),
              Violation("note", Severity.WARNING, "m")]  # non-blocking, ignored
        reject, keep = partition_on_exhaustion(vs, rs)
        assert [v.rule_id for v in reject] == ["nonempty"]
        assert [v.rule_id for v in keep] == ["line_width"]
