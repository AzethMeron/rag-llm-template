"""The harness loop: every terminal outcome and the paths that reach it."""
from __future__ import annotations

import pytest

from ragkit.core.records import Record, Status
from ragkit.harness import (
    Harness,
    JsonFieldSchema,
    OutputMemory,
    Panel,
    Persona,
    RuleSet,
    ValidatorPipeline,
    learn_memory,
)
from ragkit.harness.agents import Attempt, Outcome
from ragkit.harness.context import ContextAssembler, LiteralBlock
from ragkit.harness.context.assembler import _PlacedBlock

from .conftest import ACCEPT, build_harness, build_pool, ok, raw, refuse, truncated


def _object(*issues: str, improved: str | None = None) -> tuple[str | None, str]:
    payload: dict = {"acceptable": False, "issues": list(issues)}
    if improved is not None:
        payload["improved_output"] = improved
    return ok(payload)


def _record(source: str = "hello", **meta: object) -> Record:
    return Record(record_id="1", source=source, meta=meta)


class TestHappyPath:
    def test_verified_when_produced_and_accepted(self) -> None:
        harness = build_harness(produce=[ok({"output": "RESULT"})], review=[ACCEPT])
        outcome = harness.process(_record())
        assert outcome.status is Status.VERIFIED
        assert outcome.output == "RESULT" and outcome.rounds == 1

    def test_skipped_blank_input(self) -> None:
        harness = build_harness(produce=[], review=[])
        assert harness.process(_record("   ")).status is Status.SKIPPED

    def test_skipped_status_short_circuits(self) -> None:
        harness = build_harness(produce=[], review=[])
        record = Record(record_id="1", source="x", status=Status.SKIPPED)
        assert harness.process(record).status is Status.SKIPPED


class TestRevision:
    def test_objection_then_accept_verifies(self) -> None:
        harness = build_harness(
            produce=[ok({"output": "first"}), ok({"output": "second"})],
            review=[_object("too literal", improved="better"), ACCEPT])
        outcome = harness.process(_record())
        assert outcome.status is Status.VERIFIED
        assert outcome.output == "second" and outcome.rounds == 2

    def test_unresolved_within_budget_is_produced(self) -> None:
        harness = build_harness(
            produce=[ok({"output": "a"}), ok({"output": "b"}), ok({"output": "c"})],
            review=[_object("x"), _object("y"), _object("z")], max_revisions=2)
        outcome = harness.process(_record())
        assert outcome.status is Status.PRODUCED
        assert "unresolved within budget" in (outcome.error or "")


class TestMechanicalFailure:
    def test_empty_output_is_rejected_after_repairs(self) -> None:
        harness = build_harness(produce=[ok({"output": ""})] * 3, review=[], max_repairs=2)
        outcome = harness.process(_record())
        assert outcome.status is Status.REJECTED
        assert any(v.rule_id == "nonempty" for v in outcome.violations)

    def test_placeholder_repair_then_success(self) -> None:
        harness = build_harness(
            produce=[ok({"output": "no placeholder"}), ok({"output": "[[0]] fixed"})],
            review=[ACCEPT], max_repairs=2)
        outcome = harness.process(_record("[[0]] source"))
        assert outcome.status is Status.VERIFIED and outcome.output == "[[0]] fixed"

    def test_keep_flagged_rule_keeps_the_output(self) -> None:
        ruleset = RuleSet(keep_flagged_rules=frozenset({"line_width"}))
        harness = build_harness(produce=[ok({"output": "far too long to fit"})] * 3, review=[],
                                ruleset=ruleset, max_repairs=2)
        outcome = harness.process(_record("hi", max_columns=5))
        assert outcome.status is Status.PRODUCED
        assert "flagged rule" in (outcome.error or "")


class TestReviewAbstention:
    def test_unreadable_reviewer_reply_keeps_for_review(self) -> None:
        harness = build_harness(produce=[ok({"output": "x"})], review=[raw("not json")])
        outcome = harness.process(_record())
        assert outcome.status is Status.PRODUCED
        assert "review incomplete" in (outcome.error or "")

    def test_a_later_reviewer_still_runs_after_an_abstention(self) -> None:
        # Two reviewers: the first abstains (unreadable), the second accepts. The panel did not
        # fully run, so the record is kept for review rather than verified.
        harness = build_harness(produce=[ok({"output": "x"})],
                                review=[raw("garbage"), ACCEPT], reviewers=2)
        outcome = harness.process(_record())
        assert outcome.status is Status.PRODUCED
        assert len(outcome.reviews) == 2


class TestEnvelopeRecovery:
    def test_recovered_envelope_is_kept_but_unconfirmed(self) -> None:
        # finish_reason "stop" (the helper default) with truncated JSON is the recoverable
        # envelope case; a "length" finish is the distinct truncation error.
        harness = build_harness(produce=[raw('{"output": "partial')], review=[ACCEPT],
                                max_repairs=0)
        outcome = harness.process(_record())
        assert outcome.status is Status.PRODUCED
        assert "truncated JSON envelope" in (outcome.error or "")
        assert outcome.output == "partial"

    def test_recovery_disabled_rejects(self) -> None:
        harness = build_harness(produce=[raw('{"output": "partial')], review=[],
                                max_repairs=0, repair_truncated_json=False)
        assert harness.process(_record()).status is Status.REJECTED

    def test_length_finish_is_a_truncation_rejection(self) -> None:
        harness = build_harness(produce=[truncated('{"output": "unfin')], review=[], max_repairs=0)
        assert harness.process(_record()).status is Status.REJECTED


class TestInfrastructureVsContent:
    def test_refusal_is_rejected(self) -> None:
        harness = build_harness(produce=[refuse()], review=[])
        outcome = harness.process(_record())
        assert outcome.status is Status.REJECTED and "refused" in (outcome.error or "")

    def test_unparseable_output_is_rejected_after_repairs(self) -> None:
        harness = build_harness(produce=[raw("not json at all")] * 3, review=[], max_repairs=2)
        assert harness.process(_record()).status is Status.REJECTED


class TestConstruction:
    def _panel(self, from_rules: bool) -> Panel:
        return Panel(producer=Persona("p", "producer", "prod", instructions="do"),
                     reviewers=(Persona("c", "reviewer", "rev", from_rules=from_rules,
                                        instructions="" if from_rules else "review"),))

    def test_from_rules_reviewer_without_advisory_is_refused(self) -> None:
        pool = build_pool([], [])
        context = ContextAssembler([_PlacedBlock("literal", LiteralBlock("x"))])
        with pytest.raises(ValueError, match="no \\[\\[advisory\\]\\]"):
            Harness(pool, self._panel(from_rules=True), RuleSet(), JsonFieldSchema(),
                    ValidatorPipeline(RuleSet()), context)

    def test_from_rules_reviewer_with_advisory_is_allowed(self) -> None:
        pool = build_pool([], [])
        context = ContextAssembler([_PlacedBlock("literal", LiteralBlock("x"))])
        ruleset = RuleSet(advisory_rules=(("register", "keep the register formal"),))
        Harness(pool, self._panel(from_rules=True), ruleset, JsonFieldSchema(),
                ValidatorPipeline(ruleset), context)  # no error


class TestOutcomeApplication:
    def test_applied_to_carries_only_the_verdict(self) -> None:
        harness = build_harness(produce=[ok({"output": "R"})], review=[ACCEPT])
        outcome = harness.process(_record())
        member = Record(record_id="other", source="hello", rel_path="f", line_no=9)
        applied = outcome.applied_to(member)
        assert applied.record_id == "other" and applied.rel_path == "f"  # provenance kept
        assert applied.output == "R" and applied.status is Status.VERIFIED

    def test_notes_capture_violations_and_errors(self) -> None:
        outcome = Outcome(record=_record(), status=Status.REJECTED, output=None,
                          error="mechanical rules still violated after repairs")
        assert any("error:" in n for n in outcome.applied().notes)


class TestLearnMemory:
    def test_injectable_output_is_learned(self) -> None:
        memory = OutputMemory()
        outcome = Outcome(record=_record(), status=Status.VERIFIED, output="R")
        learn_memory(memory, outcome)
        assert memory.get("hello") == "R"

    def test_rejected_output_is_not_learned(self) -> None:
        memory = OutputMemory()
        learn_memory(memory, Outcome(record=_record(), status=Status.REJECTED, output="bad"))
        assert len(memory) == 0

    def test_no_memory_is_a_no_op(self) -> None:
        learn_memory(None, Outcome(record=_record(), status=Status.VERIFIED, output="R"))


class TestAttempt:
    def test_previous_attempt_reaches_the_producer_via_context(self) -> None:
        # The revision includes the previous attempt as a context block, so the second produce
        # sees it -- exercised indirectly by the objection-then-accept flow above; here we just
        # confirm the Attempt shape carries suggestions.
        attempt = Attempt(target="t", issues=("a",), suggestions=("s",))
        assert attempt.target == "t" and attempt.suggestions == ("s",)
