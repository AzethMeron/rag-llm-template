"""The harness loop: every terminal outcome and the paths that reach it."""
from __future__ import annotations

import logging
import re

import pytest

from ragkit.core.ports import Retrieved
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
from ragkit.harness.context import ContextAssembler, LiteralBlock, RetrievedBlock
from ragkit.harness.context.assembler import _PlacedBlock
from ragkit.llm.errors import LlmError

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


class _BoomValidator:
    """A pluggable validator with a bug in it — the poison-pill shape."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def validate(self, _record: Record, _output: str, _context: dict) -> list:
        raise self._exc


class TestPluginDefects:
    """A defect in pluggable code must burn one record, never the whole batch. Regression for the
    poison pill: an exception escaping ``process`` left the record PENDING and aborted the run, so
    every resume hit the same deterministic exception and aborted again."""

    def test_validator_exception_rejects_only_this_record(self) -> None:
        harness = build_harness(produce=[ok({"output": "R"})], review=[],
                                extra_validators=[_BoomValidator(re.error("bad pattern"))])
        outcome = harness.process(_record())
        assert outcome.status is Status.REJECTED and outcome.output is None

    def test_the_diagnostic_names_the_type_and_the_failing_frame(self) -> None:
        harness = build_harness(produce=[ok({"output": "R"})], review=[],
                                extra_validators=[_BoomValidator(KeyError("speaker"))])
        error = harness.process(_record()).error or ""
        assert "unexpected KeyError" in error and "in validate()" in error
        assert "test_agents.py:" in error and "speaker" in error

    def test_the_defect_is_logged_with_a_traceback(
            self, caplog: pytest.LogCaptureFixture) -> None:
        harness = build_harness(produce=[ok({"output": "R"})], review=[],
                                extra_validators=[_BoomValidator(ValueError("nope"))])
        with caplog.at_level(logging.ERROR, logger="ragkit.harness.agents"):
            harness.process(_record())
        assert "ValueError" in caplog.text and "Traceback" in caplog.text

    def test_output_schema_exception_is_also_contained(self) -> None:
        class _BoomSchema:
            name = "boom"

            def json_schema(self) -> dict:
                return {"type": "object", "properties": {"output": {"type": "string"}}}

            def extract(self, _reply: dict) -> str:
                raise TypeError("schema plugin bug")

        harness = build_harness(produce=[ok({"output": "R"})], review=[])
        harness.output_schema = _BoomSchema()  # type: ignore[assignment]
        outcome = harness.process(_record())
        assert outcome.status is Status.REJECTED
        assert "unexpected TypeError" in (outcome.error or "")

    def test_a_bare_llm_error_still_propagates(self) -> None:
        # The broad handler must not swallow infrastructure failure: "the server is down" has to
        # stop the run, leaving the record PENDING for a resume, not burn it as REJECTED.
        harness = build_harness(produce=[ok({"output": "R"})], review=[],
                                extra_validators=[_BoomValidator(LlmError("server is down"))])
        with pytest.raises(LlmError, match="server is down"):
            harness.process(_record())

    def test_keyboard_interrupt_still_propagates(self) -> None:
        harness = build_harness(produce=[ok({"output": "R"})], review=[],
                                extra_validators=[_BoomValidator(KeyboardInterrupt())])
        with pytest.raises(KeyboardInterrupt):
            harness.process(_record())


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


class _StubRetriever:
    def __init__(self, hits: list[Retrieved]) -> None:
        self._hits = hits

    def retrieve(self, query: str, *, k: int, min_score: float = 0.0) -> tuple[Retrieved, ...]:
        return tuple(self._hits[:k])


class TestCapture:
    """Outcome.context_passage/retrieved: what actually produced the output, captured from the
    calls RetrievedBlock and the harness's own prompt assembly already make -- never a second
    retrieval (see ragkit.harness.capture)."""

    def _harness_with_retriever(self, retriever: _StubRetriever) -> Harness:
        pool = build_pool([ok({"output": "RESULT"})], [ACCEPT])
        panel = Panel(producer=Persona("p", "producer", "prod", instructions="produce the output"),
                     reviewers=(Persona("r", "reviewer", "rev", instructions="review it"),))
        ruleset = RuleSet()
        context = ContextAssembler([_PlacedBlock("retrieved", RetrievedBlock())])
        return Harness(pool, panel, ruleset, JsonFieldSchema("output"),
                       ValidatorPipeline(ruleset), context, retriever=retriever)

    def test_retrieved_hits_and_passage_are_captured(self) -> None:
        hit = Retrieved("c1", "an example", 0.9)
        harness = self._harness_with_retriever(_StubRetriever([hit]))
        outcome = harness.process(_record())
        assert outcome.status is Status.VERIFIED
        assert outcome.retrieved == (hit,)
        assert "an example" in outcome.context_passage

    def test_no_hits_leaves_retrieved_empty(self) -> None:
        harness = self._harness_with_retriever(_StubRetriever([]))
        outcome = harness.process(_record())
        assert outcome.retrieved == ()

    def test_skipped_record_captures_nothing(self) -> None:
        harness = build_harness(produce=[], review=[])
        outcome = harness.process(_record("   "))
        assert outcome.context_passage == "" and outcome.retrieved == ()

    def test_passage_captured_even_without_a_retriever(self) -> None:
        # build_harness's fixed context is a literal-only block; capture still records the
        # assembled passage even though nothing was retrieved.
        harness = build_harness(produce=[ok({"output": "RESULT"})], review=[ACCEPT])
        outcome = harness.process(_record())
        assert outcome.context_passage == "Do the task." and outcome.retrieved == ()

    def test_rejected_outcome_still_carries_its_capture(self) -> None:
        # Force a rejection: empty output against a non-empty source, no reviewers to consult.
        hit = Retrieved("c1", "an example", 0.9)
        pool = build_pool([ok({"output": ""})] * 3, [])
        panel = Panel(producer=Persona("p", "producer", "prod", instructions="x"),
                     reviewers=(), max_repairs=2)
        ruleset = RuleSet()
        context = ContextAssembler([_PlacedBlock("retrieved", RetrievedBlock())])
        harness = Harness(pool, panel, ruleset, JsonFieldSchema("output"),
                          ValidatorPipeline(ruleset), context, retriever=_StubRetriever([hit]))
        outcome = harness.process(_record())
        assert outcome.status is Status.REJECTED
        assert outcome.retrieved == (hit,)


class TestAttempt:
    def test_previous_attempt_reaches_the_producer_via_context(self) -> None:
        # The revision includes the previous attempt as a context block, so the second produce
        # sees it -- exercised indirectly by the objection-then-accept flow above; here we just
        # confirm the Attempt shape carries suggestions.
        attempt = Attempt(target="t", issues=("a",), suggestions=("s",))
        assert attempt.target == "t" and attempt.suggestions == ("s",)
