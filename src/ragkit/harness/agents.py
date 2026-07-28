"""The harness: the bounded produce → mechanical check → review panel → revise loop.

This is the task-agnostic core lifted from a translation engine, where it turned out to have
nothing to do with translation. The mechanical check sits between the model steps on purpose:
placeholder integrity, emptiness, forbidden patterns and width are decidable in code, so they are
settled before any GPU time is spent on an opinion, and an output that fails them is sent straight
back with a precise reason. Only what needs judgement reaches the panel of reviewer personas.

Errors are handled by blast radius (see :mod:`ragkit.llm.errors`): an ``LlmContentError`` on the
produce path records this record ``REJECTED`` and lets the run continue; a bare ``LlmError``
propagates and stops the run with the journal intact. Everything is bounded — ``max_repairs``
mechanical repairs per generation, ``max_revisions`` review rounds per record — after which the
output is recorded rather than retried forever.

Prompt assembly obeys one rule throughout: an optional section is included only when it has
content, so nothing reaches the model but signal.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from ragkit.core.placeholders import placeholder_indices
from ragkit.core.ports import Message, OutputSchema, Retriever, SchemaIntrospector, SqlStore
from ragkit.core.records import Record, Status
from ragkit.core.rules import Violation
from ragkit.llm.errors import (
    LlmContentError,
    LlmIncompleteJsonError,
    LlmRefusalError,
    LlmTruncationError,
)
from ragkit.llm.pool import ModelPool

from .context import ContextAssembler
from .memory import OutputMemory
from .roles import Panel, Persona
from .rules import RuleSet
from .validators import ValidatorPipeline, blocking, partition_on_exhaustion

logger = logging.getLogger(__name__)

PLACEHOLDER_CONTRACT = """
The text contains placeholders written as [[0]], [[1]], and so on. Each stands for a piece of
code the surrounding program will substitute. These rules override every other instruction:
- Reproduce every placeholder exactly, including the double brackets.
- Use each placeholder exactly once. Never add, drop, merge, or renumber them.
- You may move a placeholder to wherever the output's word order requires.
- Never translate, expand, or guess what a placeholder contains.
""".strip()

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["acceptable", "issues"],
    "properties": {
        "acceptable": {"type": "boolean"},
        # Nullable because a reviewer that accepts has no issues and, in json_object mode, encodes
        # that empty list as null; the consumer coerces null/absent to an empty list.
        "issues": {"type": ["array", "null"], "items": {"type": "string"}},
        "improved_output": {"type": ["string", "null"]},
    },
}


def _clip(text: str, limit: int = 200) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


@dataclass(frozen=True, slots=True)
class Review:
    role: str
    acceptable: bool
    issues: tuple[str, ...] = ()
    improved: str | None = None
    error: str | None = None
    """Set when this reviewer's reply could not be evaluated: an abstention that cannot veto, but
    means the panel did not fully run, so the record is kept for review rather than verified."""


@dataclass(frozen=True, slots=True)
class Attempt:
    """A rejected output and everything known about why, carried into the next try so the model
    revises rather than re-produces blind."""

    target: str
    issues: tuple[str, ...]
    suggestions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Outcome:
    """Result of running one record through the full harness."""

    record: Record
    status: Status
    output: str | None
    violations: tuple[Violation, ...] = ()
    reviews: tuple[Review, ...] = ()
    rounds: int = 0
    error: str | None = None

    def applied(self) -> Record:
        return self.applied_to(self.record)

    def applied_to(self, record: Record) -> Record:
        """This outcome written onto ``record`` (used to share one output across duplicates: only
        the verdict travels, so each occurrence keeps its own id and provenance)."""
        notes = tuple(f"{v.rule_id}: {v.message}" for v in self.violations)
        notes += tuple(f"{r.role}: {issue}" for r in self.reviews for issue in r.issues)
        notes += tuple(f"{r.role}: could not be evaluated ({r.error})"
                       for r in self.reviews if r.error)
        if self.error:
            notes += (f"error: {self.error}",)
        return replace(record, status=self.status, output=self.output, notes=notes)


class Harness:
    """The personas and the loop that sequences them, over one record at a time.

    Thread-safe for :func:`ragkit.harness.runner.run_batch`'s concurrent workers: :meth:`process`
    is free of shared mutable state except the per-reviewer leniency window (guarded here), the
    model pool's usage stats (guarded there), and the injected memory (guarded there). The context
    retriever and sql store are read-only and need no lock.
    """

    def __init__(self, pool: ModelPool, panel: Panel, ruleset: RuleSet,
                 output_schema: OutputSchema, validators: ValidatorPipeline,
                 context: ContextAssembler, *, memory: OutputMemory | None = None,
                 retriever: Retriever | None = None, sql_store: SqlStore | None = None,
                 introspector: SchemaIntrospector | None = None,
                 sanitize: Callable[[str], str] = lambda text: text,
                 input_label: str = "Input to act on:",
                 stand_in: str = "they") -> None:
        self.pool = pool
        self.panel = panel
        self.ruleset = ruleset
        self.output_schema = output_schema
        self.validators = validators
        self.context = context
        self.memory = memory
        self.retriever = retriever
        self.sql_store = sql_store
        self.introspector = introspector
        self.sanitize = sanitize
        self.input_label = input_label
        self.stand_in = stand_in
        if any(r.from_rules for r in panel.reviewers) and not ruleset.advisory_rules:
            raise ValueError(
                "a from_rules reviewer is configured but the rule set has no [[advisory]] "
                "criteria, so it would judge against nothing; add advisory rules or drop it")
        self._reply_history: dict[str, deque[bool]] = {}
        self._history_lock = threading.Lock()

    # -- prompt assembly -----------------------------------------------------

    def _shared_context(self, previous: Attempt | None) -> dict[str, Any]:
        return {"lexicon": self.validators.lexicon, "memory": self.memory,
                "retriever": self.retriever, "sql_store": self.sql_store,
                "introspector": self.introspector, "previous_attempt": previous,
                "stand_in": self.stand_in}

    def _system_prompt(self, instructions: str, *, placeholders: bool) -> str:
        sections = [instructions]
        if placeholders:
            sections.append(PLACEHOLDER_CONTRACT)
        if self.ruleset.style_directives:
            directives = "\n".join(f"- {d}" for d in self.ruleset.style_directives)
            sections.append(f"Style policy:\n{directives}")
        sections.append("Respond only with the requested JSON object.")
        return "\n\n".join(sections)

    def _user_prompt(self, record: Record, previous: Attempt | None) -> str:
        passage = self.context.assemble(record, self._shared_context(previous))
        parts = [passage] if passage else []
        parts.append(f"{self.input_label}\n{record.source}")
        return "\n\n".join(parts)

    # -- roles ---------------------------------------------------------------

    def produce(self, record: Record, previous: Attempt | None = None) -> str:
        persona = self.panel.producer
        client, model_id = self.pool.client_for(persona.model)
        has_placeholders = bool(placeholder_indices(record.source))
        reply = client.complete_json(
            [Message("system", self._system_prompt(persona.instructions,
                                                   placeholders=has_placeholders)),
             Message("user", self._user_prompt(record, previous))],
            self.output_schema.json_schema(), role=persona.id, temperature=0.3,
            max_tokens=self.panel.limits.produce_budget(len(record.source)), model=model_id)
        return self.sanitize(self.output_schema.extract(reply))

    def review(self, record: Record, output: str, reviewer: Persona) -> Review:
        client, model_id = self.pool.client_for(reviewer.model)
        instructions = self._rule_instructions() if reviewer.from_rules else reviewer.instructions
        has_placeholders = bool(placeholder_indices(record.source))
        user = (
            f"{self._user_prompt(record, None)}\n\n"
            f"Proposed output:\n{output}\n\n"
            f"Judge only your assigned criteria. Acceptable is the expected answer: object only "
            f"when you can name a definite error a competent editor would have to fix. If you do "
            f"object, list each concrete problem and supply a corrected output.")
        reply = client.complete_json(
            [Message("system", self._system_prompt(instructions, placeholders=has_placeholders)),
             Message("user", user)],
            REVIEW_SCHEMA, role=reviewer.id, temperature=0.0,
            max_tokens=self.panel.limits.review_budget(reviewer), model=model_id)
        improved = reply.get("improved_output") or None
        acceptable = bool(reply["acceptable"])
        issues = tuple(str(i) for i in (reply.get("issues") or []))
        if not acceptable and not issues:
            issues = ("marked unacceptable without a specific reason; reconsider the output "
                      "against the assigned criteria",)
        return Review(role=reviewer.id, acceptable=acceptable, issues=issues,
                      improved=improved if improved and improved != output else None)

    def review_panel(self, record: Record, output: str) -> list[Review]:
        """Run reviewers in order, stopping at the first objection. An unreadable reply is
        isolated as an abstention (it cannot veto) so it does not abort the panel; the record is
        then kept for review rather than verified on a panel that did not fully run."""
        reviews: list[Review] = []
        for reviewer in self.panel.reviewers:
            try:
                review = self.review(record, output, reviewer)
            except LlmContentError as exc:
                surfaces = self._register_reply(reviewer, bad=True)
                logger.warning(
                    "reviewer %r could not evaluate record %s; it abstains and the record is kept "
                    "for human review. input: %s | model reply: %s",
                    reviewer.id, record.record_id, _clip(record.source),
                    _clip(exc.body or "<empty>", 1000),
                    extra={"leniency_suppress_console": not surfaces})
                reviews.append(Review(role=reviewer.id, acceptable=True,
                                      error=f"reply could not be evaluated: {exc}"))
                continue
            self._register_reply(reviewer, bad=False)
            reviews.append(review)
            if not review.acceptable:
                break
        return reviews

    def _register_reply(self, reviewer: Persona, *, bad: bool) -> bool:
        with self._history_lock:
            history = self._reply_history.get(reviewer.id)
            if history is None or history.maxlen != reviewer.leniency.window:
                history = deque(history or (), maxlen=reviewer.leniency.window)
                self._reply_history[reviewer.id] = history
            history.append(bad)
            return bad and reviewer.leniency.surfaces(sum(history))

    def _rule_instructions(self) -> str:
        criteria = "\n".join(f"- {rule_id}: {description}"
                             for rule_id, description in self.ruleset.advisory_rules)
        return ("You are a compliance checker. Judge the output against exactly these project "
                f"rules and nothing else:\n{criteria}")

    # -- orchestration -------------------------------------------------------

    def _generate(self, record: Record,
                  feedback: Attempt | None) -> tuple[str, list[Violation], bool]:
        """Produce and mechanically repair until the hard rules pass or the budget runs out.

        The third return value is ``True`` when the output came from a recovered (truncated) JSON
        envelope rather than an ordinary reply: never trusted at face value, it is mechanically
        checked and reviewed like any candidate, and this loop spends at least one more repair
        round trying for an ordinary reply before falling back to it on the budget-exhausting round.
        """
        carried = feedback.issues if feedback else ()
        suggestions = feedback.suggestions if feedback else ()
        attempt = feedback
        target = ""
        violations: list[Violation] = []
        for round_index in range(self.panel.max_repairs + 1):
            last_round = round_index == self.panel.max_repairs
            try:
                target = self.produce(record, attempt)
            except (LlmRefusalError, LlmTruncationError):
                raise
            except LlmIncompleteJsonError as exc:
                if not self.panel.repair_truncated_json:
                    if last_round:
                        raise
                    attempt = Attempt(target=target, issues=(*carried, _RETRY_JSON),
                                      suggestions=suggestions)
                    continue
                target = self.sanitize(self.output_schema.extract(exc.recovered))
                violations = self.validators.check(record, target)
                if last_round:
                    return target, violations, True
                attempt = Attempt(target=target, issues=(*carried, _RETRY_ENVELOPE),
                                  suggestions=suggestions)
                continue
            except LlmContentError:
                if last_round:
                    raise
                attempt = Attempt(target=target, issues=(*carried, _RETRY_JSON),
                                  suggestions=suggestions)
                continue
            violations = self.validators.check(record, target)
            hard = blocking(violations)
            if not hard:
                return target, violations, False
            attempt = Attempt(target=target, issues=carried + tuple(v.message for v in hard),
                              suggestions=suggestions)
        return target, violations, False

    def process(self, record: Record) -> Outcome:
        """Run one record through production, mechanical checks, review and revision.

        Returns an Outcome only for verdicts about *this* record. Infrastructure failures
        propagate as ``LlmError`` instead, because the caller journals every returned Outcome as a
        completed result: turning "the server was down" into a terminal REJECTED would burn the
        record on a run that still exits 0.
        """
        if record.status is Status.SKIPPED or not record.source.strip():
            return Outcome(record=record, status=Status.SKIPPED, output=None)

        feedback: Attempt | None = None
        all_reviews: list[Review] = []
        try:
            for round_index in range(self.panel.max_revisions + 1):
                target, violations, from_repair = self._generate(record, feedback)
                reject, keep = partition_on_exhaustion(violations, self.ruleset)
                if reject:
                    return Outcome(record=record, status=Status.REJECTED, output=target,
                                   violations=tuple(violations), reviews=tuple(all_reviews),
                                   rounds=round_index + 1,
                                   error="mechanical rules still violated after repairs")
                if keep:
                    # Imperfect but the only alternative is showing nothing: keep and flag.
                    return Outcome(record=record, status=Status.PRODUCED, output=target,
                                   violations=tuple(violations), reviews=tuple(all_reviews),
                                   rounds=round_index + 1,
                                   error="kept despite a flagged rule (the alternative is showing "
                                         "nothing)")
                reviews = self.review_panel(record, target)
                all_reviews.extend(reviews)
                objections = [r for r in reviews if not r.acceptable]
                if not objections:
                    return self._settle(record, target, violations, all_reviews, round_index,
                                        reviews, from_repair)
                if round_index == self.panel.max_revisions:
                    return Outcome(record=record, status=Status.PRODUCED, output=target,
                                   violations=tuple(violations), reviews=tuple(all_reviews),
                                   rounds=round_index + 1,
                                   error="review objections unresolved within budget")
                feedback = Attempt(
                    target=target,
                    issues=tuple(f"{r.role}: {issue}" for r in objections for issue in r.issues),
                    suggestions=tuple(r.improved for r in objections if r.improved))
        except LlmRefusalError as exc:
            return Outcome(record=record, status=Status.REJECTED, output=None,
                           reviews=tuple(all_reviews), error=f"model refused: {exc}")
        except LlmContentError as exc:
            return Outcome(record=record, status=Status.REJECTED, output=None,
                           reviews=tuple(all_reviews), error=str(exc))
        raise AssertionError("unreachable: revision loop always returns")

    def _settle(self, record: Record, target: str, violations: list[Violation],
                all_reviews: list[Review], round_index: int, reviews: list[Review],
                from_repair: bool) -> Outcome:
        """The accepted-by-panel branch: verified, or kept-but-flagged when the panel did not fully
        run or the output was only recovered from a truncated envelope."""
        common: dict[str, Any] = {
            "record": record, "output": target, "violations": tuple(violations),
            "reviews": tuple(all_reviews), "rounds": round_index + 1}
        unevaluated = [r for r in reviews if r.error]
        if unevaluated:
            return Outcome(status=Status.PRODUCED, error="review incomplete: "
                           + "; ".join(f"{r.role}: {r.error}" for r in unevaluated), **common)
        if from_repair:
            return Outcome(status=Status.PRODUCED,
                           error="output recovered from a truncated JSON envelope; kept but not "
                                 "independently confirmed, flagged for review", **common)
        return Outcome(status=Status.VERIFIED, **common)


_RETRY_JSON = ("your previous reply could not be read as valid JSON -- reply with the requested "
               "object as one valid JSON object and nothing else.")
_RETRY_ENVELOPE = ("your previous reply's JSON envelope was cut off before the closing brace -- "
                   "repeat the FULL output as one complete, valid JSON object, nothing after it.")


def learn_memory(memory: OutputMemory | None, outcome: Outcome) -> None:
    """Record an accepted output into memory, so a later record can reuse it as context. A no-op
    unless memory is in use and the output is injectable."""
    if memory is not None and outcome.output and outcome.status.is_injectable:
        memory.record(outcome.record.source, outcome.output)
