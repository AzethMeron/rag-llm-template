"""The grounding guards for the biomedical evidence recipe — the failure surface a decide-from-
abstracts task must make mechanical, in code, before an answer is accepted.

Two independent validators, both defence-in-depth (the panel judges the same things in prose; these
decide them in code so a bad answer cannot slip through a lenient reviewer):

* :class:`GroundedEvidenceValidator` — every quote in the ``evidence`` array must be *real*: it must
  appear verbatim (after whitespace/quote normalisation) in the abstracts actually retrieved for
  this question. A quote that appears in no retrieved abstract is a hallucinated citation and blocks.
  A non-abstaining decision (anything other than ``unsupported``) with no evidence at all is refused:
  a confident yes/no/maybe with nothing to stand on is exactly what the "abstain unless there is
  positive evidence" rule forbids. If no retriever is wired, grounding cannot be verified — the
  validator blocks and says so rather than passing silently (a decide-from-memory recipe with no
  memory to check against is a misconfiguration, not a green light).

* :class:`DecisionEnumValidator` — the ``decision`` must be one of the allowed labels
  (``yes``/``no``/``maybe``/``unsupported``), so a downstream eval and any automation act on a known
  category, never free text.

Both parse the model's JSON defensively: malformed output is a blocking violation, never a crash.
Quote matching reuses the predictive-maintenance normalisation (lower-case, whitespace-collapse, and
strip surrounding quotation marks) because a real model wraps its citation in quotes even when the
text inside is verbatim from an abstract, and the abstract itself is unquoted.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ragkit.core.ports import Retriever
from ragkit.core.records import Record
from ragkit.core.rules import Severity, Violation

_WHITESPACE = re.compile(r"\s+")
# Quotation marks (straight and curly, single and double) are stripped before the grounding
# substring check: a real model routinely wraps its citation in quotes ("...") even when the text
# inside is verbatim from an abstract, and the abstract passage itself is unquoted. Matching would
# then fail on the surrounding quote characters alone -- rejecting a correctly-grounded citation.
_QUOTES = re.compile(r"[\"'“”‘’«»`]")

_ALLOWED_DECISIONS = ("yes", "no", "maybe", "unsupported")


def _normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", _QUOTES.sub("", text)).strip().lower()


def _violation(rule_id: str, message: str) -> Violation:
    return Violation(rule_id, Severity.ERROR, message)


def _parse(output: str, rule_id: str) -> tuple[Mapping[str, Any] | None, Violation | None]:
    """Parse the model output as a JSON object. Returns ``(decision, None)`` on success, or
    ``(None, blocking_violation)`` describing why it could not be read — never raises."""
    try:
        decision = json.loads(output)
    except json.JSONDecodeError as exc:
        return None, _violation(rule_id, f"the answer is not valid JSON: {exc}")
    if not isinstance(decision, dict):
        return None, _violation(rule_id, f"the answer must be a JSON object, got "
                                f"{type(decision).__name__}")
    return decision, None


class GroundedEvidenceValidator:
    """Refuses a biomedical answer whose evidence quotes are missing (for a non-abstaining decision)
    or not found in the abstracts actually retrieved for the question."""

    CONFIG_KEYS = frozenset({"decision_field", "evidence_field", "unsupported_value", "k",
                             "min_score", "min_quote_len"})

    def __init__(self, *, decision_field: str = "decision", evidence_field: str = "evidence",
                 unsupported_value: str = "unsupported", k: int = 20, min_score: float = 0.0,
                 min_quote_len: int = 12) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if min_quote_len < 1:
            raise ValueError(f"min_quote_len must be >= 1, got {min_quote_len}")
        self._decision_field = decision_field
        self._evidence_field = evidence_field
        self._unsupported_value = unsupported_value.strip().lower()
        self._k = k
        self._min_score = min_score
        self._min_quote_len = min_quote_len

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> GroundedEvidenceValidator:
        return cls(
            decision_field=str(options.get("decision_field", "decision")),
            evidence_field=str(options.get("evidence_field", "evidence")),
            unsupported_value=str(options.get("unsupported_value", "unsupported")),
            k=int(options.get("k", 20)), min_score=float(options.get("min_score", 0.0)),
            min_quote_len=int(options.get("min_quote_len", 12)))

    def validate(self, record: Record, output: str,
                 context: Mapping[str, Any]) -> list[Violation]:
        decision, error = _parse(output, "ungrounded_evidence")
        if decision is None:
            return [error]  # type: ignore[list-item]  # error is set when decision is None

        label = decision.get(self._decision_field)
        abstaining = isinstance(label, str) and label.strip().lower() == self._unsupported_value

        evidence = decision.get(self._evidence_field)
        quotes = [q for q in evidence if isinstance(q, str)] if isinstance(evidence, list) else []

        # Grounding can only be verified against the memory the prompt drew from. Without it, block
        # rather than accept an unverifiable citation.
        retriever = context.get("retriever")
        if not isinstance(retriever, Retriever):
            return [_violation("ungrounded_evidence",
                               "grounding cannot be verified: no retriever (abstracts memory) is "
                               "wired into the run, so a cited answer cannot be accepted")]

        if not quotes:
            if abstaining:
                return []  # abstaining with no evidence is exactly the allowed behaviour
            return [_violation("ungrounded_evidence",
                               f"field {self._evidence_field!r} must cite at least one abstract "
                               f"quote for a {label!r} decision; answer 'unsupported' instead of "
                               f"deciding with no evidence")]

        hits = retriever.retrieve(record.source, k=self._k, min_score=self._min_score)
        corpus = _normalise(" ".join(hit.text for hit in hits))

        violations: list[Violation] = []
        for quote in quotes:
            normalised = _normalise(quote)
            if len(normalised) < self._min_quote_len:
                violations.append(_violation(
                    "ungrounded_evidence",
                    f"citation {quote!r} is too short to verify (< {self._min_quote_len} chars); "
                    f"quote a specific phrase from the abstracts"))
            elif normalised not in corpus:
                violations.append(_violation(
                    "ungrounded_evidence",
                    f"citation {quote!r} does not appear in any retrieved abstract; it looks "
                    f"invented. Quote only text from the abstracts shown"))
        return violations


class DecisionEnumValidator:
    """Refuses an answer whose ``decision`` is not one of the allowed labels (or whose JSON cannot
    be read at all)."""

    CONFIG_KEYS = frozenset({"field", "allowed"})

    def __init__(self, *, field: str = "decision",
                 allowed: Sequence[str] = _ALLOWED_DECISIONS) -> None:
        if not allowed:
            raise ValueError("DecisionEnumValidator needs a non-empty 'allowed' set")
        self._field = field
        self._allowed = {a.strip().lower() for a in allowed}
        self._allowed_display = tuple(allowed)

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> DecisionEnumValidator:
        allowed = options.get("allowed", list(_ALLOWED_DECISIONS))
        if not isinstance(allowed, list) or not allowed:
            raise ValueError("DecisionEnumValidator needs a non-empty 'allowed' array")
        return cls(field=str(options.get("field", "decision")), allowed=[str(a) for a in allowed])

    def validate(self, record: Record, output: str,
                 context: Mapping[str, Any]) -> list[Violation]:
        decision, error = _parse(output, "invalid_decision")
        if decision is None:
            return [error]  # type: ignore[list-item]  # error is set when decision is None
        value = decision.get(self._field)
        if not isinstance(value, str) or value.strip().lower() not in self._allowed:
            return [_violation("invalid_decision",
                               f"field {self._field!r} must be one of "
                               f"{list(self._allowed_display)}, got {value!r}")]
        return []
