"""The grounded-decision validator — the failure surface a decide-from-memory task must guard.

A maintenance decision that cites the manuals is only trustworthy if the citations are *real*. This
validator makes the "abstain unless there is positive evidence" rule mechanical, defence-in-depth,
decided in code before a decision is accepted:

* **Every decision must cite evidence.** The ``evidence`` field must be a non-empty list of quotes;
  a confident diagnosis with nothing to stand on is refused (abstain-or-cite).
* **Every citation must be grounded.** Each quote is checked against the manual passages actually
  retrieved for this record (re-retrieved here through the same deterministic retriever the context
  block used, at a depth ≥ the block's, so anything the model was shown is in scope). A quote that
  appears in no retrieved passage is a hallucinated citation — a blocking violation. Matching is on
  normalised text (lower-cased, whitespace-collapsed) and a quote must be at least ``min_quote_len``
  characters, so a trivially short fragment cannot "match" everything.
* **Severity must be one of the allowed values**, so a decision's urgency is a known category the
  downstream eval and any automation can act on, not free text.

If no retriever is wired, grounding cannot be verified; rather than pass silently, the validator
blocks and says so — a decide-from-memory recipe with no memory to check against is a
misconfiguration, not a green light.
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
# inside is verbatim from a manual, and the manual passage itself is unquoted. Matching would then
# fail on the surrounding quote characters alone -- rejecting a correctly-grounded citation. Found
# by running a live model; a mock returning clean quotes never exercised it.
_QUOTES = re.compile(r"[\"'“”‘’«»`]")


def _normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", _QUOTES.sub("", text)).strip().lower()


def _error(message: str) -> Violation:
    return Violation("ungrounded_decision", Severity.ERROR, message)


class GroundedDecisionValidator:
    """Refuses a maintenance decision that is uncited, ungrounded, or wrongly categorised."""

    CONFIG_KEYS = frozenset({"diagnosis_field", "severity_field", "evidence_field", "severities",
                             "k", "min_score", "min_quote_len"})

    def __init__(self, *, severities: Sequence[str], diagnosis_field: str = "diagnosis",
                 severity_field: str = "severity", evidence_field: str = "evidence",
                 k: int = 20, min_score: float = 0.0, min_quote_len: int = 12) -> None:
        if not severities:
            raise ValueError("GroundedDecisionValidator needs a non-empty 'severities' set")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if min_quote_len < 1:
            raise ValueError(f"min_quote_len must be >= 1, got {min_quote_len}")
        self._diagnosis_field = diagnosis_field
        self._severity_field = severity_field
        self._evidence_field = evidence_field
        self._severities = {s.lower() for s in severities}
        self._severities_display = tuple(severities)
        self._k = k
        self._min_score = min_score
        self._min_quote_len = min_quote_len

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> GroundedDecisionValidator:
        severities = options.get("severities")
        if not isinstance(severities, list) or not severities:
            raise ValueError("GroundedDecisionValidator needs a non-empty 'severities' array")
        return cls(
            severities=[str(s) for s in severities],
            diagnosis_field=str(options.get("diagnosis_field", "diagnosis")),
            severity_field=str(options.get("severity_field", "severity")),
            evidence_field=str(options.get("evidence_field", "evidence")),
            k=int(options.get("k", 20)), min_score=float(options.get("min_score", 0.0)),
            min_quote_len=int(options.get("min_quote_len", 12)))

    def validate(self, record: Record, output: str,
                 context: Mapping[str, Any]) -> list[Violation]:
        try:
            decision = json.loads(output)
        except json.JSONDecodeError as exc:
            return [_error(f"the decision is not valid JSON: {exc}")]
        if not isinstance(decision, dict):
            return [_error(f"the decision must be a JSON object, got {type(decision).__name__}")]

        violations: list[Violation] = []
        violations.extend(self._check_diagnosis(decision))
        violations.extend(self._check_severity(decision))
        violations.extend(self._check_evidence(record, decision, context))
        return violations

    def _check_diagnosis(self, decision: Mapping[str, Any]) -> list[Violation]:
        diagnosis = decision.get(self._diagnosis_field)
        if not isinstance(diagnosis, str) or not diagnosis.strip():
            return [_error(f"field {self._diagnosis_field!r} must be a non-empty diagnosis")]
        return []

    def _check_severity(self, decision: Mapping[str, Any]) -> list[Violation]:
        severity = decision.get(self._severity_field)
        if not isinstance(severity, str) or severity.strip().lower() not in self._severities:
            return [_error(f"field {self._severity_field!r} must be one of "
                           f"{list(self._severities_display)}, got {severity!r}")]
        return []

    def _check_evidence(self, record: Record, decision: Mapping[str, Any],
                        context: Mapping[str, Any]) -> list[Violation]:
        evidence = decision.get(self._evidence_field)
        quotes = [q for q in evidence if isinstance(q, str)] if isinstance(evidence, list) else []
        if not quotes:
            return [_error(f"field {self._evidence_field!r} must cite at least one manual quote; a "
                           f"decision with no supporting evidence is refused")]

        retriever = context.get("retriever")
        if not isinstance(retriever, Retriever):
            return [_error("grounding cannot be verified: no retriever (manuals memory) is wired "
                           "into the run, so a cited decision cannot be accepted")]
        hits = retriever.retrieve(record.source, k=self._k, min_score=self._min_score)
        corpus = _normalise(" ".join(hit.text for hit in hits))

        violations: list[Violation] = []
        for quote in quotes:
            normalised = _normalise(quote)
            if len(normalised) < self._min_quote_len:
                violations.append(_error(
                    f"citation {quote!r} is too short to verify (< {self._min_quote_len} chars); "
                    f"cite a specific phrase from the manuals"))
            elif normalised not in corpus:
                violations.append(_error(
                    f"citation {quote!r} does not appear in any retrieved manual passage; it looks "
                    f"invented. Cite only text from the manuals shown"))
        return violations
