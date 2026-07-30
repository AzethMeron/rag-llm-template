"""The citation-grounding validator — the failure surface a cite-from-memory task must guard.

A legal answer that cites passages is only trustworthy if the citations are *real*. This validator
makes the "cite what you retrieved, or abstain" rule mechanical, decided in code before an answer is
accepted:

* **A substantive answer must cite.** Unless the answer abstains (contains the abstention marker,
  ``"brak podstaw"`` by default), the ``citations`` field must be a non-empty list of passage ids;
  an answer that asserts a legal position while citing nothing is refused.
* **Every citation must be grounded.** Each cited id is checked against the passages actually
  retrieved for this question (re-retrieved here through the same deterministic retriever the
  context block used, at a depth ``>=`` the block's, so anything the model was shown is in scope). A
  citation to an id that is in no retrieved passage is a fabricated citation — a blocking violation.
* **Abstention passes.** ``"brak podstaw"`` with no citations is a valid answer: the question's
  passages did not support one, and the model said so.

If no retriever is wired, grounding cannot be verified; rather than pass silently the validator
blocks and says so — a cite-from-memory recipe with no memory to check against is a
misconfiguration, not a green light.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ragkit.core.ports import Retriever
from ragkit.core.records import Record
from ragkit.core.rules import Severity, Violation
from ragkit.harness.capture import capture_retrieved


def _error(message: str) -> Violation:
    return Violation("ungrounded_citation", Severity.ERROR, message)


class CitationGroundingValidator:
    """Refuses a legal answer that is uncited (while substantive) or cites a passage it never
    retrieved. Abstention (``"brak podstaw"``) with no citations is accepted."""

    CONFIG_KEYS = frozenset({"answer_field", "citations_field", "abstain_marker", "k", "min_score"})

    def __init__(self, *, answer_field: str = "answer", citations_field: str = "citations",
                 abstain_marker: str = "brak podstaw", k: int = 20,
                 min_score: float = 0.0) -> None:
        if not answer_field.strip():
            raise ValueError("CitationGroundingValidator needs a non-empty 'answer_field'")
        if not citations_field.strip():
            raise ValueError("CitationGroundingValidator needs a non-empty 'citations_field'")
        if not abstain_marker.strip():
            raise ValueError("CitationGroundingValidator needs a non-empty 'abstain_marker'")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self._answer_field = answer_field
        self._citations_field = citations_field
        self._abstain_marker = abstain_marker.strip().lower()
        self._k = k
        self._min_score = min_score

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> CitationGroundingValidator:
        return cls(
            answer_field=str(options.get("answer_field", "answer")),
            citations_field=str(options.get("citations_field", "citations")),
            abstain_marker=str(options.get("abstain_marker", "brak podstaw")),
            k=int(options.get("k", 20)), min_score=float(options.get("min_score", 0.0)))

    def validate(self, record: Record, output: str,
                 context: Mapping[str, Any]) -> list[Violation]:
        try:
            answer = json.loads(output)
        except json.JSONDecodeError as exc:
            return [_error(f"the answer is not valid JSON: {exc}")]
        if not isinstance(answer, dict):
            return [_error(f"the answer must be a JSON object, got {type(answer).__name__}")]

        text = answer.get(self._answer_field)
        if not isinstance(text, str) or not text.strip():
            return [_error(f"field {self._answer_field!r} must be a non-empty answer")]
        if self._abstain_marker in text.lower():
            # Abstention: the passages did not support an answer and the model said so. No
            # citations are required (or checked) for an honest 'brak podstaw'.
            return []

        return self._check_citations(record, answer, context)

    def _check_citations(self, record: Record, answer: Mapping[str, Any],
                         context: Mapping[str, Any]) -> list[Violation]:
        raw = answer.get(self._citations_field)
        citations = [str(c) for c in raw if isinstance(c, (str, int))] if isinstance(raw, list) \
            else []
        if not citations:
            return [_error(f"field {self._citations_field!r} must cite at least one retrieved "
                           f"passage id; a substantive answer with no citation is refused (answer "
                           f"'brak podstaw' if nothing is relevant)")]

        retriever = context.get("retriever")
        if not isinstance(retriever, Retriever):
            return [_error("grounding cannot be verified: no retriever (legal-passage memory) is "
                           "wired into the run, so a cited answer cannot be accepted")]
        hits = retriever.retrieve(record.source, k=self._k, min_score=self._min_score)
        capture_retrieved(context, hits)
        retrieved_ids = {hit.chunk_id for hit in hits}

        violations: list[Violation] = []
        for cited in citations:
            if cited not in retrieved_ids:
                violations.append(_error(
                    f"citation {cited!r} refers to no passage retrieved for this question; it "
                    f"looks invented. Cite only ids of the passages shown"))
        return violations
