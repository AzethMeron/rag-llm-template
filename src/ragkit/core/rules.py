"""The shared vocabulary of a rule violation: :class:`Severity`, :class:`Violation`.

A validator (see the ``Validator`` port in :mod:`ragkit.core.ports`) inspects a produced
output and returns zero or more :class:`Violation` objects. Whether a violation *blocks*
acceptance is carried by its :class:`Severity`, so the harness can treat "must be fixed" and
"worth noting" differently without re-deciding per rule. This lives in the core because both
the validators (which produce violations) and the harness (which acts on them) depend on the
same shape, and neither should own it.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class Severity(str, Enum):
    ERROR = "error"
    """The output must be rejected and re-produced (or, past the budget, handled per the
    rule's exhaustion policy)."""
    WARNING = "warning"
    """Recorded for review; does not block acceptance."""


@dataclass(frozen=True, slots=True)
class Violation:
    """One rule violation: which rule, how severe, and a message the model can act on."""

    rule_id: str
    severity: Severity
    message: str

    @property
    def blocking(self) -> bool:
        return self.severity is Severity.ERROR


def blocking(violations: Sequence[Violation]) -> list[Violation]:
    """Just the blocking (error-severity) violations, preserving order.

    Lives here, beside :class:`Violation`, and is imported by
    :mod:`ragkit.harness.validators` -- which carried a byte-identical second copy. Two
    definitions of "which violations stop a run" is one more than the rule can survive."""
    return [violation for violation in violations if violation.blocking]
