"""Rule violations: severity, the blocking flag, and the filter."""
from __future__ import annotations

from ragkit.core.rules import Severity, Violation, blocking


def test_error_violation_blocks() -> None:
    assert Violation("r", Severity.ERROR, "m").blocking is True


def test_warning_violation_does_not_block() -> None:
    assert Violation("r", Severity.WARNING, "m").blocking is False


def test_blocking_filters_to_errors_preserving_order() -> None:
    violations = [
        Violation("a", Severity.WARNING, "m"),
        Violation("b", Severity.ERROR, "m"),
        Violation("c", Severity.ERROR, "m"),
    ]
    assert [v.rule_id for v in blocking(violations)] == ["b", "c"]


def test_severity_value_is_the_member() -> None:
    assert Severity("error") is Severity.ERROR
