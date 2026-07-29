"""The structured error base: diagnostic context in the message, and None dropped from it."""
from __future__ import annotations

from ragkit.core.errors import RagkitError


def test_reason_only() -> None:
    exc = RagkitError("something went wrong")
    assert str(exc) == "something went wrong"
    assert exc.reason == "something went wrong"
    assert exc.context == {}


def test_context_is_rendered_and_inspectable() -> None:
    exc = RagkitError("bad value", expected=5, actual=9)
    assert "expected=5" in str(exc) and "actual=9" in str(exc)
    assert exc.context["actual"] == 9


def test_none_context_values_are_dropped() -> None:
    exc = RagkitError("missing", path=None)
    assert str(exc) == "missing"
    assert "path" not in exc.context


def test_is_an_exception() -> None:
    assert isinstance(RagkitError("x"), Exception)
