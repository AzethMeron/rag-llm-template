"""JSON value-type checking, shared by the client's schema-shape validation and any validator that
must check a declared field type against a value.

One home for the subtle rule that ``bool`` is **not** an ``integer`` or ``number`` even though
Python makes ``bool`` a subclass of ``int`` — reimplementing that in two places is how a validator
and the client's shape check drift apart. Stdlib-only, so it stays in the core.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

JSON_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def json_type_matches(value: Any, declared: Any) -> bool:
    """Whether ``value`` has the JSON type ``declared`` names — a single type name, or a union
    list of them. A type name this does not recognise is accepted, so the check tightens the common
    flat case without reimplementing all of JSON Schema."""
    if isinstance(declared, list):
        return any(json_type_matches(value, item) for item in declared)
    check = JSON_TYPE_CHECKS.get(declared) if isinstance(declared, str) else None
    return check is None or check(value)
