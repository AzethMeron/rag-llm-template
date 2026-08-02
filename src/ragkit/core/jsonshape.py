"""JSON value-type checking, shared by the client's schema-shape validation and any validator that
must check a declared field type against a value.

One home for the subtle rule that ``bool`` is **not** an ``integer`` or ``number`` even though
Python makes ``bool`` a subclass of ``int`` — reimplementing that in two places is how a validator
and the client's shape check drift apart. Stdlib-only, so it stays in the core.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .errors import RagkitError

JSON_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


class JsonShapeError(RagkitError):
    """A declared JSON type is not one this module knows how to check."""


def json_type_matches(value: Any, declared: Any) -> bool:
    """Whether ``value`` has the JSON type ``declared`` names — a single type name, or a union
    list of them.

    An unrecognised name raises rather than passing. Returning ``True`` for it, as this used to,
    meant a schema author's typo (``"boolena"``, ``"str"``) silently switched that field's check
    off entirely — the check would report the field fine no matter what came back, which is worse
    than not having it. Every JSON type has a name here, so an unknown one is a mistake, not an
    advanced schema this simply does not model.
    """
    if isinstance(declared, list):
        return any(json_type_matches(value, item) for item in declared)
    check = JSON_TYPE_CHECKS.get(declared) if isinstance(declared, str) else None
    if check is None:
        raise JsonShapeError(
            f"unknown JSON type {declared!r} in a schema; known types are "
            f"{sorted(JSON_TYPE_CHECKS)}")
    return check(value)
