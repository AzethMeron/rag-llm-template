"""Field-value checks for a filled form — the mechanical guard that a produced form is not just
*shaped* right but *filled* right.

The output-schema's JSON shape check (in the client) already guarantees the reply is an object with
the declared fields at the declared top-level JSON types. This validator adds the value-level
contract the shape check cannot express: a required field is actually filled (not left null or
blank), a number is in range, a categorical is one of the allowed values. Each is decided in code
from the field's declared constraints, never from the model's opinion, and a failure is a blocking
:class:`~ragkit.core.rules.Violation` so the record is repaired or rejected rather than shipped with
a bad value.

Constraints per field (all optional except ``name`` and ``type``): ``nonempty`` (a string must have
non-whitespace content), ``min``/``max`` (inclusive numeric bounds), ``min_exclusive`` (an
exclusive lower bound, e.g. a price must be > 0), and ``enum`` (the value must be one of a fixed
set, compared case-insensitively for strings).
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ragkit.core.jsonshape import json_type_matches
from ragkit.core.records import Record
from ragkit.core.rules import Severity, Violation

_JSON_TYPES = frozenset({"string", "integer", "number", "boolean"})


def _error(message: str) -> Violation:
    return Violation("form_field", Severity.ERROR, message)


@dataclass(frozen=True, slots=True)
class FieldRule:
    """The value contract for one form field."""

    name: str
    type: str
    nonempty: bool = False
    minimum: float | None = None
    maximum: float | None = None
    min_exclusive: float | None = None
    enum: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.type not in _JSON_TYPES:
            raise ValueError(f"field {self.name!r}: type must be one of {sorted(_JSON_TYPES)}, "
                             f"got {self.type!r}")

    def check(self, value: Any) -> list[Violation]:
        if value is None:
            return [_error(f"field {self.name!r} was not filled (null)")]
        if not json_type_matches(value, self.type):
            return [_error(f"field {self.name!r} should be {self.type}, got "
                           f"{type(value).__name__} {value!r}")]
        violations: list[Violation] = []
        if self.nonempty and isinstance(value, str) and not value.strip():
            violations.append(_error(f"field {self.name!r} is blank"))
        violations.extend(self._range_checks(value))
        if self.enum and not _in_enum(value, self.enum):
            violations.append(_error(f"field {self.name!r} value {value!r} is not one of the "
                                     f"allowed values {list(self.enum)}"))
        return violations

    def _range_checks(self, value: Any) -> list[Violation]:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return []
        checks = (
            (self.minimum is not None and value < self.minimum,
             f"field {self.name!r} value {value} is below the minimum {self.minimum}"),
            (self.maximum is not None and value > self.maximum,
             f"field {self.name!r} value {value} is above the maximum {self.maximum}"),
            (self.min_exclusive is not None and value <= self.min_exclusive,
             f"field {self.name!r} value {value} must be greater than {self.min_exclusive}"),
        )
        return [_error(message) for failed, message in checks if failed]


def _in_enum(value: Any, allowed: Sequence[str]) -> bool:
    if isinstance(value, str):
        lowered = {item.lower() for item in allowed}
        return value.lower() in lowered
    return str(value) in allowed


class FieldTypesValidator:
    """Checks a filled form's field values against their declared per-field constraints."""

    CONFIG_KEYS = frozenset({"field"})

    def __init__(self, fields: Sequence[FieldRule]) -> None:
        if not fields:
            raise ValueError("FieldTypesValidator needs at least one field rule")
        self._fields = tuple(fields)

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> FieldTypesValidator:
        raw = options.get("field")
        if not isinstance(raw, list) or not raw:
            raise ValueError("FieldTypesValidator needs a non-empty 'field' array")
        rules = []
        for entry in raw:
            if not isinstance(entry, Mapping) or "name" not in entry:
                raise ValueError("each field rule needs at least a 'name'")
            rules.append(FieldRule(
                name=str(entry["name"]), type=str(entry.get("type", "string")),
                nonempty=bool(entry.get("nonempty", False)),
                minimum=_opt_number(entry, "min"), maximum=_opt_number(entry, "max"),
                min_exclusive=_opt_number(entry, "min_exclusive"),
                enum=tuple(str(item) for item in entry.get("enum", ()))))
        return cls(rules)

    def validate(self, record: Record, output: str,  # noqa: ARG002  -- record unused; port shape
                 context: Mapping[str, Any]) -> list[Violation]:  # noqa: ARG002  -- context unused
        try:
            form = json.loads(output)
        except json.JSONDecodeError as exc:
            return [_error(f"the filled form is not valid JSON: {exc}")]
        if not isinstance(form, dict):
            return [_error(f"the filled form must be a JSON object, got {type(form).__name__}")]
        violations: list[Violation] = []
        for rule in self._fields:
            violations.extend(rule.check(form.get(rule.name)))
        return violations


def _opt_number(entry: Mapping[str, Any], key: str) -> float | None:
    if key not in entry:
        return None
    value = entry[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"field constraint {key!r} must be a number, got {value!r}")
    return float(value)
