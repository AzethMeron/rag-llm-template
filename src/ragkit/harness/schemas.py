"""Ready-made :class:`~ragkit.core.ports.OutputSchema` implementations.

An output schema describes what the producer generates (a JSON schema the backend constrains or
prompts for) and how to pull the output string out of the parsed reply. Most tasks produce a
single string field — a translation, a SQL statement — for which :class:`JsonFieldSchema` is
enough; a recipe with a richer shape supplies its own component through the ``OUTPUT_SCHEMAS``
registry.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ragkit.core.config import read_bool, read_string
from ragkit.core.ports import OutputSchema
from ragkit.core.registry import Registry

OUTPUT_SCHEMAS: Registry[OutputSchema] = Registry(
    "output schema", OutputSchema,  # type: ignore[type-abstract]
    entry_point_group="ragkit.output_schemas")
"""Registry for output-schema components. The port is passed to parameterise the registry, which
mypy flags because a Protocol is abstract; the ignore is scoped to exactly that argument."""


class JsonFieldSchema:
    """A one-string-field output: ``{"<field>": "<the output>"}``. The common case."""

    CONFIG_KEYS = frozenset({"field", "description"})

    def __init__(self, field: str = "output", description: str = "") -> None:
        if not field.strip():
            raise ValueError("JsonFieldSchema needs a non-empty field name")
        self.name = field
        self._field = field
        self._description = description

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> JsonFieldSchema:
        return cls(field=str(options.get("field", "output")),
                   description=str(options.get("description", "")))

    def json_schema(self) -> dict[str, Any]:
        prop: dict[str, Any] = {"type": "string"}
        if self._description:
            prop["description"] = self._description
        return {"type": "object", "additionalProperties": False,
                "required": [self._field], "properties": {self._field: prop}}

    def extract(self, reply: Mapping[str, Any]) -> str:
        return str(reply[self._field])


OUTPUT_SCHEMAS.register("json_field", JsonFieldSchema)


_JSON_TYPES = frozenset({"string", "integer", "number", "boolean", "array"})
"""Field types a form may declare. ``array`` is an array of strings (a list of citations, tags,
steps); the scalar types are the obvious ones."""


@dataclass(frozen=True, slots=True)
class FormField:
    """One field of a form to fill: its name, JSON type, an optional prompt description, and
    whether the model must supply it. ``array`` fields are arrays of strings."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a form field needs a non-empty name")
        if self.type not in _JSON_TYPES:
            raise ValueError(
                f"form field {self.name!r}: type must be one of {sorted(_JSON_TYPES)}, got "
                f"{self.type!r}")


class FormSchema:
    """A multi-field form output: ``{field1: ..., field2: ...}``. Each field is declared with a
    name and JSON type, so the producer is asked (or grammar-constrained) to fill exactly those
    fields. :meth:`extract` returns the filled form as a **canonical JSON string** (sorted keys),
    which is what becomes the record's single ``output`` — it round-trips through the journal and
    can be scored field by field by a validator or the fill-accuracy eval. Field *value* checks
    (type conformance, enum membership, ranges) are a validator's job, layered on top."""

    CONFIG_KEYS = frozenset({"name", "fields"})

    def __init__(self, fields: Sequence[FormField], name: str = "form") -> None:
        if not fields:
            raise ValueError("FormSchema needs at least one field")
        seen: set[str] = set()
        for field in fields:
            if field.name in seen:
                raise ValueError(f"FormSchema has a duplicate field {field.name!r}")
            seen.add(field.name)
        self.name = name
        self._fields = tuple(fields)

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> FormSchema:
        raw = options.get("fields")
        if not isinstance(raw, list) or not raw:
            raise ValueError("FormSchema needs a non-empty 'fields' array")
        fields = []
        for entry in raw:
            if not isinstance(entry, Mapping) or "name" not in entry:
                raise ValueError("each form field needs at least a 'name'")
            field_opts = dict(entry)
            fields.append(FormField(
                name=read_string(field_opts, "name", "", label="form field"),
                type=read_string(field_opts, "type", "string", label="form field"),
                description=read_string(field_opts, "description", "", label="form field"),
                required=read_bool(field_opts, "required", True, label="form field")))
        return cls(fields, name=read_string(dict(options), "name", "form", label="form schema"))

    def json_schema(self) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        for field in self._fields:
            prop: dict[str, Any] = {"type": field.type}
            if field.type == "array":
                prop["items"] = {"type": "string"}
            if field.description:
                prop["description"] = field.description
            properties[field.name] = prop
        required = [f.name for f in self._fields if f.required]
        return {"type": "object", "additionalProperties": False,
                "required": required, "properties": properties}

    def extract(self, reply: Mapping[str, Any]) -> str:
        # Only the declared fields, in a stable order, so the stored output is canonical and two
        # equal forms serialise identically (the eval compares these strings' parsed fields).
        form = {f.name: reply.get(f.name) for f in self._fields}
        return json.dumps(form, sort_keys=True, ensure_ascii=False)


OUTPUT_SCHEMAS.register("form", FormSchema)
