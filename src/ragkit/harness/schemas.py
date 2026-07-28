"""Ready-made :class:`~ragkit.core.ports.OutputSchema` implementations.

An output schema describes what the producer generates (a JSON schema the backend constrains or
prompts for) and how to pull the output string out of the parsed reply. Most tasks produce a
single string field — a translation, a SQL statement — for which :class:`JsonFieldSchema` is
enough; a recipe with a richer shape supplies its own component through the ``OUTPUT_SCHEMAS``
registry.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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
