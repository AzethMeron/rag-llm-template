"""Request-shaping for one family of local model, and the small registry that selects it.

"The same request" does not mean "the same bytes on the wire" across models served over an
OpenAI-compatible endpoint. The sharpest difference is structured output: Qwen accepts a strict
``json_schema`` grammar, while some builds (Bielik, EuroLLM) reject it ("Failed to initialize
samplers") but honour ``json_object`` with the shape described in the prompt. A :class:`Backend`
is one object per family that shapes the request accordingly; the client never changes when the
model does.

**Why a dedicated registry here, not the general** :class:`~ragkit.core.registry.Registry`.
Backends are stateless singletons carrying model *hints*, and selection has a bespoke mode —
``auto`` picks the backend whose hints match the served model id. The general registry builds a
fresh instance per call from TOML options and has no "match by model" step, so it is the wrong
tool for this one port. This is the codec-registry pattern the general registry's own docstring
names as the bounded exception: written only at import, collision-refusing, holding no per-run
state. User extensibility is preserved — :func:`register_backend` is public, and a config may
also name a custom backend by dotted path.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from ragkit.core.errors import RagkitError
from ragkit.core.ports import Message, StructuredRequest

DEFAULT_BACKEND = "generic"
"""Backend used when a caller names none: grammar-constrained decoding, correct for Qwen and
most local servers. A model that rejects it selects another by name."""


class BackendError(RagkitError):
    """A backend was requested that does not exist, or two were registered under one name."""


class BaseBackend(ABC):
    """Request-shaping for one model family. Stateless and shared: one instance is registered
    under its name (and aliases) and reused for every request."""

    def __init__(self, name: str, *, aliases: Sequence[str] = (),
                 model_hints: Sequence[str] = ()) -> None:
        if not name.strip():
            raise BackendError("a backend needs a non-empty name")
        self.name = name
        self.aliases = tuple(aliases)
        # Substrings of a served model id that point to this backend, used by `auto` selection
        # and by the diagnostic hint when a structured request is rejected.
        self.model_hints = tuple(hint.lower() for hint in model_hints)

    @abstractmethod
    def structured_request(self, messages: Sequence[Message],
                           schema: dict[str, Any]) -> StructuredRequest:
        """Shape a request for output conforming to ``schema`` for this model."""

    def matches_model(self, model: str) -> bool:
        lowered = model.lower()
        return any(hint in lowered for hint in self.model_hints)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r}>"


class SchemaBackend(BaseBackend):
    """Grammar-constrained decoding: ``response_format`` = ``json_schema``, strict — the
    strongest guarantee and the default. Messages are sent unchanged."""

    def structured_request(self, messages: Sequence[Message],
                           schema: dict[str, Any]) -> StructuredRequest:
        return StructuredRequest(
            messages=tuple(messages),
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": schema},
            })


class JsonObjectBackend(BaseBackend):
    """Valid-JSON-only mode: ``response_format`` = ``json_object``, with the schema's shape
    described in prose and appended to the prompt. The reply is still validated against the
    schema by the client, so this is a weaker *constraint*, not a weaker *check*."""

    def structured_request(self, messages: Sequence[Message],
                           schema: dict[str, Any]) -> StructuredRequest:
        return StructuredRequest(
            messages=_append_json_instruction(messages, schema),
            response_format={"type": "json_object"})


def _append_json_instruction(messages: Sequence[Message],
                             schema: dict[str, Any]) -> tuple[Message, ...]:
    """Append an instruction to answer as a JSON object with the schema's fields.

    A minimal, literal description — field names and their JSON types — rather than the full
    schema, because the model needs to know the shape to produce, not to reason about a
    specification. Appended to the final user turn so it is the last thing read.
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields = []
    for name, spec in properties.items():
        kind = spec.get("type", "string")
        # A union type is a list in the schema; render it as "string|null" rather than letting a
        # Python list repr leak into the prompt the model reads.
        rendered_kind = "|".join(kind) if isinstance(kind, list) else kind
        marker = "" if name in required else " (optional)"
        fields.append(f'"{name}": <{rendered_kind}>{marker}')
    shape = "{" + ", ".join(fields) + "}"
    instruction = (
        "Respond with a single JSON object and nothing else, of the form "
        f"{shape}. Put it all on one line, escape any double quote inside a value as "
        '\\", use no literal line breaks inside a value, and close every string and '
        "brace. For an empty value use [] for a list and \"\" for a string; use null only "
        "for a field whose type is shown as |null. Output only the JSON.")

    result = list(messages)
    for index in range(len(result) - 1, -1, -1):
        if result[index].role == "user":
            result[index] = Message("user", f"{result[index].content}\n\n{instruction}")
            return tuple(result)
    result.append(Message("user", instruction))
    return tuple(result)


# --- registry -----------------------------------------------------------------

# Written only at import (below), effectively immutable thereafter; a name collision with a
# *different* backend is refused rather than silently overwritten. See the module docstring for
# why this bespoke registry, not the general component registry.
_REGISTRY: dict[str, BaseBackend] = {}


def register_backend(backend: BaseBackend) -> BaseBackend:
    """Register ``backend`` under its name and every alias. Returns it, for convenience.

    A name collision with a *different* backend is an error, not a silent overwrite: two models
    answering to one name would make selection depend on import order. Re-registering the same
    instance is a no-op.
    """
    for key in (backend.name, *backend.aliases):
        lowered = key.lower()
        existing = _REGISTRY.get(lowered)
        if existing is not None and existing is not backend:
            raise BackendError(
                f"backend name {key!r} is already registered to {existing!r}; cannot also "
                f"register {backend!r}")
        _REGISTRY[lowered] = backend
    return backend


def get_backend(name: str) -> BaseBackend:
    """Look up a backend by name or alias, listing what is available when it is missing."""
    backend = _REGISTRY.get(name.lower())
    if backend is None:
        raise BackendError(
            f"unknown backend {name!r}; available backends are {', '.join(available_backends())}")
    return backend


def available_backends() -> list[str]:
    """The distinct registered backend names, sorted."""
    return sorted({backend.name for backend in _REGISTRY.values()})


def suggest_backend(model: str) -> str | None:
    """The backend whose model hints match ``model``, or ``None`` if none do."""
    seen: dict[int, BaseBackend] = {}
    for backend in _REGISTRY.values():
        seen.setdefault(id(backend), backend)
    for backend in seen.values():
        if backend.matches_model(model):
            return backend.name
    return None


def resolve_backend(name: str, model: str) -> BaseBackend:
    """Select a backend by ``name``, or by the served ``model`` id when ``name`` is ``"auto"``.

    ``auto`` is what hardens a model switch: it picks the backend whose hints match the model,
    falling back to the default when none do. The caller is expected to report which backend was
    chosen, so the selection is never silent.
    """
    if name == "auto":
        guess = suggest_backend(model)
        return get_backend(guess) if guess is not None else get_backend(DEFAULT_BACKEND)
    return get_backend(name)


# Built-in profiles. `generic` is grammar-constrained (Qwen and most llama.cpp/vLLM builds);
# the json_object families describe the shape in the prompt instead. A user adds a family by
# calling register_backend from their own module (named by dotted path in config) -- no edit
# here.
register_backend(SchemaBackend("generic", aliases=("openai", "json_schema"),
                               model_hints=("qwen",)))
register_backend(JsonObjectBackend("bielik", model_hints=("bielik",)))
register_backend(JsonObjectBackend("eurollm", model_hints=("eurollm", "euro-llm")))
register_backend(JsonObjectBackend("gemma", model_hints=("gemma",)))
