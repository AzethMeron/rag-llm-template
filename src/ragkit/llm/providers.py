"""The inference *providers* — the endpoint kinds the transport can talk to — and the small
registry that selects one.

Every provider this framework supports speaks the same OpenAI-compatible HTTP, so a single
transport (the clients in :mod:`ragkit.llm.client`, :mod:`ragkit.retrieve.embedding` and
:mod:`ragkit.retrieve.rerank`) serves them all. What differs between a llama.cpp router, an Ollama
server and a generic OpenAI-compatible endpoint is not the wire protocol but two things the
transport itself cannot know:

- **which roles the endpoint can fill** — its ``capabilities`` (``chat``/``embedding``/``rerank``).
  Ollama, for instance, serves chat and embeddings but has no ``/v1/rerank``; asking it to rerank
  must fail with a clear message at build time, not a puzzling ``404`` in the middle of a run.
- **which request extras are safe to send** — llama.cpp's ``chat_template_kwargs`` (used to turn a
  reasoning model's chain-of-thought off) is honoured by a ``--jinja`` llama.cpp build but rejected
  outright by a strict OpenAI-compatible server, which ``400``s the whole request on the unknown
  field. That family quirk belongs on the provider, not smeared through the client.

**Why a dedicated registry here, not the general** :class:`~ragkit.core.registry.Registry`. This is
the same codec-registry pattern :mod:`ragkit.llm.backends` uses and the general registry's own
docstring names as its bounded exception: the built-ins are stateless singletons written only at
import, collision-refusing, holding no per-run state. A ``models.toml`` selects one by name;
:func:`register_provider` is public so a third party can add an endpoint kind (named by that name in
config) without editing framework code.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ragkit.core.errors import RagkitError

DEFAULT_PROVIDER = "llamacpp-router"
"""The provider assumed when a caller (or a ``ServerConfig``) names none: a llama.cpp router, which
is what the bundled ``serve_models.sh`` launches and what the defaults are tuned for."""

CAPABILITIES = frozenset({"chat", "embedding", "rerank"})
"""The roles an endpoint may fill. A provider declares the subset it can actually serve; a request
for one it does not is refused rather than attempted."""


class ProviderError(RagkitError):
    """A provider was requested that does not exist, two were registered under one name, or a
    provider was described with an unknown capability."""


class LlmProvider:
    """One endpoint kind: its name, the capabilities it can serve, and its request quirks.

    Stateless and shared — one instance is registered under its name and reused for every request,
    exactly like a :class:`~ragkit.llm.backends.BaseBackend`. Satisfies the
    :class:`~ragkit.core.ports.Provider` port structurally.
    """

    __slots__ = ("_sends_template_kwargs", "capabilities", "name")

    def __init__(self, name: str, *, capabilities: frozenset[str] | set[str],
                 sends_template_kwargs: bool) -> None:
        if not name.strip():
            raise ProviderError("a provider needs a non-empty name")
        unknown = frozenset(capabilities) - CAPABILITIES
        if unknown:
            raise ProviderError(
                f"provider {name!r} declares unknown capabilities {sorted(unknown)}; known "
                f"capabilities are {sorted(CAPABILITIES)}")
        self.name = name
        self.capabilities = frozenset(capabilities)
        self._sends_template_kwargs = sends_template_kwargs

    def supports(self, capability: str) -> bool:
        """Whether this endpoint kind can serve ``capability``. An unknown capability name is a
        programming error (a typo would otherwise read as an unsupported feature), so it raises
        rather than returning ``False``."""
        if capability not in CAPABILITIES:
            raise ProviderError(
                f"unknown capability {capability!r}; known capabilities are "
                f"{sorted(CAPABILITIES)}")
        return capability in self.capabilities

    def chat_payload_extras(self, *, enable_reasoning: bool) -> Mapping[str, Any]:
        """The extra chat-completion payload fields this endpoint kind understands.

        Only a llama.cpp ``--jinja`` build reads ``chat_template_kwargs`` to suppress a reasoning
        model's chain-of-thought; a template that does not declare the flag ignores it, but a strict
        OpenAI-compatible server rejects the unknown field. So the extra is emitted only for a
        provider that honours it, and only when reasoning is being suppressed.
        """
        if self._sends_template_kwargs and not enable_reasoning:
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return {}

    def __repr__(self) -> str:
        return f"<LlmProvider {self.name!r} {sorted(self.capabilities)}>"


# Written only at import (below), effectively immutable thereafter; a name collision with a
# *different* provider is refused rather than silently overwritten, so selection cannot depend on
# import order. See the module docstring for why this bespoke registry, not the general one.
_REGISTRY: dict[str, LlmProvider] = {}


def register_provider(provider: LlmProvider) -> LlmProvider:
    """Register ``provider`` under its name. Returns it, for convenience.

    A name collision with a *different* provider is an error, not a silent overwrite. Re-registering
    the same instance is a no-op.
    """
    lowered = provider.name.lower()
    existing = _REGISTRY.get(lowered)
    if existing is not None and existing is not provider:
        raise ProviderError(
            f"provider name {provider.name!r} is already registered to {existing!r}; cannot also "
            f"register {provider!r}")
    _REGISTRY[lowered] = provider
    return provider


def get_provider(name: str) -> LlmProvider:
    """Look up a provider by name, listing what is available when it is missing."""
    provider = _REGISTRY.get(name.lower())
    if provider is None:
        raise ProviderError(
            f"unknown provider {name!r}; available providers are "
            f"{', '.join(available_providers())}")
    return provider


def available_providers() -> list[str]:
    """The registered provider names, sorted."""
    return sorted(provider.name for provider in _REGISTRY.values())


# Built-in endpoint kinds. Only a llama.cpp router honours chat_template_kwargs; only Ollama lacks a
# rerank endpoint. A user adds an endpoint kind by calling register_provider from their own module
# (named by that name in models.toml) -- no edit here.
register_provider(LlmProvider(
    "llamacpp-router", capabilities=CAPABILITIES, sends_template_kwargs=True))
register_provider(LlmProvider(
    "ollama", capabilities=frozenset({"chat", "embedding"}), sends_template_kwargs=False))
register_provider(LlmProvider(
    "openai-compatible", capabilities=CAPABILITIES, sends_template_kwargs=False))
