"""The provider registry and the built-in endpoint kinds: capability queries, the request-quirk
split (only a llama.cpp router sends the reasoning-suppression kwarg), and registry hygiene."""
from __future__ import annotations

import pytest

from ragkit.llm.providers import (
    CAPABILITIES,
    LlmProvider,
    ProviderError,
    available_providers,
    get_provider,
    register_provider,
)


class TestBuiltins:
    def test_the_three_are_registered(self) -> None:
        # A superset check, not equality: the registry is module-global, so other test modules'
        # synthetic providers may also be present (the same property backends rely on).
        assert {"llamacpp-router", "ollama", "openai-compatible"}.issubset(available_providers())
        assert available_providers() == sorted(available_providers())

    def test_llamacpp_router_serves_everything(self) -> None:
        provider = get_provider("llamacpp-router")
        assert provider.capabilities == CAPABILITIES
        assert all(provider.supports(c) for c in CAPABILITIES)

    def test_ollama_has_no_rerank(self) -> None:
        provider = get_provider("ollama")
        assert provider.supports("chat")
        assert provider.supports("embedding")
        assert not provider.supports("rerank")

    def test_openai_compatible_serves_everything(self) -> None:
        assert get_provider("openai-compatible").capabilities == CAPABILITIES

    def test_lookup_is_case_insensitive(self) -> None:
        assert get_provider("Ollama") is get_provider("ollama")


class TestChatPayloadExtras:
    def test_llamacpp_suppresses_reasoning_by_default(self) -> None:
        extras = get_provider("llamacpp-router").chat_payload_extras(enable_reasoning=False)
        assert extras == {"chat_template_kwargs": {"enable_thinking": False}}

    def test_llamacpp_sends_nothing_when_reasoning_is_on(self) -> None:
        assert get_provider("llamacpp-router").chat_payload_extras(enable_reasoning=True) == {}

    def test_ollama_never_sends_the_llamacpp_kwarg(self) -> None:
        provider = get_provider("ollama")
        assert provider.chat_payload_extras(enable_reasoning=False) == {}
        assert provider.chat_payload_extras(enable_reasoning=True) == {}

    def test_openai_compatible_never_sends_the_llamacpp_kwarg(self) -> None:
        # A strict OpenAI server would 400 on the unknown field, so it is never emitted.
        assert get_provider("openai-compatible").chat_payload_extras(enable_reasoning=False) == {}


class TestSupportsValidation:
    def test_unknown_capability_is_a_programming_error(self) -> None:
        with pytest.raises(ProviderError, match="unknown capability"):
            get_provider("ollama").supports("vision")


class TestConstruction:
    def test_blank_name_is_refused(self) -> None:
        with pytest.raises(ProviderError, match="non-empty name"):
            LlmProvider("  ", capabilities={"chat"}, sends_template_kwargs=False)

    def test_unknown_capability_in_the_declaration_is_refused(self) -> None:
        with pytest.raises(ProviderError, match="unknown capabilities"):
            LlmProvider("x", capabilities={"chat", "vision"}, sends_template_kwargs=False)

    def test_repr_names_capabilities(self) -> None:
        text = repr(LlmProvider("x", capabilities={"chat"}, sends_template_kwargs=False))
        assert "x" in text and "chat" in text


class TestRegistry:
    def test_unknown_provider_lists_the_known_ones(self) -> None:
        with pytest.raises(ProviderError, match="unknown provider 'vllm'; available providers"):
            get_provider("vllm")

    def test_registering_the_same_instance_twice_is_a_noop(self) -> None:
        provider = LlmProvider("re-register-me", capabilities={"chat"}, sends_template_kwargs=False)
        register_provider(provider)
        assert register_provider(provider) is provider

    def test_name_collision_with_a_different_instance_is_refused(self) -> None:
        register_provider(
            LlmProvider("collide", capabilities={"chat"}, sends_template_kwargs=False))
        with pytest.raises(ProviderError, match="already registered"):
            register_provider(
                LlmProvider("collide", capabilities={"embedding"}, sends_template_kwargs=False))
