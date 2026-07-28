"""Backends: request shaping per model family, and the name/auto selection registry."""
from __future__ import annotations

import pytest

from ragkit.core.ports import Message
from ragkit.llm.backends import (
    BackendError,
    JsonObjectBackend,
    SchemaBackend,
    available_backends,
    get_backend,
    register_backend,
    resolve_backend,
    suggest_backend,
)


class TestRequestShaping:
    def test_schema_backend_sends_a_strict_json_schema_and_unchanged_messages(self) -> None:
        backend = SchemaBackend("x")
        messages = [Message("system", "s"), Message("user", "u")]
        request = backend.structured_request(messages, {"type": "object"})
        assert request.response_format["type"] == "json_schema"
        assert request.response_format["json_schema"]["strict"] is True
        assert request.messages == tuple(messages)  # unchanged

    def test_json_object_backend_describes_the_shape_in_the_last_user_turn(self) -> None:
        backend = JsonObjectBackend("x")
        schema = {"type": "object", "required": ["translation"],
                  "properties": {"translation": {"type": "string"},
                                 "note": {"type": ["string", "null"]}}}
        request = backend.structured_request([Message("system", "s"), Message("user", "u")], schema)
        assert request.response_format == {"type": "json_object"}
        last = request.messages[-1]
        assert last.role == "user"
        assert '"translation": <string>' in last.content
        assert '"note": <string|null> (optional)' in last.content  # union rendered, optional noted

    def test_json_object_appends_a_user_turn_when_none_exists(self) -> None:
        backend = JsonObjectBackend("x")
        request = backend.structured_request([Message("system", "only system")], {"type": "object"})
        assert request.messages[-1].role == "user"
        assert "single JSON object" in request.messages[-1].content


class TestRegistry:
    def test_generic_is_grammar_constrained(self) -> None:
        assert isinstance(get_backend("generic"), SchemaBackend)

    def test_aliases_resolve(self) -> None:
        assert get_backend("openai").name == "generic"
        assert get_backend("json_schema").name == "generic"

    def test_auto_picks_by_model_hint(self) -> None:
        assert resolve_backend("auto", "qwen3-2b-instruct").name == "generic"
        assert resolve_backend("auto", "bielik-11b").name == "bielik"
        assert resolve_backend("auto", "gemma-4-e2b").name == "gemma"

    def test_auto_falls_back_to_default_when_no_hint_matches(self) -> None:
        assert resolve_backend("auto", "some-unknown-model").name == "generic"

    def test_explicit_name_overrides_auto(self) -> None:
        assert resolve_backend("bielik", "qwen3").name == "bielik"

    def test_suggest_returns_none_when_no_hint_matches(self) -> None:
        assert suggest_backend("mystery-model") is None

    def test_unknown_backend_lists_the_available(self) -> None:
        with pytest.raises(BackendError, match=r"unknown backend 'nope'.*available"):
            get_backend("nope")

    def test_available_is_sorted_and_distinct(self) -> None:
        names = available_backends()
        assert names == sorted(names)
        assert "generic" in names and "bielik" in names

    def test_empty_name_is_refused(self) -> None:
        with pytest.raises(BackendError, match="non-empty name"):
            SchemaBackend("   ")

    def test_collision_with_a_different_backend_is_refused(self) -> None:
        register_backend(SchemaBackend("temp-unique-1"))
        with pytest.raises(BackendError, match="already registered"):
            register_backend(JsonObjectBackend("temp-unique-1"))

    def test_re_registering_the_same_instance_is_a_no_op(self) -> None:
        backend = SchemaBackend("temp-unique-2")
        register_backend(backend)
        register_backend(backend)  # idempotent
        assert get_backend("temp-unique-2") is backend
