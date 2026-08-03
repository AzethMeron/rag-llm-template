"""The transport client: happy path, the retry policy, the error taxonomy by blast radius, the
JSON-envelope recovery, schema-shape validation, code-fence tolerance, and RAII."""
from __future__ import annotations

import httpx
import pytest

import threading

from ragkit.core.ports import Message, SamplingParams
from ragkit.llm import LlmClient, ServerConfig
from ragkit.llm.client import UsageStats
from ragkit.llm.backends import resolve_backend
from ragkit.llm.providers import get_provider
from ragkit.llm.errors import (
    LlmContentError,
    LlmError,
    LlmIncompleteJsonError,
    LlmRefusalError,
    LlmTruncationError,
)

from .conftest import always, chat_reply, client_returning, request_body

SCHEMA = {"type": "object", "required": ["translation"],
          "properties": {"translation": {"type": "string"}}}
USER = [Message("system", "s"), Message("user", "u")]


def _json_call(client: LlmClient, **kwargs: object) -> dict:
    return client.complete_json(USER, SCHEMA, role="translate", **kwargs)  # type: ignore[arg-type]


class TestHappyPath:
    def test_complete_json_returns_the_parsed_object(self) -> None:
        client = client_returning(always(chat_reply('{"translation": "kot"}')))
        assert _json_call(client) == {"translation": "kot"}

    def test_model_override_is_sent(self) -> None:
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(request_body(request))
            return chat_reply('{"translation": "x"}')

        client = client_returning(handler, model="config-model")
        _json_call(client, model="override-model")
        assert seen["model"] == "override-model"

    def test_usage_is_accounted(self) -> None:
        client = client_returning(always(chat_reply('{"translation": "x"}',
                                                    prompt_tokens=10, completion_tokens=4)))
        _json_call(client)
        assert client.stats.prompt_tokens == 10
        assert client.stats.completion_tokens == 4
        assert client.stats.peak_prompt_tokens == 10


class TestUsageStatsConcurrency:
    def test_concurrent_updates_do_not_lose_increments(self) -> None:
        # One UsageStats is shared by every worker in a concurrent run; its docstring warns a bare
        # `+=` would lose updates. Drive real contention on the lock-guarded counters and assert
        # every increment survives (a missing lock would drop some under contention).
        stats = UsageStats()
        threads, per_thread = 8, 250

        def worker() -> None:
            for _ in range(per_thread):
                stats.record({"prompt_tokens": 1, "completion_tokens": 2}, 0.001)
                stats.record_retry()
                stats.record_refusal()

        workers = [threading.Thread(target=worker) for _ in range(threads)]
        for t in workers:
            t.start()
        for t in workers:
            t.join()

        total = threads * per_thread
        assert stats.requests == total
        assert stats.prompt_tokens == total and stats.completion_tokens == 2 * total
        assert stats.retries == total and stats.refusals == total


class TestSampling:
    def _sent(self, **kwargs: object) -> dict:
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(request_body(request))
            return chat_reply('{"translation": "x"}')

        _json_call(client_returning(handler), **kwargs)
        return seen

    def test_default_sends_only_temperature_no_llamacpp_only_knobs(self) -> None:
        # A strict-OpenAI endpoint must never be handed top_k/min_p/repeat_penalty it would reject.
        sent = self._sent()
        assert sent["temperature"] == 0.2
        assert not ({"top_p", "top_k", "min_p", "seed", "repeat_penalty", "presence_penalty",
                     "frequency_penalty", "stop"} & sent.keys())

    def test_persona_sampling_knobs_are_forwarded(self) -> None:
        sampling = SamplingParams(temperature=0.7, top_p=0.9, top_k=40, min_p=0.05, seed=123,
                                  presence_penalty=0.5, frequency_penalty=-0.5, repeat_penalty=1.1,
                                  stop=("</end>",))
        sent = self._sent(sampling=sampling)
        assert sent["temperature"] == 0.7 and sent["top_p"] == 0.9 and sent["top_k"] == 40
        assert sent["min_p"] == 0.05 and sent["seed"] == 123 and sent["repeat_penalty"] == 1.1
        assert sent["presence_penalty"] == 0.5 and sent["frequency_penalty"] == -0.5
        assert sent["stop"] == ["</end>"]

    def test_max_tokens_is_separate_from_sampling(self) -> None:
        sent = self._sent(sampling=SamplingParams(temperature=0.0), max_tokens=321)
        assert sent["max_tokens"] == 321 and sent["temperature"] == 0.0


class TestRetryPolicy:
    def test_a_5xx_is_retried_then_succeeds(self) -> None:
        calls = {"n": 0}

        def handler(_request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(503, text="temporarily unavailable")
            return chat_reply('{"translation": "ok"}')

        client = client_returning(handler, config=ServerConfig(retry_backoff_seconds=0))
        assert _json_call(client) == {"translation": "ok"}
        assert client.stats.retries == 1

    def test_a_429_rate_limit_is_retried_then_succeeds(self) -> None:
        # 429 (Too Many Requests) and 408 are transient -- retried with backoff like a 5xx, not
        # raised at once the way a deterministic 4xx is (the chat client used to raise on a 429 the
        # embedding client already retried).
        calls = {"n": 0}

        def handler(_request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, text="Too Many Requests")
            return chat_reply('{"translation": "ok"}')

        client = client_returning(handler, config=ServerConfig(retry_backoff_seconds=0))
        assert _json_call(client) == {"translation": "ok"}
        assert client.stats.retries == 1

    def test_a_4xx_is_not_retried_and_names_the_fix(self) -> None:
        client = client_returning(
            always(httpx.Response(400, text="Failed to initialize samplers")),
            model="bielik-11b", backend="generic",
            config=ServerConfig(model="bielik-11b"))
        with pytest.raises(LlmError, match="json_object backend"):
            _json_call(client)

    def test_transport_failure_is_retried_then_surfaces(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        client = client_returning(handler, config=ServerConfig(max_retries=2,
                                                               retry_backoff_seconds=0))
        with pytest.raises(LlmError, match="transport failure"):
            _json_call(client)

    def test_malformed_response_body_is_retried_then_surfaces(self) -> None:
        client = client_returning(always(httpx.Response(200, text="not json at all")),
                                  config=ServerConfig(max_retries=2, retry_backoff_seconds=0))
        with pytest.raises(LlmError, match="malformed response"):
            _json_call(client)

    def test_context_overflow_5xx_is_diagnosed(self) -> None:
        client = client_returning(
            always(httpx.Response(500, text="the prompt exceeds context size n_ctx")),
            config=ServerConfig(max_retries=1, retry_backoff_seconds=0))
        with pytest.raises(LlmError, match="larger than the server's context window"):
            _json_call(client)


class TestErrorTaxonomy:
    def test_refusal_when_content_is_null(self) -> None:
        client = client_returning(always(chat_reply(None)))
        with pytest.raises(LlmRefusalError):
            _json_call(client)
        assert client.stats.refusals == 1

    def test_truncation_when_finish_reason_is_length(self) -> None:
        client = client_returning(always(chat_reply('{"translation": "unfinis',
                                                    finish_reason="length")))
        with pytest.raises(LlmTruncationError):
            _json_call(client)

    def test_unparseable_json_is_a_content_error(self) -> None:
        client = client_returning(always(chat_reply("this is not json {")))
        with pytest.raises(LlmContentError, match="not valid JSON"):
            _json_call(client)

    def test_blast_radius_hierarchy(self) -> None:
        # A content error IS an LlmError, but a bare LlmError is not a content error -- the
        # distinction the caller relies on to record-and-continue vs stop-the-run.
        assert issubclass(LlmContentError, LlmError)
        assert not issubclass(LlmError, LlmContentError)


class TestEnvelopeRecovery:
    def test_a_brace_only_truncation_is_recovered_as_a_candidate(self) -> None:
        # The value's own closing quote is present; only the object brace is missing.
        client = client_returning(always(chat_reply('{"translation": "kot"')))
        with pytest.raises(LlmIncompleteJsonError) as info:
            _json_call(client)
        assert info.value.recovered == {"translation": "kot"}

    def test_a_quote_and_brace_truncation_is_recovered(self) -> None:
        # Severed mid-value: both the closing quote and the brace are missing.
        client = client_returning(always(chat_reply('{"translation": "the cat sat')))
        with pytest.raises(LlmIncompleteJsonError) as info:
            _json_call(client)
        assert info.value.recovered["translation"] == "the cat sat"

    def test_recovery_that_still_fails_the_schema_falls_through_to_a_plain_error(self) -> None:
        # Recovers to an object, but one missing the required field: not an incomplete-envelope
        # success, just malformed -- the plain content error.
        client = client_returning(always(chat_reply('{"other": "x"')))
        with pytest.raises(LlmContentError) as info:
            _json_call(client)
        assert not isinstance(info.value, LlmIncompleteJsonError)

    def test_deeply_broken_json_is_not_recovered(self) -> None:
        client = client_returning(always(chat_reply('{"translation": ["a", ')))
        with pytest.raises(LlmContentError) as info:
            _json_call(client)
        assert not isinstance(info.value, LlmIncompleteJsonError)


class TestSchemaShape:
    def test_missing_required_field_is_refused(self) -> None:
        client = client_returning(always(chat_reply('{"other": "x"}')))
        with pytest.raises(LlmContentError, match="missing required fields"):
            _json_call(client)

    def test_wrong_field_type_is_refused(self) -> None:
        client = client_returning(always(chat_reply('{"translation": 5}')))
        with pytest.raises(LlmContentError, match="wrong type"):
            _json_call(client)

    def test_non_object_reply_is_refused(self) -> None:
        client = client_returning(always(chat_reply('["a", "b"]')))
        with pytest.raises(LlmContentError, match="expected a JSON object"):
            _json_call(client)

    def test_union_type_and_absent_optional_are_accepted(self) -> None:
        schema = {"type": "object", "required": ["a"],
                  "properties": {"a": {"type": "string"}, "b": {"type": ["array", "null"]}}}
        client = client_returning(always(chat_reply('{"a": "x", "b": null}')))
        assert client.complete_json(USER, schema, role="r") == {"a": "x", "b": None}


class TestCodeFence:
    def test_a_multiline_fence_is_stripped(self) -> None:
        client = client_returning(always(chat_reply('```json\n{"translation": "k"}\n```')))
        assert _json_call(client) == {"translation": "k"}

    def test_a_oneline_fence_is_stripped(self) -> None:
        client = client_returning(always(chat_reply('```{"translation": "k"}```')))
        assert _json_call(client) == {"translation": "k"}

    def test_a_fenced_and_truncated_reply_still_reaches_recovery(self) -> None:
        # An unclosed fence must not strand the payload; the envelope recovery is still reached.
        client = client_returning(always(chat_reply('```json\n{"translation": "k"')))
        with pytest.raises(LlmIncompleteJsonError):
            _json_call(client)


class TestContextWarning:
    def test_fires_once_when_the_budget_is_tight(self, caplog: pytest.LogCaptureFixture) -> None:
        # A small window and a big output budget push the projection past 80%.
        client = client_returning(always(chat_reply('{"translation": "x"}')),
                                  config=ServerConfig(model="qwen3", context_window=100))
        with caplog.at_level("WARNING"):
            client.complete(USER, role="translate", max_tokens=90, schema=SCHEMA)
            client.complete(USER, role="translate", max_tokens=90, schema=SCHEMA)
        assert sum("prompt budget is tight" in r.message for r in caplog.records) == 1

    def test_silent_when_window_unset(self, caplog: pytest.LogCaptureFixture) -> None:
        client = client_returning(always(chat_reply('{"translation": "x"}')))
        with caplog.at_level("WARNING"):
            client.complete(USER, role="translate", max_tokens=9000, schema=SCHEMA)
        assert not any("prompt budget" in r.message for r in caplog.records)


class TestHealthAndConfig:
    def test_health_true_on_200(self) -> None:
        client = client_returning(always(httpx.Response(200, json={"data": []})))
        assert client.health() is True

    def test_health_false_on_transport_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        client = client_returning(handler)
        assert client.health() is False

    @pytest.mark.parametrize("kwargs", [
        {"max_retries": 0}, {"timeout_seconds": 0}, {"retry_backoff_seconds": -1},
        {"context_window": -1},
    ])
    def test_invalid_server_config_is_refused(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            ServerConfig(**kwargs)


class TestPlainAndConfig:
    def test_plain_completion_without_a_schema(self) -> None:
        client = client_returning(always(chat_reply("just some text")))
        assert client.complete(USER, role="x") == "just some text"

    def test_no_schema_sends_no_response_format(self) -> None:
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(request_body(request))
            return chat_reply("text")

        client_returning(handler).complete(USER, role="x")
        assert "response_format" not in seen

    def test_enable_reasoning_omits_the_thinking_kwarg(self) -> None:
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(request_body(request))
            return chat_reply("t")

        client = client_returning(handler, config=ServerConfig(enable_reasoning=True))
        client.complete(USER, role="x")
        assert "chat_template_kwargs" not in seen

    def test_reasoning_suppressed_by_default(self) -> None:
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(request_body(request))
            return chat_reply("t")

        client_returning(handler).complete(USER, role="x")
        assert seen["chat_template_kwargs"] == {"enable_thinking": False}

    def test_non_llamacpp_provider_never_sends_the_llamacpp_kwarg(self) -> None:
        # The kwarg is a llama.cpp --jinja extension; a strict OpenAI/Ollama server would reject it.
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(request_body(request))
            return chat_reply("t")

        config = ServerConfig(provider=get_provider("ollama"))
        client_returning(handler, config=config).complete(USER, role="x")
        assert "chat_template_kwargs" not in seen

    def test_rate_is_zero_before_any_request(self) -> None:
        client = client_returning(always(chat_reply("t")))
        assert client.stats.completion_tokens_per_second == 0.0

    def test_context_warning_silent_when_budget_is_comfortable(
            self, caplog: pytest.LogCaptureFixture) -> None:
        client = client_returning(always(chat_reply('{"translation": "x"}')),
                                  config=ServerConfig(model="qwen3", context_window=100000))
        with caplog.at_level("WARNING"):
            client.complete(USER, role="x", max_tokens=10, schema=SCHEMA)
        assert not any("prompt budget" in r.message for r in caplog.records)

    def test_used_as_a_context_manager(self) -> None:
        with client_returning(always(chat_reply('{"translation": "x"}'))) as client:
            assert _json_call(client) == {"translation": "x"}

    def test_a_generic_rejection_names_no_specific_fix(self) -> None:
        client = client_returning(always(httpx.Response(400, text="bad request, unspecified")))
        with pytest.raises(LlmError) as info:
            _json_call(client)
        assert "request rejected" in str(info.value) and "json_object" not in str(info.value)

    def test_a_plain_request_rejection_is_diagnosed_without_a_schema_hint(self) -> None:
        # No schema on the request, so the schema-specific diagnosis path is not taken.
        client = client_returning(always(httpx.Response(400, text="bad request")))
        with pytest.raises(LlmError, match="request rejected"):
            client.complete(USER, role="x")


class TestOwnership:
    def test_close_releases_an_owned_client(self) -> None:
        client = LlmClient(ServerConfig(), backend=resolve_backend("generic", "qwen"))
        client.close()  # owns its httpx.Client; no error

    def test_close_leaves_an_injected_client_open(self) -> None:
        transport = httpx.MockTransport(always(chat_reply('{"translation":"x"}')))
        http = httpx.Client(transport=transport)
        client = LlmClient(ServerConfig(), backend=resolve_backend("generic", "qwen"), client=http)
        client.close()
        # The injected client is the injector's; still usable.
        assert http.post("http://x/chat/completions", json={}).status_code == 200
        http.close()
