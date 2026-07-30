"""The embedding client: batching, L2 normalisation, placeholder stripping, and errors."""
from __future__ import annotations

import json

import httpx
import pytest

from ragkit.retrieve.embedding import EmbeddingClient, EmbeddingError, _clean, dedup_embed

from .conftest import embedding_client, fake_embedder


class TestEmbed:
    def test_returns_l2_normalised_rows(self) -> None:
        client = fake_embedder(lambda t: [3.0, 4.0])  # norm 5 -> normalised (0.6, 0.8)
        [row] = client.embed(["x"])
        assert row == pytest.approx([0.6, 0.8])

    def test_embed_one(self) -> None:
        client = fake_embedder(lambda t: [1.0, 0.0])
        assert client.embed_one("x") == pytest.approx([1.0, 0.0])

    def test_empty_input(self) -> None:
        client = fake_embedder(lambda t: [1.0])
        assert client.embed([]) == []

    def test_batches(self) -> None:
        seen: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            inputs = json.loads(request.content)["input"]
            seen.append(len(inputs))
            return httpx.Response(200, json={"data": [{"embedding": [1.0]} for _ in inputs]})

        client = EmbeddingClient(base_url="http://x/v1", batch_size=2,
                                 client=httpx.Client(transport=httpx.MockTransport(handler)))
        client.embed(["a", "b", "c"])
        assert seen == [2, 1]  # two batches

    def test_placeholders_are_stripped_before_embedding(self) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            inputs = json.loads(request.content)["input"]
            seen.extend(inputs)
            return httpx.Response(200, json={"data": [{"embedding": [1.0]} for _ in inputs]})

        embedding_client(handler).embed(["[[0]] hello [[1]]"])
        assert "[[0]]" not in seen[0]


class TestDedupEmbed:
    def test_a_repeated_text_is_embedded_once(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            inputs = json.loads(request.content)["input"]
            calls.extend(inputs)
            return httpx.Response(200, json={"data": [{"embedding": [1.0]} for _ in inputs]})

        client = embedding_client(handler)
        result = dedup_embed(client, ["same", "same", "different"])
        assert calls == ["same", "different"]  # embedded once each, not three times
        assert set(result) == {"same", "different"}

    def test_every_result_maps_back_to_its_own_text(self) -> None:
        # Distinct directions, not just magnitudes -- embed() L2-normalises, so same-direction
        # vectors of different magnitude would collapse to the same normalised result.
        client = fake_embedder(lambda t: [1.0, 0.0] if t == "a" else [0.0, 1.0])
        result = dedup_embed(client, ["a", "bb"])
        assert result["a"] == pytest.approx([1.0, 0.0])
        assert result["bb"] == pytest.approx([0.0, 1.0])

    def test_no_cache_persists_across_separate_calls(self) -> None:
        # Deduplication is scoped to one call's batch, not a RAM-map remembered across calls -- a
        # rare cross-call duplicate is simply re-embedded, by design (see the docstring).
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            inputs = json.loads(request.content)["input"]
            calls.extend(inputs)
            return httpx.Response(200, json={"data": [{"embedding": [1.0]} for _ in inputs]})

        client = embedding_client(handler)
        dedup_embed(client, ["same"])
        dedup_embed(client, ["same"])
        assert calls == ["same", "same"]

    def test_empty_input(self) -> None:
        client = fake_embedder(lambda t: [1.0])
        assert dedup_embed(client, []) == {}


class TestErrors:
    def test_http_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="down")
        with pytest.raises(EmbeddingError, match="embedding request failed"):
            embedding_client(handler).embed(["x"])

    def test_wrong_count(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"embedding": [1.0]}]})  # 1 for 2
        with pytest.raises(EmbeddingError, match="returned 1 embeddings for 2"):
            embedding_client(handler).embed(["a", "b"])

    def test_missing_embedding_field(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"nope": 1}]})
        with pytest.raises(EmbeddingError, match="no 'embedding'"):
            embedding_client(handler).embed(["x"])

    def test_no_data_key(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"nope": 1})
        with pytest.raises(EmbeddingError, match="malformed embedding response"):
            embedding_client(handler).embed(["x"])

    def test_ragged_matrix(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            inputs = json.loads(_request.content)["input"]
            return httpx.Response(200, json={"data": [{"embedding": [1.0, 2.0]},
                                                      {"embedding": [1.0]}][:len(inputs)]})
        with pytest.raises(EmbeddingError, match="uniform numeric matrix"):
            embedding_client(handler).embed(["a", "b"])

    def test_bad_batch_size(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            EmbeddingClient(base_url="http://x", batch_size=0)

    def test_bad_retry_config(self) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            EmbeddingClient(base_url="http://x", max_retries=-1)
        with pytest.raises(ValueError, match="retry_backoff_seconds"):
            EmbeddingClient(base_url="http://x", retry_backoff_seconds=-1)


class TestRetries:
    """A transient failure during a long ingest must be retried, not fatal (regression: a single
    embedding timeout killed a multi-hour dense build)."""

    def _flaky(self, fail_times: int, exc: type[httpx.HTTPError] | int):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] <= fail_times:
                if isinstance(exc, int):
                    return httpx.Response(exc, text="transient")
                raise exc("transient", request=request)
            inputs = json.loads(request.content)["input"]
            return httpx.Response(200, json={"data": [{"embedding": [1.0, 0.0]} for _ in inputs]})
        return handler, calls

    def test_retries_a_timeout_then_succeeds(self) -> None:
        handler, calls = self._flaky(2, httpx.ReadTimeout)
        out = embedding_client(handler, max_retries=3).embed(["a"])
        assert len(out) == 1 and calls["n"] == 3  # 2 timeouts + 1 success

    def test_retries_a_503_then_succeeds(self) -> None:
        handler, calls = self._flaky(1, 503)
        out = embedding_client(handler, max_retries=3).embed(["a"])
        assert len(out) == 1 and calls["n"] == 2

    def test_gives_up_after_the_retry_budget(self) -> None:
        handler, calls = self._flaky(99, httpx.ConnectError)
        with pytest.raises(EmbeddingError, match="after 3 attempt"):
            embedding_client(handler, max_retries=2).embed(["a"])
        assert calls["n"] == 3  # initial + 2 retries

    def test_does_not_retry_a_client_error(self) -> None:
        handler, calls = self._flaky(99, 400)  # 4xx is deterministic
        with pytest.raises(EmbeddingError, match="after 1 attempt"):
            embedding_client(handler, max_retries=3).embed(["a"])
        assert calls["n"] == 1  # not retried

    def test_does_not_retry_a_malformed_reply(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"nope": 1})  # deterministic
        with pytest.raises(EmbeddingError, match="malformed"):
            embedding_client(handler, max_retries=3).embed(["x"])


class TestClean:
    def test_strips_placeholders(self) -> None:
        assert _clean("[[0]] hi [[1]]").strip() == "hi"

    def test_all_placeholder_becomes_space(self) -> None:
        assert _clean("[[0]][[1]]") == " "
