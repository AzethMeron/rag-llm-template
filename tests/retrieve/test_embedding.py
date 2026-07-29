"""The embedding client: batching, L2 normalisation, placeholder stripping, and errors."""
from __future__ import annotations

import json

import httpx
import pytest

from ragkit.retrieve.embedding import EmbeddingClient, EmbeddingError, _clean

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


class TestClean:
    def test_strips_placeholders(self) -> None:
        assert _clean("[[0]] hi [[1]]").strip() == "hi"

    def test_all_placeholder_becomes_space(self) -> None:
        assert _clean("[[0]][[1]]") == " "
