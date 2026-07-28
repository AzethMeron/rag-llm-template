"""Remaining branch coverage for the retrieve layer."""
from __future__ import annotations

import sys

import httpx
import pytest

from ragkit.retrieve.embedding import EmbeddingClient, EmbeddingError
from ragkit.retrieve.rerank import RerankClient

from .conftest import embedding_client


def test_non_2d_embeddings_are_refused() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        import json
        inputs = json.loads(request.content)["input"]
        return httpx.Response(200, json={"data": [{"embedding": 5} for _ in inputs]})  # scalar
    with pytest.raises(EmbeddingError, match="2-D"):
        embedding_client(handler).embed(["x"])


def test_embedding_client_owns_and_closes() -> None:
    def block(_name: str) -> None:  # pragma: no cover - not called
        raise AssertionError
    client = EmbeddingClient(base_url="http://x")  # owns its httpx.Client
    client.close()  # no error


def test_missing_numpy_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "numpy", None)
    with pytest.raises(EmbeddingError, match="needs numpy"):
        EmbeddingClient(base_url="http://x")


def test_rerank_client_owns_and_closes() -> None:
    RerankClient(base_url="http://x").close()  # owns its httpx.Client; no error


def test_embedding_close_leaves_injected_client(monkeypatch: pytest.MonkeyPatch) -> None:
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    client = EmbeddingClient(base_url="http://x", client=http)
    client.close()  # injected -> not closed
    assert http.post("http://x/embeddings", json={}).status_code == 200
    http.close()


def test_hybrid_close_without_a_reranker() -> None:
    from ragkit.core.ports import Retrieved
    from ragkit.retrieve.hybrid import HybridRetriever

    class Stub:
        def retrieve(self, query: str, *, k: int, min_score: float = 0.0) -> tuple[Retrieved, ...]:
            return ()

    HybridRetriever(Stub(), Stub()).close()  # no reranker -> the False branch of the guard
