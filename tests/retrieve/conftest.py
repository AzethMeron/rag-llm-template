"""Helpers to drive the embedding and rerank clients with no server."""
from __future__ import annotations

from collections.abc import Callable

import httpx

from ragkit.retrieve.embedding import EmbeddingClient
from ragkit.retrieve.rerank import RerankClient


def embedding_client(handler: Callable[[httpx.Request], httpx.Response], *,
                     max_retries: int = 0) -> EmbeddingClient:
    # Default max_retries=0 (fail fast) keeps the tests deterministic and quick; retry tests pass a
    # count explicitly. retry_backoff_seconds=0 means a retry never actually sleeps.
    return EmbeddingClient(base_url="http://x/v1", max_retries=max_retries, retry_backoff_seconds=0,
                           client=httpx.Client(transport=httpx.MockTransport(handler)))


def rerank_client(handler: Callable[[httpx.Request], httpx.Response]) -> RerankClient:
    return RerankClient(base_url="http://x/v1",
                        client=httpx.Client(transport=httpx.MockTransport(handler)))


def fake_embedder(vector_of: Callable[[str], list[float]]) -> EmbeddingClient:
    """An embedding client whose endpoint returns ``vector_of(text)`` for each input."""
    def handler(request: httpx.Request) -> httpx.Response:
        import json
        inputs = json.loads(request.content)["input"]
        return httpx.Response(200, json={"data": [{"embedding": vector_of(t)} for t in inputs]})
    return embedding_client(handler)
