"""Helpers for driving the model client with no server: an in-memory HTTP transport that
returns scripted chat-completion replies."""
from __future__ import annotations

import json
from collections.abc import Callable

import httpx

from ragkit.llm import LlmClient, ServerConfig
from ragkit.llm.backends import resolve_backend


def chat_reply(content: str | None, *, finish_reason: str = "stop",
               prompt_tokens: int = 5, completion_tokens: int = 3) -> httpx.Response:
    """A well-formed OpenAI-style chat completion carrying ``content``."""
    return httpx.Response(200, json={
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    })


def client_returning(
    handler: Callable[[httpx.Request], httpx.Response], *, model: str = "qwen3",
    backend: str = "auto", config: ServerConfig | None = None,
) -> LlmClient:
    """An :class:`LlmClient` whose transport is driven by ``handler`` — no server, no network."""
    transport = httpx.MockTransport(handler)
    server = config or ServerConfig(model=model)
    return LlmClient(server, backend=resolve_backend(backend, model),
                     client=httpx.Client(transport=transport))


def always(response: httpx.Response) -> Callable[[httpx.Request], httpx.Response]:
    """A handler that returns ``response`` for every request."""
    def handler(_request: httpx.Request) -> httpx.Response:
        return response
    return handler


def request_body(request: httpx.Request) -> dict:
    return json.loads(request.content)
