"""Transient-vs-deterministic HTTP failure classification, shared by every client that talks to a
model-serving endpoint — chat (:mod:`ragkit.llm.client`), embeddings and rerank
(:mod:`ragkit.retrieve`). One home so all three retry the *same* failures the same way, rather than
each client deciding differently (the chat client used to raise on a 429 the embedding client
retried, and the rerank client retried nothing at all).
"""
from __future__ import annotations

import httpx

# Statuses worth retrying with backoff: rate-limiting (429), request-timeout (408), and the
# transient 5xx a busy/overloaded server returns -- llama.cpp answers 503 when every --parallel
# slot is in use, and a cloud OpenAI-compatible endpoint answers 429 under load. A genuinely
# deterministic 4xx (400/401/403/404/422) is deliberately NOT here: retrying it cannot help.
TRANSIENT_HTTP_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def is_transient_http_error(exc: httpx.HTTPError) -> bool:
    """True when ``exc`` is a transient failure worth retrying with backoff: a timeout, a dropped
    connection, or a :data:`TRANSIENT_HTTP_STATUS` response (from ``raise_for_status``). A
    deterministic status error or any other ``httpx.HTTPError`` returns ``False``."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in TRANSIENT_HTTP_STATUS
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))
