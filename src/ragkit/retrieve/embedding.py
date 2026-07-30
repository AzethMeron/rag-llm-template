"""A client for a local, OpenAI-compatible ``/v1/embeddings`` endpoint (llama.cpp, ollama, vLLM).

Encodes texts into L2-normalised dense vectors, so a later dot product is the cosine similarity.
Placeholders are stripped before embedding, mirroring the lexical side, so a token every line
shares (``[[0]]``) cannot pull unrelated lines together. numpy is imported lazily inside the one
module that needs it, so the retrieval layer imports without it until a dense path is actually
built. Behind an injectable ``httpx.Client`` for testing with an in-memory transport.
"""
from __future__ import annotations

import re
import time
from collections.abc import Sequence
from typing import Any

import httpx

from ragkit.core.errors import RagkitError

_PLACEHOLDER = re.compile(r"\[\[\d+\]\]")
# Transient failures worth retrying: a timed-out or dropped connection, or a server-side/rate-limit
# status. A 4xx (bad request) or a malformed reply is deterministic and is raised at once.
_TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})

DEFAULT_EMBEDDING_MIN_SCORE = 0.55
"""A sensible starting cosine floor. On a different scale from the lexical floor: embedding
cosines cluster high (a loosely related pair still scores ~0.5), so the lexical 0.30 would admit
almost everything. Tune per corpus."""


class EmbeddingError(RagkitError):
    """The embedding endpoint is unreachable or misbehaving, or numpy is unavailable. Structured so
    a run stops with a message naming the cause rather than proceeding on empty vectors."""

    def __init__(self, reason: str, *, url: str | None = None) -> None:
        super().__init__(reason, endpoint=url)
        self.url = url


def _clean(text: str) -> str:
    """Strip placeholders before embedding; never empty (an all-placeholder line embeds a space)."""
    return _PLACEHOLDER.sub(" ", text).strip() or " "


class EmbeddingClient:
    """Embeds text via a local endpoint, returning L2-normalised vectors as plain lists of floats
    (so callers that do not want numpy need not import it)."""

    def __init__(self, *, base_url: str, model: str = "local", batch_size: int = 64,
                 timeout_seconds: float = 120.0, max_retries: int = 4,
                 retry_backoff_seconds: float = 1.0, client: httpx.Client | None = None) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        if retry_backoff_seconds < 0:
            raise ValueError(f"retry_backoff_seconds must be >= 0, got {retry_backoff_seconds}")
        self._np = _require_numpy()
        self._url = base_url.rstrip("/") + "/embeddings"
        self._model = model
        self._batch = batch_size
        self._max_retries = max_retries
        self._backoff = retry_backoff_seconds
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = client is None

    def embed(self, texts: Sequence[str]) -> list[Sequence[float]]:
        """Embed ``texts`` (batched), L2-normalised. Rows are returned in input order."""
        matrix = self._embed_matrix(texts)
        return [row.tolist() for row in matrix]

    def embed_one(self, text: str) -> list[float]:
        return list(self.embed([text])[0])

    def _embed_matrix(self, texts: Sequence[str]) -> Any:
        np = self._np
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch):
            chunk = [_clean(t) for t in texts[start:start + self._batch]]
            vectors.extend(self._request(chunk))
        try:
            matrix = np.asarray(vectors, dtype=np.float32)
        except (ValueError, TypeError) as exc:
            raise EmbeddingError(f"embeddings are not a uniform numeric matrix: {exc}",
                                 url=self._url) from exc
        if matrix.ndim != 2:
            raise EmbeddingError(f"expected 2-D embeddings, got shape {matrix.shape}",
                                 url=self._url)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
        return matrix

    def _post_with_retry(self, chunk: list[str]) -> list[Any]:
        """POST one batch, retrying a transient failure (timeout, dropped connection, 5xx/429) with
        exponential backoff so a long ingest survives a passing stall instead of dying on it. A
        deterministic failure (4xx, malformed reply) is raised at once, never retried."""
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(
                    self._url, json={"model": self._model, "input": chunk})
                response.raise_for_status()
                return response.json()["data"]
            except (KeyError, ValueError) as exc:
                raise EmbeddingError(f"malformed embedding response: {exc}", url=self._url) from exc
            except httpx.HTTPError as exc:
                if attempt >= self._max_retries or not _is_transient(exc):
                    raise EmbeddingError(
                        f"embedding request failed after {attempt + 1} attempt(s): {exc}",
                        url=self._url) from exc
                time.sleep(self._backoff * 2 ** attempt)
        raise EmbeddingError("embedding retries exhausted", url=self._url)  # pragma: no cover

    def _request(self, chunk: list[str]) -> list[list[float]]:
        data = self._post_with_retry(chunk)
        if len(data) != len(chunk):
            raise EmbeddingError(
                f"endpoint returned {len(data)} embeddings for {len(chunk)} inputs", url=self._url)
        try:
            return [item["embedding"] for item in data]
        except (KeyError, TypeError) as exc:
            raise EmbeddingError(
                f"malformed embedding response, an item has no 'embedding': {exc}",
                url=self._url) from exc

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


def dedup_embed(embedder: EmbeddingClient, texts: Sequence[str]) -> dict[str, Sequence[float]]:
    """Embed the distinct texts in ``texts`` once each, keyed by text. A batch built from a real
    corpus routinely repeats an index text (a duplicate line, a shared passage); embedding each
    distinct value once rather than once per occurrence is a pure cost saving with no RAM-map
    persisted across batches — a rare cross-batch duplicate is simply re-embedded."""
    unique = list(dict.fromkeys(texts))
    return dict(zip(unique, embedder.embed(unique), strict=True))


def _is_transient(exc: httpx.HTTPError) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _TRANSIENT_STATUS
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


def _require_numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:
        raise EmbeddingError(
            "dense embedding needs numpy, which is not installed (pip install numpy; it ships in "
            "requirements.txt for the dense/vector feature). Use a lexical-only retrieval config "
            "if you do not want it.") from exc
    return numpy
