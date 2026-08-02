"""Optional cross-encoder reranker over a local ``/v1/rerank`` endpoint (llama.cpp built with
``--reranking``, or any Jina/TEI-compatible rerank server).

A first-stage retriever must be cheap enough to run over the whole candidate pool; a reranker is
the opposite trade — expensive but precise, so it only ever scores a short list a first stage
narrowed down. It never replaces a first-stage retriever on its own; the hybrid retriever is what
puts one in front of it. Behind an injectable ``httpx.Client`` so it is testable with an in-memory
transport and no server.

**Normalisation happens here, in the driver.** ``rerank()`` returns relevance already in
``[0, 1]``, the convention the vector drivers also follow — each one knows its own backend's scale
and converts, rather than leaving every caller to guess. The scales genuinely differ: llama.cpp
returns an unbounded cross-encoder logit, while Jina and Cohere return a ``relevance_score``
already in ``[0, 1]``. ``score_scale`` says which, because nothing in the response distinguishes
them; getting it wrong is silent, not loud (squashing an already-``[0, 1]`` score maps it into
``[0.5, 0.731]``, which preserves ranking but makes every floor meaningless).
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import httpx

from ragkit.core.errors import RagkitError

SCORE_SCALES = frozenset({"logit", "unit"})
"""What an endpoint's ``relevance_score`` means. ``logit``: an unbounded cross-encoder logit
(llama.cpp ``--reranking``), squashed with :func:`sigmoid`. ``unit``: already in ``[0, 1]``
(Jina, Cohere, most TEI deployments), passed through with a clamp."""


class RerankError(RagkitError):
    """The rerank endpoint is unreachable or returned something the client cannot use. Structured
    so a caller can catch it and fail clearly, naming the endpoint, rather than let a raw HTTP or
    parsing exception escape."""

    def __init__(self, reason: str, *, url: str | None = None) -> None:
        super().__init__(reason, endpoint=url)
        self.url = url


class RerankClient:
    """A thin client for one ``/v1/rerank`` endpoint. Does no I/O at construction (a reranker has
    no index to build); every call is a fresh request over the candidates it is given."""

    def __init__(self, *, base_url: str, model: str = "local", timeout_seconds: float = 120.0,
                 score_scale: str = "logit", client: httpx.Client | None = None) -> None:
        if score_scale not in SCORE_SCALES:
            raise RerankError(
                f"score_scale must be one of {sorted(SCORE_SCALES)}, got {score_scale!r}",
                url=base_url)
        self._url = base_url.rstrip("/") + "/rerank"
        self._model = model
        self._score_scale = score_scale
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = client is None

    def rerank(self, query: str, documents: Sequence[str]) -> list[tuple[int, float]]:
        """Score ``documents`` against ``query``, best first. Returns ``(index, relevance)`` pairs
        with ``relevance`` in ``[0, 1]`` (see ``score_scale`` and the module docstring); the
        indices are guaranteed a permutation of ``range(len(documents))``, so a caller may index by
        them without defending itself. Empty ``documents`` short-circuits with no HTTP call."""
        if not documents:
            return []
        try:
            response = self._client.post(
                self._url,
                json={"model": self._model, "query": query, "documents": list(documents)})
            response.raise_for_status()
            results = response.json()["results"]
        except httpx.HTTPError as exc:
            raise RerankError(f"rerank request failed: {exc}", url=self._url) from exc
        except (KeyError, ValueError) as exc:
            raise RerankError(f"malformed rerank response: {exc}", url=self._url) from exc
        if len(results) != len(documents):
            raise RerankError(
                f"endpoint returned {len(results)} results for {len(documents)} documents",
                url=self._url)
        try:
            scored = [(int(item["index"]), float(item["relevance_score"])) for item in results]
        except (KeyError, TypeError, ValueError) as exc:
            raise RerankError(
                f"malformed rerank response, a result is missing 'index' or 'relevance_score': "
                f"{exc}", url=self._url) from exc
        self._validate(scored, len(documents))
        normalized = [(index, self._to_relevance(score)) for index, score in scored]
        # Defensively re-sorted: the server conventionally returns best-first, but the contract
        # does not require it and a caller must be able to trust the order. Both scales are
        # monotone, so sorting before or after normalisation gives the same order.
        normalized.sort(key=lambda pair: -pair[1])
        return normalized

    def _to_relevance(self, score: float) -> float:
        if self._score_scale == "logit":
            return sigmoid(score)
        return max(0.0, min(1.0, score))

    def _validate(self, scored: list[tuple[int, float]], n: int) -> None:
        for index, score in scored:
            if not 0 <= index < n:
                raise RerankError(f"rerank result index {index} is out of range for {n} documents",
                                  url=self._url)
            if math.isnan(score):
                # NaN is unorderable: it would corrupt the sort and be ranked wherever a comparison
                # happened to leave it. ±inf is left alone -- it orders and squashes to 1.0/0.0.
                raise RerankError(f"rerank result for document {index} has a NaN relevance_score",
                                  url=self._url)
        # Distinctness is not implied by "right count, all in range": [(0,.9),(0,.8)] passes both.
        duplicates = sorted(i for i, count in Counter(i for i, _ in scored).items() if count > 1)
        if duplicates:
            raise RerankError(
                f"endpoint returned duplicate result indices {duplicates} for {n} documents",
                url=self._url)

    def close(self) -> None:
        """Release the HTTP client if this object opened it; an injected one is the injector's."""
        if self._owns_client:
            self._client.close()


def sigmoid(x: float) -> float:
    """Squash a cross-encoder's unbounded logit into ``[0, 1]`` for any finite ``x``.

    Branching on the sign is the only correct form: the naive ``1/(1+exp(-x))`` calls ``exp`` on a
    large *positive* argument for very negative ``x`` and dies with ``OverflowError`` past about
    -710, so one outlier logit would kill a whole run. Each branch only ever exponentiates a
    non-positive number, which underflows harmlessly to 0.0. ``NaN`` raises: there is no relevance
    it could honestly stand for, and it would silently poison the ranking downstream."""
    if math.isnan(x):
        raise ValueError(f"cannot squash a NaN score to a relevance in [0, 1] (sigmoid, {x!r})")
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exponential = math.exp(x)
    return exponential / (1.0 + exponential)
