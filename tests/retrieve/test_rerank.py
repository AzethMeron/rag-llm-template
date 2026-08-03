"""The rerank client and its whole error taxonomy, plus the sigmoid squash."""
from __future__ import annotations

import math

import httpx
import pytest

from ragkit.retrieve.rerank import RerankClient, RerankError, sigmoid

from .conftest import rerank_client


def _results(*scores: float) -> httpx.Response:
    return httpx.Response(200, json={
        "results": [{"index": i, "relevance_score": s} for i, s in enumerate(scores)]})


class TestRerank:
    def test_scores_and_sorts_best_first(self) -> None:
        # Server returns them out of order; the client re-sorts. Scores come back squashed,
        # because the default score_scale is the llama.cpp logit.
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"results": [
                {"index": 0, "relevance_score": 0.1}, {"index": 1, "relevance_score": 0.9}]})
        result = rerank_client(handler).rerank("q", ["a", "b"])
        assert [index for index, _ in result] == [1, 0]
        assert result == [(1, pytest.approx(sigmoid(0.9))), (0, pytest.approx(sigmoid(0.1)))]

    def test_empty_documents_short_circuits(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("should not be called")
        assert rerank_client(handler).rerank("q", []) == []

    def test_http_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="down")
        with pytest.raises(RerankError, match="rerank request failed"):
            rerank_client(handler).rerank("q", ["a"])

    def test_a_transient_status_is_retried_then_succeeds(self) -> None:
        # A busy rerank server answers 503 (all --parallel slots in use); retried with backoff like
        # the embedding client, rather than aborting a long hybrid run (the client had no retry).
        calls = {"n": 0}

        def handler(_request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(503, text="busy") if calls["n"] == 1 else _results(0.5)
        result = rerank_client(handler, max_retries=3).rerank("q", ["a"])
        assert result == [(0, pytest.approx(sigmoid(0.5)))]
        assert calls["n"] == 2  # one retry

    def test_gives_up_after_the_retry_budget(self) -> None:
        calls = {"n": 0}

        def handler(_request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(503, text="busy")
        with pytest.raises(RerankError, match="after 3 attempt"):
            rerank_client(handler, max_retries=2).rerank("q", ["a"])
        assert calls["n"] == 3  # initial + 2 retries

    def test_a_4xx_is_not_retried(self) -> None:
        calls = {"n": 0}

        def handler(_request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(400, text="bad request")
        with pytest.raises(RerankError, match="rerank request failed"):
            rerank_client(handler, max_retries=3).rerank("q", ["a"])
        assert calls["n"] == 1  # deterministic 4xx -> no retry

    def test_bad_retry_config(self) -> None:
        with pytest.raises(RerankError, match="max_retries"):
            RerankClient(base_url="http://x", max_retries=-1)

    def test_wrong_count(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return _results(0.5)  # one result for two docs
        with pytest.raises(RerankError, match="returned 1 results for 2"):
            rerank_client(handler).rerank("q", ["a", "b"])

    def test_index_out_of_range(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"results": [{"index": 5, "relevance_score": 0.5}]})
        with pytest.raises(RerankError, match="out of range"):
            rerank_client(handler).rerank("q", ["a"])

    def test_nan_score(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b'{"results":[{"index":0,"relevance_score":NaN}]}',
                                  headers={"content-type": "application/json"})
        with pytest.raises(RerankError, match="NaN"):
            rerank_client(handler).rerank("q", ["a"])

    def test_duplicate_indices(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"results": [
                {"index": 0, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.8}]})
        with pytest.raises(RerankError, match="duplicate result indices"):
            rerank_client(handler).rerank("q", ["a", "b"])

    def test_malformed_missing_field(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"results": [{"index": 0}]})
        with pytest.raises(RerankError, match="missing 'index' or 'relevance_score'"):
            rerank_client(handler).rerank("q", ["a"])

    def test_malformed_no_results_key(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"nope": []})
        with pytest.raises(RerankError, match="malformed rerank response"):
            rerank_client(handler).rerank("q", ["a"])

    def test_close_ownership(self) -> None:
        http = httpx.Client(transport=httpx.MockTransport(lambda r: _results(0.5)))
        from ragkit.retrieve.rerank import RerankClient
        client = RerankClient(base_url="http://x", client=http)
        client.close()
        assert http.post("http://x/rerank", json={}).status_code == 200  # injected, still open
        http.close()


class TestScoreScale:
    """Regression: the hybrid retriever squashed whatever `rerank()` returned, which is right for
    llama.cpp's unbounded logit and wrong for a Jina/Cohere `relevance_score` already in [0, 1] --
    that got mapped into [0.5, 0.731], preserving the ranking but making every floor meaningless
    (a min_score of 0.30 then admits everything). Normalising in the driver, which knows its own
    endpoint's scale, is the fix; `score_scale` says which scale that is."""

    def _client(self, scale: str) -> RerankClient:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"results": [
                {"index": 0, "relevance_score": 0.2}, {"index": 1, "relevance_score": 0.8}]})
        return rerank_client(handler, score_scale=scale)

    def test_logit_scale_squashes(self) -> None:
        scored = dict(self._client("logit").rerank("q", ["a", "b"]))
        assert scored[0] == pytest.approx(sigmoid(0.2)) and scored[0] > 0.5

    def test_unit_scale_passes_through(self) -> None:
        # The whole point: a 0.2 relevance stays 0.2, so a 0.30 floor still drops it.
        scored = dict(self._client("unit").rerank("q", ["a", "b"]))
        assert scored == {0: pytest.approx(0.2), 1: pytest.approx(0.8)}

    def test_unit_scale_clamps_a_stray_out_of_range_score(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"results": [
                {"index": 0, "relevance_score": 1.4}, {"index": 1, "relevance_score": -0.3}]})
        assert dict(rerank_client(handler, score_scale="unit").rerank("q", ["a", "b"])) == {
            0: 1.0, 1: 0.0}

    def test_both_scales_preserve_the_ranking(self) -> None:
        for scale in ("logit", "unit"):
            assert [i for i, _ in self._client(scale).rerank("q", ["a", "b"])] == [1, 0]

    def test_an_unknown_scale_is_refused(self) -> None:
        with pytest.raises(RerankError, match="score_scale must be one of"):
            RerankClient(base_url="http://x", score_scale="probability")


class TestSigmoid:
    def test_squashes_to_unit_interval(self) -> None:
        assert sigmoid(0.0) == 0.5
        assert sigmoid(-2.0) < 0.5 < sigmoid(2.0)
        # The extremes saturate exactly (underflow to 0.0, and 1/(1+0) == 1.0), which is fine.
        assert sigmoid(-1000.0) == 0.0 and sigmoid(1000.0) == 1.0

    def test_no_overflow_on_extreme_negatives(self) -> None:
        assert sigmoid(-1e308) == 0.0  # would OverflowError with the naive form

    def test_nan_is_refused(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            sigmoid(math.nan)
