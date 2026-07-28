"""The rerank client and its whole error taxonomy, plus the sigmoid squash."""
from __future__ import annotations

import math

import httpx
import pytest

from ragkit.retrieve.rerank import RerankError, sigmoid

from .conftest import rerank_client


def _results(*scores: float) -> httpx.Response:
    return httpx.Response(200, json={
        "results": [{"index": i, "relevance_score": s} for i, s in enumerate(scores)]})


class TestRerank:
    def test_scores_and_sorts_best_first(self) -> None:
        # Server returns them out of order; the client re-sorts.
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"results": [
                {"index": 0, "relevance_score": 0.1}, {"index": 1, "relevance_score": 0.9}]})
        result = rerank_client(handler).rerank("q", ["a", "b"])
        assert result == [(1, 0.9), (0, 0.1)]

    def test_empty_documents_short_circuits(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("should not be called")
        assert rerank_client(handler).rerank("q", []) == []

    def test_http_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="down")
        with pytest.raises(RerankError, match="rerank request failed"):
            rerank_client(handler).rerank("q", ["a"])

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
