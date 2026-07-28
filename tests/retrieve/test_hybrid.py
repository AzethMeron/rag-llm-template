"""The hybrid retriever: fusion, the reranked and unreranked relevance scales, MMR, fail-loud."""
from __future__ import annotations

import httpx
import pytest

from ragkit.core.ports import Retrieved
from ragkit.retrieve.hybrid import HybridError, HybridRetriever
from ragkit.retrieve.rerank import RerankError

from .conftest import rerank_client


class _StubRetriever:
    def __init__(self, hits: list[Retrieved]) -> None:
        self._hits = hits

    def retrieve(self, query: str, *, k: int, min_score: float = 0.0) -> tuple[Retrieved, ...]:
        return tuple(h for h in self._hits if h.score >= min_score)[:k]


def _hits(*items: tuple[str, float]) -> list[Retrieved]:
    return [Retrieved(chunk_id=cid, text=f"text of {cid}", score=score) for cid, score in items]


class TestFusion:
    def test_agreement_ranks_first(self) -> None:
        lexical = _StubRetriever(_hits(("a", 0.9), ("b", 0.4)))
        dense = _StubRetriever(_hits(("b", 0.9), ("c", 0.8)))
        hybrid = HybridRetriever(lexical, dense, lexical_min_score=0.0, dense_min_score=0.0)
        result = hybrid.retrieve("q", k=3)
        # b is found by both arms, so it should rank first under RRF.
        assert result[0].chunk_id == "b"

    def test_empty_when_neither_arm_finds_anything(self) -> None:
        hybrid = HybridRetriever(_StubRetriever([]), _StubRetriever([]))
        assert hybrid.retrieve("q", k=3) == ()

    def test_k_zero(self) -> None:
        hybrid = HybridRetriever(_StubRetriever(_hits(("a", 0.9))), _StubRetriever([]))
        assert hybrid.retrieve("q", k=0) == ()

    def test_unreranked_relevance_is_a_fixed_scale(self) -> None:
        # A candidate both arms rank first scores 1.0 on the fixed RRF scale.
        both = _StubRetriever(_hits(("a", 0.9)))
        hybrid = HybridRetriever(both, both, lexical_min_score=0.0, dense_min_score=0.0)
        [hit] = hybrid.retrieve("q", k=1)
        assert hit.score == pytest.approx(1.0)

    def test_min_score_gate(self) -> None:
        lexical = _StubRetriever(_hits(("a", 0.9)))
        dense = _StubRetriever(_hits(("b", 0.9)))
        # Each is found by only one arm -> fused relevance 0.5; a floor above that drops both.
        hybrid = HybridRetriever(lexical, dense, lexical_min_score=0.0, dense_min_score=0.0)
        assert hybrid.retrieve("q", k=3, min_score=0.6) == ()


class TestRerank:
    def test_reranker_rescoring(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            import json
            docs = json.loads(request.content)["documents"]
            # Prefer whichever document mentions 'c'.
            return httpx.Response(200, json={"results": [
                {"index": i, "relevance_score": 5.0 if "c" in d else -5.0}
                for i, d in enumerate(docs)]})

        lexical = _StubRetriever(_hits(("a", 0.9)))
        dense = _StubRetriever(_hits(("c", 0.9)))
        hybrid = HybridRetriever(lexical, dense, reranker=rerank_client(handler),
                                 lexical_min_score=0.0, dense_min_score=0.0)
        result = hybrid.retrieve("q", k=2)
        assert result[0].chunk_id == "c"  # reranker lifted it

    def test_dead_reranker_propagates(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="down")
        lexical = _StubRetriever(_hits(("a", 0.9)))
        dense = _StubRetriever(_hits(("b", 0.9)))
        hybrid = HybridRetriever(lexical, dense,
                                 reranker=rerank_client(handler), lexical_min_score=0.0,
                                 dense_min_score=0.0)
        with pytest.raises(RerankError):  # no silent fallback to the unreranked order
            hybrid.retrieve("q", k=2)

    def test_close_closes_the_reranker(self) -> None:
        http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
        from ragkit.retrieve.rerank import RerankClient
        hybrid = HybridRetriever(_StubRetriever([]), _StubRetriever([]),
                                 reranker=RerankClient(base_url="http://x", client=http))
        hybrid.close()


class TestConstruction:
    def test_bad_candidate_pool(self) -> None:
        with pytest.raises(HybridError, match="candidate_pool"):
            HybridRetriever(_StubRetriever([]), _StubRetriever([]), candidate_pool=0)

    def test_bad_mmr_lambda(self) -> None:
        with pytest.raises(HybridError, match="mmr_lambda"):
            HybridRetriever(_StubRetriever([]), _StubRetriever([]), mmr_lambda=2.0)
