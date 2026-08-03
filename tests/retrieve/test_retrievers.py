"""The first-stage retrievers over real storage indexes."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from ragkit.core.ports import Pairing
from ragkit.retrieve.retrievers import (
    DenseRetriever,
    LexicalRetriever,
    RetrieverError,
    trigram_similarity,
)
from ragkit.store.pairings.sqlite import SqlitePairings
from ragkit.store.vector.lancedb import LanceVectorIndex

from .conftest import fake_embedder

TEXT = {"a": "the quick brown fox", "b": "a lazy sleeping dog", "c": "the quick red fox"}


def _resolve(chunk_id: str) -> str | None:
    return TEXT.get(chunk_id)


class TestLexical:
    def _index(self) -> SqlitePairings:
        index = SqlitePairings()
        index.add([Pairing(chunk_id=chunk_id, source=text) for chunk_id, text in TEXT.items()])
        return index

    def test_retrieves_matching_chunks(self) -> None:
        retriever = LexicalRetriever(self._index(), _resolve)
        hits = retriever.retrieve("quick fox", k=5)
        assert {h.chunk_id for h in hits} == {"a", "c"}
        assert all(h.text == TEXT[h.chunk_id] for h in hits)

    def test_min_score_floor(self) -> None:
        retriever = LexicalRetriever(self._index(), _resolve)
        assert retriever.retrieve("quick", k=5, min_score=1.1) == ()  # nothing clears >1

    def test_k_zero(self) -> None:
        assert LexicalRetriever(self._index(), _resolve).retrieve("quick", k=0) == ()

    def test_unresolvable_id_raises_because_the_store_is_co_located(self) -> None:
        # The lexical index and text store are the SAME PairingStore, whose contract says a hit and
        # its text cannot drift apart -- so an unresolvable hit is an internal invariant violation
        # (a corrupt store), raised loudly rather than silently dropped into a short result list.
        index = SqlitePairings()
        index.add([Pairing(chunk_id="ghost", source="the quick brown fox")])
        with pytest.raises(RetrieverError, match="internally inconsistent"):
            LexicalRetriever(index, lambda cid: None).retrieve("fox", k=5)


class TestDense:
    def _index(self, tmp_path: Path) -> LanceVectorIndex:
        index = LanceVectorIndex(str(tmp_path / "v"), dim=2)
        index.upsert(["a", "b"], [[1.0, 0.0], [0.0, 1.0]], [{}, {}])
        return index

    def test_retrieves_by_embedding_similarity(self, tmp_path: Path) -> None:
        embedder = fake_embedder(lambda t: [1.0, 0.0] if "fox" in t else [0.0, 1.0])
        retriever = DenseRetriever(self._index(tmp_path), embedder,
                                   lambda cid: {"a": "fox text", "b": "dog text"}.get(cid))
        hits = retriever.retrieve("a fox query", k=2)
        assert hits[0].chunk_id == "a"

    def test_k_zero(self, tmp_path: Path) -> None:
        embedder = fake_embedder(lambda t: [1.0, 0.0])
        assert DenseRetriever(self._index(tmp_path), embedder, _resolve).retrieve("x", k=0) == ()

    def test_unresolvable_id_is_dropped_with_a_warning(self, tmp_path: Path,
                                                       caplog: pytest.LogCaptureFixture) -> None:
        # The dense arm's vector index and pairing store are separate and can legitimately drift
        # between reconciles, so an unresolvable hit is dropped -- but never silently: a warning
        # names the drop count so a badly-stale index cannot quietly halve every result set.
        embedder = fake_embedder(lambda t: [1.0, 0.0])
        retriever = DenseRetriever(self._index(tmp_path), embedder, lambda cid: None)
        with caplog.at_level(logging.WARNING):
            hits = retriever.retrieve("x", k=2)
        assert hits == ()
        assert "dropped 2 of 2" in caplog.text and "stale" in caplog.text


def test_trigram_similarity() -> None:
    assert trigram_similarity("hello", "hello") == 1.0
    assert trigram_similarity("hello", "world") < 0.2
    assert trigram_similarity("", "x") == 0.0
    assert trigram_similarity("ab", "ab") == 1.0  # short strings
