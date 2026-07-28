"""The first-stage retrievers over real storage indexes."""
from __future__ import annotations

from pathlib import Path

from ragkit.retrieve.retrievers import DenseRetriever, LexicalRetriever, trigram_similarity
from ragkit.store.lexical.fts5 import Fts5Index
from ragkit.store.vector.lancedb import LanceVectorIndex

from .conftest import fake_embedder

TEXT = {"a": "the quick brown fox", "b": "a lazy sleeping dog", "c": "the quick red fox"}


def _resolve(chunk_id: str) -> str | None:
    return TEXT.get(chunk_id)


class TestLexical:
    def _index(self) -> Fts5Index:
        index = Fts5Index()
        for chunk_id, text in TEXT.items():
            index.index(chunk_id, text)
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

    def test_unresolvable_id_is_dropped(self) -> None:
        # The index knows 'a' but the text store has lost it: dropped, not surfaced empty.
        index = Fts5Index()
        index.index("ghost", "the quick brown fox")
        hits = LexicalRetriever(index, lambda cid: None).retrieve("fox", k=5)
        assert hits == ()


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


def test_trigram_similarity() -> None:
    assert trigram_similarity("hello", "hello") == 1.0
    assert trigram_similarity("hello", "world") < 0.2
    assert trigram_similarity("", "x") == 0.0
    assert trigram_similarity("ab", "ab") == 1.0  # short strings
