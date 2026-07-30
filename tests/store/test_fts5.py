"""The FTS5 lexical index: ranking, the score convention, deletion, and query escaping."""
from __future__ import annotations

import pytest

from ragkit.store.lexical.fts5 import Fts5Index


def _index() -> Fts5Index:
    index = Fts5Index()
    index.index("doc1", "the quick brown fox jumps")
    index.index("doc2", "a slow green turtle walks")
    index.index("doc3", "the quick red fox runs")
    return index


class TestSearch:
    def test_matches_and_ranks_higher_is_better(self) -> None:
        results = _index().search("quick fox", k=5)
        ids = [chunk_id for chunk_id, _score in results]
        assert set(ids) == {"doc1", "doc3"}  # doc2 shares no term
        assert all(score >= 0 for _id, score in results)

    def test_scores_are_ordered(self) -> None:
        results = _index().search("quick", k=5)
        scores = [score for _id, score in results]
        assert scores == sorted(scores, reverse=True)  # best first

    def test_empty_query_returns_nothing(self) -> None:
        assert _index().search("   ", k=5) == []

    def test_k_zero_returns_nothing(self) -> None:
        assert _index().search("quick", k=0) == []

    def test_no_match_returns_nothing(self) -> None:
        assert _index().search("elephant", k=5) == []

    def test_punctuation_in_query_does_not_raise(self) -> None:
        # A bare colon or quote would be FTS5 query syntax; the driver quotes each term.
        assert _index().search('fox: "quick"', k=5) is not None


class TestMutation:
    def test_reindex_is_idempotent(self) -> None:
        index = Fts5Index()
        index.index("d", "first text")
        index.index("d", "second text")  # replaces
        assert index.count() == 1
        assert index.search("second", k=1) and not index.search("first", k=1)

    def test_delete(self) -> None:
        index = _index()
        index.delete("doc1")
        assert index.count() == 2
        assert all(chunk_id != "doc1" for chunk_id, _ in index.search("quick", k=5))


class TestBulkIngest:
    def test_index_many_writes_searchable_rows(self) -> None:
        index = Fts5Index()
        written = index.index_many([("a", "the quick fox"), ("b", "a lazy dog")])
        assert written == 2
        assert index.count() == 2
        assert index.search("fox", k=5)[0][0] == "a"

    def test_index_many_empty_writes_nothing(self) -> None:
        assert Fts5Index().index_many([]) == 0


class TestConfig:
    def test_from_config(self) -> None:
        index = Fts5Index.from_config({"tokenizer": "porter"})
        index.index("d", "running quickly")
        assert index.search("run", k=1)  # porter stems 'running' -> 'run'


class TestErrors:
    def test_bad_tokenizer_is_a_structured_error(self) -> None:
        from ragkit.store.lexical.fts5 import LexicalIndexError
        with pytest.raises(LexicalIndexError, match="FTS5"):
            Fts5Index(tokenizer="no_such_tokenizer")

    def test_query_on_a_closed_index_is_structured(self) -> None:
        from ragkit.store.lexical.fts5 import LexicalIndexError
        index = _index()
        index.close()
        with pytest.raises(LexicalIndexError, match="query failed"):
            index.search("quick", k=1)
