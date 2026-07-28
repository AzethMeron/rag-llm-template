"""The FTS5 lexical index: ranking, the score convention, deletion, and query escaping."""
from __future__ import annotations

import pytest

from ragkit.store.lexical.fts5 import Fts5Index, _as_match, _bm25_to_relevance


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


class TestScoreTransform:
    def test_monotonic_higher_is_better(self) -> None:
        # More-negative bm25 (a better match) maps to a higher relevance.
        assert _bm25_to_relevance(-5.0) > _bm25_to_relevance(-1.0)

    def test_bounded_in_unit_interval(self) -> None:
        for bm25 in (-100.0, -10.0, -1.0, 0.0):
            assert 0.0 <= _bm25_to_relevance(bm25) <= 1.0


class TestMatchEscaping:
    def test_words_are_or_combined_and_quoted(self) -> None:
        assert _as_match("quick fox") == '"quick" OR "fox"'

    def test_empty_query(self) -> None:
        assert _as_match("   ") == '""'

    def test_embedded_quote_is_escaped(self) -> None:
        assert _as_match('a"b') == '"a""b"'


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
