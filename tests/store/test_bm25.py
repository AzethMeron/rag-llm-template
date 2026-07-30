"""The BM25 score transform and FTS5 query-escaping helpers, shared by every FTS5-backed driver."""
from __future__ import annotations

from ragkit.store.lexical.bm25 import as_match, bm25_to_relevance


class TestScoreTransform:
    def test_monotonic_higher_is_better(self) -> None:
        # More-negative bm25 (a better match) maps to a higher relevance.
        assert bm25_to_relevance(-5.0) > bm25_to_relevance(-1.0)

    def test_bounded_in_unit_interval(self) -> None:
        for bm25 in (-100.0, -10.0, -1.0, 0.0):
            assert 0.0 <= bm25_to_relevance(bm25) <= 1.0


class TestMatchEscaping:
    def test_words_are_or_combined_and_quoted(self) -> None:
        assert as_match("quick fox") == '"quick" OR "fox"'

    def test_empty_query(self) -> None:
        assert as_match("   ") == '""'

    def test_embedded_quote_is_escaped(self) -> None:
        assert as_match('a"b') == '"a""b"'
