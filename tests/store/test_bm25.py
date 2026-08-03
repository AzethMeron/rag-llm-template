"""The BM25 score transform and FTS5 query-escaping helpers, shared by both pairing drivers."""
from __future__ import annotations

from ragkit.store.lexical.bm25 import as_match, bm25_to_relevance


class TestScoreTransform:
    def test_monotonic_higher_is_better(self) -> None:
        # The input is a raw BM25 magnitude: larger means a better match.
        assert bm25_to_relevance(5.0) > bm25_to_relevance(1.0)

    def test_bounded_in_unit_interval(self) -> None:
        for raw in (0.0, 1.0, 10.0, 100.0, 1e6):
            assert 0.0 <= bm25_to_relevance(raw) < 1.0

    def test_no_match_scores_zero(self) -> None:
        assert bm25_to_relevance(0.0) == 0.0

    def test_it_still_separates_strong_matches(self) -> None:
        # Why the rational form, not `1 - 2**-raw`: that saturates to 0.999 by raw=10, collapsing
        # every strong match onto one score so a high floor cannot tell them apart.
        assert bm25_to_relevance(20.0) - bm25_to_relevance(10.0) > 0.01


class TestMatchEscaping:
    def test_words_are_or_combined_and_quoted(self) -> None:
        assert as_match("quick fox") == '"quick" OR "fox"'

    def test_empty_query(self) -> None:
        assert as_match("   ") == '""'

    def test_embedded_quote_is_escaped(self) -> None:
        assert as_match('a"b') == '"a""b"'
