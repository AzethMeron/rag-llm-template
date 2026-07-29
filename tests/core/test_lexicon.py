"""The lexicon: term-to-rendering entries retrieved by literal occurrence, longest-first."""
from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.core.lexicon import (
    Entry,
    LexiconError,
    read_lexicon,
    relevant_entries,
    write_lexicon,
)


class TestEntry:
    def test_to_json_round_trips(self) -> None:
        entry = Entry(term="cat", rendering="kot", category="animal", entity_id=3)
        clone = Entry(**__import__("json").loads(entry.to_json()))
        assert clone == entry


class TestReadWrite:
    def test_round_trip(self, tmp_path: Path) -> None:
        entries = [Entry(term="a", rendering="x"), Entry(term="abc", rendering="y")]
        path = tmp_path / "lex.jsonl"
        assert write_lexicon(entries, path) == 2
        loaded = read_lexicon(path)
        assert {e.term for e in loaded} == {"a", "abc"}

    def test_missing_file_is_empty_not_an_error(self, tmp_path: Path) -> None:
        assert read_lexicon(tmp_path / "none.jsonl") == []

    def test_a_directory_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(LexiconError, match="not a regular file"):
            read_lexicon(tmp_path)

    def test_malformed_json_is_located(self, tmp_path: Path) -> None:
        path = tmp_path / "lex.jsonl"
        path.write_text("{bad}\n", encoding="utf-8")
        with pytest.raises(LexiconError, match="invalid JSON"):
            read_lexicon(path)

    def test_missing_required_fields_are_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "lex.jsonl"
        path.write_text('{"term": "a"}\n', encoding="utf-8")
        with pytest.raises(LexiconError, match="needs at least 'term' and 'rendering'"):
            read_lexicon(path)

    def test_unknown_field_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "lex.jsonl"
        path.write_text('{"term": "a", "rendering": "b", "bogus": 1}\n', encoding="utf-8")
        with pytest.raises(LexiconError):
            read_lexicon(path)

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "lex.jsonl"
        path.write_text('\n{"term": "a", "rendering": "b"}\n\n', encoding="utf-8")
        assert len(read_lexicon(path)) == 1


class TestRelevantEntries:
    def test_matches_substrings_longest_first(self) -> None:
        entries = [Entry(term="cat", rendering="kot"),
                   Entry(term="category", rendering="kategoria")]
        hits = relevant_entries("this category is broad", entries)
        assert [e.term for e in hits] == ["category", "cat"]

    def test_no_match_is_empty(self) -> None:
        assert relevant_entries("nothing here", [Entry(term="xyz", rendering="q")]) == []

    def test_limit_caps_the_result(self) -> None:
        entries = [Entry(term=c, rendering=c) for c in "abcde"]
        assert len(relevant_entries("a b c d e", entries, limit=2)) == 2

    def test_empty_term_never_matches(self) -> None:
        assert relevant_entries("anything", [Entry(term="", rendering="q")]) == []
