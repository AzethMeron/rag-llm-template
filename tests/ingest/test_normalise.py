"""Text normalisation: Unicode form folding, whitespace collapsing, dehyphenation."""
from __future__ import annotations

import pytest

from ragkit.ingest.normalise import normalise


class TestNormalise:
    def test_default_collapses_whitespace_and_strips(self) -> None:
        assert normalise("  hello   world  \n") == "hello world"

    def test_collapse_whitespace_can_be_disabled(self) -> None:
        assert normalise("a   b", collapse_whitespace=False) == "a   b"

    def test_nfc_folds_combining_characters(self) -> None:
        # "e" + combining acute (2 codepoints) NFC-folds to the single precomposed "é".
        decomposed = "é"
        assert normalise(decomposed) == "é"

    def test_nfkc_form(self) -> None:
        # A compatibility ligature ("fi") is folded apart under NFKC, not under plain NFC.
        assert normalise("ﬁ", form="NFKC") == "fi"
        assert normalise("ﬁ", form="NFC") == "ﬁ"

    def test_dehyphenate_rejoins_a_line_break_split_word(self) -> None:
        assert normalise("foo-\nbar", dehyphenate=True, collapse_whitespace=False) == "foobar"

    def test_dehyphenate_disabled_by_default(self) -> None:
        assert normalise("foo-\nbar", collapse_whitespace=False) == "foo-\nbar"

    def test_unknown_form_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown Unicode normalisation form"):
            normalise("x", form="bogus")
