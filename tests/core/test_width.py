"""Display width: the properties that matter more than any single value — additivity, and
that the same text measures the same however it is Unicode-normalised."""
from __future__ import annotations

import unicodedata

from hypothesis import given
from hypothesis import strategies as st

from ragkit.core.width import display_columns


class TestKnownWidths:
    def test_ascii_is_one_each(self) -> None:
        assert display_columns("abc") == 3

    def test_empty(self) -> None:
        assert display_columns("") == 0

    def test_cjk_wide_is_two_each(self) -> None:
        assert display_columns("日本") == 4

    def test_fullwidth_is_two(self) -> None:
        assert display_columns("Ａ") == 2  # U+FF21 FULLWIDTH LATIN A

    def test_combining_mark_adds_nothing(self) -> None:
        # 'e' + combining acute renders as one cell.
        assert display_columns("é") == 1

    def test_format_character_adds_nothing(self) -> None:
        assert display_columns("a​b") == 2  # ZERO WIDTH SPACE draws nothing

    def test_nfd_korean_measures_as_nfc(self) -> None:
        # The regression that motivated the jamo handling: NFD '한' must not measure double.
        nfc = "한"           # 한, precomposed
        nfd = unicodedata.normalize("NFD", nfc)
        assert display_columns(nfc) == display_columns(nfd) == 2

    def test_ambiguous_counts_one(self) -> None:
        # East-Asian Ambiguous is read as narrow (the conservative choice for a non-CJK terminal).
        assert display_columns("§") == 1  # SECTION SIGN, EAW=A

    def test_zwj_sequence_is_a_documented_over_count(self) -> None:
        family = "\U0001f468‍\U0001f469‍\U0001f466"
        assert display_columns(family) == 6  # three wide pictographs, joiners free


class TestProperties:
    @given(st.text(), st.text())
    def test_additive_over_concatenation(self, a: str, b: str) -> None:
        assert display_columns(a + b) == display_columns(a) + display_columns(b)

    @given(st.text())
    def test_nfc_and_nfd_measure_the_same(self, text: str) -> None:
        nfc = unicodedata.normalize("NFC", text)
        nfd = unicodedata.normalize("NFD", text)
        assert display_columns(nfc) == display_columns(nfd)

    @given(st.text())
    def test_width_is_never_negative(self, text: str) -> None:
        assert display_columns(text) >= 0
