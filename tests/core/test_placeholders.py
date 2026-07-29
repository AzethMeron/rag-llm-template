"""The placeholder convention: every ``[[n]]`` index, in order."""
from __future__ import annotations

from ragkit.core.placeholders import placeholder_indices


def test_indices_in_order() -> None:
    assert placeholder_indices("a [[0]] b [[2]] c [[1]]") == [0, 2, 1]


def test_no_placeholders() -> None:
    assert placeholder_indices("plain text") == []


def test_repeated_index_is_reported_each_time() -> None:
    assert placeholder_indices("[[0]] and [[0]]") == [0, 0]


def test_multi_digit_indices() -> None:
    assert placeholder_indices("[[10]][[3]]") == [10, 3]
