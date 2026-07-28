"""The metadata-filter compiler: correct SQL, escaping, and refusal at the boundary."""
from __future__ import annotations

import pytest

from ragkit.core.ports import FilterOp, Predicate
from ragkit.store.filters import FilterError, to_sql


def test_empty_filter_is_empty_string() -> None:
    assert to_sql(()) == ""


def test_equality_and_comparison() -> None:
    where = (Predicate("lang", FilterOp.EQ, "en"), Predicate("year", FilterOp.GE, 2020))
    assert to_sql(where) == "lang = 'en' AND year >= 2020"


@pytest.mark.parametrize("op,sql", [
    (FilterOp.NE, "!="), (FilterOp.LT, "<"), (FilterOp.LE, "<="), (FilterOp.GT, ">"),
])
def test_operators(op: FilterOp, sql: str) -> None:
    assert to_sql((Predicate("n", op, 1),)) == f"n {sql} 1"


def test_in_clause() -> None:
    assert to_sql((Predicate("id", FilterOp.IN, ["a", "b"]),)) == "id IN ('a', 'b')"


def test_string_quotes_are_escaped() -> None:
    assert to_sql((Predicate("name", FilterOp.EQ, "O'Brien"),)) == "name = 'O''Brien'"


def test_boolean_literal() -> None:
    assert to_sql((Predicate("ok", FilterOp.EQ, True),)) == "ok = 1"


def test_invalid_field_name_is_refused() -> None:
    with pytest.raises(FilterError, match="invalid filter field"):
        to_sql((Predicate("name; DROP TABLE", FilterOp.EQ, "x"),))


def test_empty_in_list_is_refused() -> None:
    with pytest.raises(FilterError, match="non-empty list"):
        to_sql((Predicate("id", FilterOp.IN, []),))


def test_unsupported_value_type_is_refused() -> None:
    with pytest.raises(FilterError, match="cannot compile a filter value"):
        to_sql((Predicate("x", FilterOp.EQ, {"a": 1}),))
