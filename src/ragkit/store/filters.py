"""Compile the framework's own metadata filter into a backend dialect.

The retrieval layer speaks one small, backend-neutral filter language — a conjunction of
:class:`~ragkit.core.ports.Predicate` (``field <op> value``) — so a caller never couples itself to
one store's dialect. Each store driver compiles it here. SQLite and LanceDB both accept a
SQL-shaped ``WHERE`` predicate, so they share :func:`to_sql`; a driver whose filter language is
different (a document store's JSON predicate) would provide its own compiler over the same AST. A
driver that cannot honour a predicate refuses it at compile time — at the boundary — rather than
returning wrong results at query time.
"""
from __future__ import annotations

from ragkit.core.errors import RagkitError
from ragkit.core.ports import Filter, FilterOp, Predicate

_SQL_OPS = {
    FilterOp.EQ: "=", FilterOp.NE: "!=", FilterOp.LT: "<", FilterOp.LE: "<=",
    FilterOp.GT: ">", FilterOp.GE: ">=",
}


class FilterError(RagkitError):
    """A filter predicate cannot be compiled for a store — a bad identifier or an unsupported
    value type. Raised at compile time so a wrong filter never silently returns wrong rows."""


def _sql_literal(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"  # standard SQL single-quote escaping
    raise FilterError(f"cannot compile a filter value of type {type(value).__name__}",
                      value=repr(value))


def _valid_identifier(field: str) -> str:
    # A field name reaches a SQL string un-parameterised, so it must be a plain identifier -- no
    # injection surface. Values are always literalised/escaped; identifiers are validated.
    if not field or not all(ch.isalnum() or ch == "_" for ch in field):
        raise FilterError(f"invalid filter field name {field!r}; use a plain identifier")
    return field


def _one(predicate: Predicate) -> str:
    field = _valid_identifier(predicate.field)
    if predicate.op is FilterOp.IN:
        values = predicate.value
        if not isinstance(values, (list, tuple)) or not values:
            raise FilterError(f"the IN filter on {field!r} needs a non-empty list of values")
        rendered = ", ".join(_sql_literal(v) for v in values)
        return f"{field} IN ({rendered})"
    sql_op = _SQL_OPS.get(predicate.op)
    if sql_op is None:  # pragma: no cover -- every non-IN FilterOp is in the table
        raise FilterError(f"unsupported filter op {predicate.op!r}")
    return f"{field} {sql_op} {_sql_literal(predicate.value)}"


def to_sql(where: Filter) -> str:
    """The ``Filter`` as a SQL ``WHERE`` predicate (without the ``WHERE`` keyword), or ``""`` when
    empty. Values are literalised and escaped; field names are validated as plain identifiers."""
    return " AND ".join(_one(predicate) for predicate in where)
