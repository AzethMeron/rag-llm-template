"""The generated-SQL safety validator — the failure surface a natural-language-to-SQL task must
guard, defence-in-depth, decided in code before the statement can reach a database.

The framework already runs the external store as a **read-only binding** (a write is refused at the
port). This validator is the second and third lines: it refuses anything that is not a single,
schema-bounded ``SELECT``, so a destructive or injected statement is caught as a blocking
:class:`~ragkit.core.rules.Violation` and the record is rejected rather than executed.

* **Parse, don't pattern-match the payload.** Comments are stripped first (so a keyword cannot hide
  behind ``--``), then the statement is required to be a single ``SELECT``/``WITH ... SELECT`` — no
  stacked statements, no DDL/DML keyword anywhere.
* **Schema-bounded.** Every referenced table must exist in the introspected schema; an unknown table
  is both an injection signal and a hallucination signal.
* **Executable (optional).** When the read-only store is wired, the statement is dry-run through
  ``EXPLAIN`` against it, catching a syntax error or an unknown column the text checks miss.

Abstains on empty output (the nonempty built-in owns that) and uses only the record and the wired
services — deterministic, never the model's opinion.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from ragkit.core.ports import SchemaIntrospector, SqlStore
from ragkit.core.records import Record
from ragkit.core.rules import Severity, Violation

_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_FORBIDDEN = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM|TRUNCATE|"
    r"GRANT|REVOKE|EXEC|EXECUTE|MERGE|CALL)\b", re.IGNORECASE)
_TABLE_REF = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
_CTE_NAME = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(", re.IGNORECASE)


def _strip_comments(sql: str) -> str:
    return _BLOCK_COMMENT.sub(" ", _LINE_COMMENT.sub(" ", sql))


def _error(message: str) -> list[Violation]:
    return [Violation("sql_unsafe", Severity.ERROR, message)]


class SqlSafetyValidator:
    """Refuses any generated SQL that is not a single, schema-bounded ``SELECT``."""

    CONFIG_KEYS = frozenset({"explain"})

    def __init__(self, explain: bool = True) -> None:
        # Whether to additionally dry-run the statement through EXPLAIN when a read-only store is
        # wired. On by default; a recipe with no store to explain against simply has none wired.
        self._explain = explain

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> SqlSafetyValidator:
        return cls(explain=bool(options.get("explain", True)))

    def validate(self, record: Record, output: str,  # noqa: ARG002  -- record unused; port shape
                 context: Mapping[str, Any]) -> list[Violation]:
        cleaned = _strip_comments(output).strip().rstrip(";").strip()
        if not cleaned:
            return []  # empty output is the nonempty check's business, not this one's

        statements = [s for s in _split_statements(_strip_comments(output)) if s.strip()]
        if len(statements) > 1:
            return _error(f"the output has {len(statements)} statements; only a single SELECT is "
                          f"allowed (stacked statements are refused)")

        head = cleaned.split(None, 1)[0].upper()
        if head not in ("SELECT", "WITH"):
            return _error(f"the output starts with {head!r}, not SELECT/WITH; only a read-only "
                          f"query is allowed")
        if match := _FORBIDDEN.search(cleaned):
            return _error(f"the output contains the forbidden keyword {match.group(0).upper()!r}; "
                          f"only a read-only SELECT is allowed")

        if unknown := self._unknown_tables(cleaned, context):
            return _error(f"the output references table(s) not in the schema: {sorted(unknown)}")

        return self._explain_check(cleaned, context)

    def _unknown_tables(self, sql: str, context: Mapping[str, Any]) -> set[str]:
        introspector = context.get("introspector")
        if not isinstance(introspector, SchemaIntrospector):
            return set()  # no schema wired: the table-existence check does not apply
        known = {name.lower() for name in introspector.schema()}
        ctes = {name.lower() for name in _CTE_NAME.findall(sql)}
        referenced = {name.lower() for name in _TABLE_REF.findall(sql)}
        return referenced - known - ctes

    def _explain_check(self, sql: str, context: Mapping[str, Any]) -> list[Violation]:
        store = context.get("sql_store")
        if not self._explain or not isinstance(store, SqlStore):
            return []
        try:
            store.query(f"EXPLAIN {sql}")
        except Exception as exc:  # noqa: BLE001 -- any store/parse error means the SQL is unusable
            return _error(f"the output failed to parse/plan against the database: {exc}")
        return []


def _split_statements(sql: str) -> list[str]:
    """Split on semicolons that are not inside a quoted string. A statement inside quotes keeps its
    semicolon, so ``'a;b'`` is one statement, while ``SELECT 1; DROP t`` is two."""
    statements: list[str] = []
    current: list[str] = []
    quote: str | None = None
    for char in sql:
        if quote:
            current.append(char)
            if char == quote:
                quote = None
        elif char in ("'", '"'):
            quote = char
            current.append(char)
        elif char == ";":
            statements.append("".join(current))
            current = []
        else:
            current.append(char)
    if "".join(current).strip():
        statements.append("".join(current))
    return statements
