"""Configurable policy over an acceptable output, in the three tiers a rule set splits into.

* **numeric** (``[limits]``) — line width and its tolerance, whether an empty output is allowed;
* **pattern** (``[[forbidden]]``) — regexes an output must not match, each with a reason;
* **prose** (``[[advisory]]`` and ``[style]``) — criteria a reviewer persona judges, because
  judging them is exactly what a language model is for.

The pattern and numeric tiers are decided by code (:mod:`ragkit.harness.validators`) and cost no
GPU time; the prose tier reaches the review panel. Two further knobs generalise hard-won
translation behaviour: ``keep_flagged_rules`` names the blocking rules that mean "imperfect, but
still worth keeping" rather than "unusable" — kept and flagged when the repair budget runs out
instead of discarded (the ``on_exhausted`` distinction) — and ``lexicon_severity`` sets whether a
missing established term blocks or merely warns.
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ragkit.core.config import (
    ConfigError,
    read_bool,
    read_float,
    read_string,
    read_string_list,
    reject_unknown,
    tables,
)
from ragkit.core.rules import Severity

MIN_LINE_COLUMNS = 20
"""Below this a "line" cannot hold even a short clause, so a smaller budget is a mistyped value
rather than a tight one — and would reject every output offered against it."""


@dataclass(frozen=True, slots=True)
class RuleSet:
    """Policy governing an acceptable output."""

    max_line_columns: int = 110
    max_columns_tolerance: float = 0.12
    """How far an output may exceed a record's hard column budget and still be accepted. The
    budget is reading time or box space, not an exact cliff: a few per cent over is a marginal
    defect, while a rejected record is not rendered at all. Zero tolerance would trade a real
    defect for a worse one."""
    require_nonempty: bool = True
    forbidden_patterns: tuple[tuple[str, str], ...] = ()
    """``(regex, reason)`` pairs the output must not match — model failure modes (a preamble, a
    refusal, a label prefix), not domain rules."""
    style_directives: tuple[str, ...] = ()
    """Short directives injected into every persona's system prompt. Anything long or nuanced
    belongs in an advisory criterion, where a reviewer judges it."""
    advisory_rules: tuple[tuple[str, str], ...] = ()
    """``(id, description)`` criteria a ``from_rules`` reviewer judges, all in one pass."""
    lexicon_severity: Severity = Severity.WARNING
    keep_flagged_rules: frozenset[str] = field(default_factory=frozenset)
    """Blocking rules whose failure, once the repair budget is spent, keeps the output (flagged
    for review) rather than rejecting it — because the only alternative is showing nothing, which
    is worse. Empty means every blocking rule rejects on exhaustion."""

    def __post_init__(self) -> None:
        if self.max_line_columns < MIN_LINE_COLUMNS:
            raise ValueError(
                f"max_line_columns={self.max_line_columns} is implausibly small "
                f"(minimum {MIN_LINE_COLUMNS})")
        if not 0.0 <= self.max_columns_tolerance <= 1.0:
            raise ValueError(
                f"max_columns_tolerance must be a fraction in [0, 1], got "
                f"{self.max_columns_tolerance}")

    @classmethod
    def load(cls, path: Path) -> RuleSet:
        if not path.is_file():
            raise ConfigError("rules file not found", path=path)
        try:
            data = tomllib.loads(path.read_text("utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"invalid TOML: {exc}", path=path) from exc

        reject_unknown(data, {"limits", "lexicon", "style", "forbidden", "advisory"},
                       label="the rules file", path=path)
        limits = reject_unknown(
            data.get("limits", {}),
            {"max_line_columns", "max_columns_tolerance", "require_nonempty", "keep_flagged_rules"},
            label="[limits]", path=path)
        style = reject_unknown(data.get("style", {}), {"directives"}, label="[style]", path=path)
        lexicon = reject_unknown(data.get("lexicon", {}), {"severity"},
                                 label="[lexicon]", path=path)

        max_columns = limits.get("max_line_columns", 110)
        if not isinstance(max_columns, int) or isinstance(max_columns, bool):
            raise ConfigError(
                f"limits.max_line_columns must be an integer, got {type(max_columns).__name__} "
                f"{max_columns!r}", path=path)

        forbidden = _forbidden(tables(data, "forbidden", path=path), path=path)
        advisory = _advisory(tables(data, "advisory", path=path), path=path)
        severity_name = read_string(lexicon, "severity", "warning", label="[lexicon]", path=path)
        try:
            lexicon_severity = Severity(severity_name)
        except ValueError as exc:
            raise ConfigError(
                f"lexicon.severity must be 'error' or 'warning', got {severity_name!r}",
                path=path) from exc

        try:
            return cls(
                max_line_columns=max_columns,
                max_columns_tolerance=read_float(limits, "max_columns_tolerance", 0.12,
                                                 label="[limits]", path=path),
                require_nonempty=read_bool(limits, "require_nonempty", True,
                                           label="[limits]", path=path),
                forbidden_patterns=forbidden,
                style_directives=read_string_list(style, "directives", label="[style]", path=path),
                advisory_rules=advisory,
                lexicon_severity=lexicon_severity,
                keep_flagged_rules=frozenset(
                    read_string_list(limits, "keep_flagged_rules", label="[limits]", path=path)))
        except ValueError as exc:
            raise ConfigError(f"[limits]: {exc}", path=path) from exc


def _forbidden(entries: list[dict], *, path: Path) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for entry in entries:
        reject_unknown(entry, {"pattern", "reason"}, label="[[forbidden]]", path=path)
        pattern, why = entry.get("pattern"), entry.get("reason", "")
        if not pattern or not isinstance(pattern, str):
            raise ConfigError("a [[forbidden]] entry needs a non-empty 'pattern'", path=path)
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ConfigError(f"forbidden pattern {pattern!r} is not a valid regex: {exc}",
                              path=path) from exc
        result.append((pattern, str(why)))
    return tuple(result)


def _advisory(entries: list[dict], *, path: Path) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for entry in entries:
        reject_unknown(entry, {"id", "description"}, label="[[advisory]]", path=path)
        rule_id, description = entry.get("id"), entry.get("description")
        if not rule_id or not description:
            raise ConfigError("an [[advisory]] entry needs 'id' and 'description'", path=path)
        result.append((str(rule_id), str(description)))
    return tuple(result)
