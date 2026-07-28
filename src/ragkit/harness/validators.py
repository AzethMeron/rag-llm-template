"""Mechanical (code-decidable) checks over a produced output, and the pluggable-validator seam.

Two kinds meet here. The **built-in** checks are intrinsic to the harness and driven entirely by
the :class:`~ragkit.harness.rules.RuleSet` (emptiness, placeholder integrity, control characters,
forbidden patterns, established-term presence, column budget) — cheap, deterministic, and settled
before any GPU time. The **pluggable** checks are task-specific and satisfy the
:class:`~ragkit.core.ports.Validator` port, resolved through :data:`VALIDATORS`: an
untranslated-echo detector, a generated-SQL safety check, a form-field-type check. A
:class:`ValidatorPipeline` runs the built-ins and then the pluggables over one output.

Two disciplines the pluggable validators must honour, promoted to framework rules because they
were the hard-won part of the original engine: a blocking check must **abstain unless it has
positive evidence** (never fire on input too thin to judge), and a check that **cannot be
evaluated is skipped, never reported as passed**.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

from ragkit.core.lexicon import Entry, relevant_entries
from ragkit.core.placeholders import placeholder_indices
from ragkit.core.ports import Validator
from ragkit.core.records import Record
from ragkit.core.registry import Registry
from ragkit.core.rules import Severity, Violation
from ragkit.core.width import display_columns

from .rules import RuleSet

# Genuinely unrenderable control characters. Tab (09), line feed (0a) and carriage return (0d) are
# deliberately allowed here -- whether a *newline* is acceptable depends on the task (a single-line
# carrier forbids it; a SQL statement or a document does not), so that is a recipe's own validator,
# not a universal rule.
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

VALIDATORS: Registry[Validator] = Registry(
    "validator", Validator,  # type: ignore[type-abstract]
    entry_point_group="ragkit.validators")
"""The registry for pluggable, task-specific validators. Built-in mechanical checks are not
registered here -- they are driven by the RuleSet directly; this seam is for the extensions. The
port is passed to parameterise the registry (a Protocol, which mypy flags as abstract)."""


def _describe_control(character: str) -> str:
    return {"\x00": "NUL", "\x0b": "VERTICAL TAB", "\x0c": "FORM FEED",
            "\x1b": "ESCAPE", "\x7f": "DELETE"}.get(character, f"U+{ord(character):04X}")


def check_mechanical(record: Record, output: str, ruleset: RuleSet, *,
                     required_terms: Sequence[str] = (),
                     max_columns: int | None = None) -> list[Violation]:
    """Every code-decidable rule violation in ``output`` against ``ruleset``.

    ``max_columns`` is the record's own hard column budget (from ``record.meta``), distinct from
    the rule set's project-wide ``max_line_columns`` guideline: a hard budget blocks past its
    tolerance, the guideline only warns.
    """
    violations: list[Violation] = []
    source = record.source

    if ruleset.require_nonempty and source.strip() and not output.strip():
        violations.append(Violation("nonempty", Severity.ERROR,
                                    "output is empty but the input is not"))

    expected = set(placeholder_indices(source))
    actual = placeholder_indices(output)
    if set(actual) != expected:
        violations.append(Violation(
            "placeholders", Severity.ERROR,
            f"placeholder set changed: expected {sorted(expected)}, found {sorted(set(actual))}"))
    elif len(actual) != len(set(actual)):
        violations.append(Violation("placeholders", Severity.ERROR,
                                    f"a placeholder is repeated: {actual}"))

    if control := _CONTROL_CHARACTERS.search(output):
        violations.append(Violation(
            "control_character", Severity.ERROR,
            f"output contains a raw {_describe_control(control.group(0))} character"))

    violations.extend(_width_violations(output, ruleset, max_columns))

    for pattern, why in ruleset.forbidden_patterns:
        if re.search(pattern, output):
            violations.append(Violation(
                "forbidden", Severity.ERROR,
                f"matches forbidden pattern {pattern!r}" + (f": {why}" if why else "")))

    for term in required_terms:
        if term and term.lower() not in output.lower():
            violations.append(Violation(
                "lexicon", ruleset.lexicon_severity,
                f"established term {term!r} does not appear in the output"))

    return violations


def _width_violations(output: str, ruleset: RuleSet, max_columns: int | None) -> list[Violation]:
    if max_columns is not None:
        # A hard budget from the medium itself (reading time, a fixed cell). Measured over the
        # whole payload; a tolerated overshoot warns, past it blocks.
        width = display_columns(output)
        if width <= max_columns:
            return []
        allowed = int(max_columns * (1.0 + ruleset.max_columns_tolerance))
        over = width - max_columns
        if width > allowed:
            return [Violation("line_width", Severity.ERROR,
                              f"output is {width} characters, {over} over the {max_columns} this "
                              f"record has room for; say it more briefly")]
        return [Violation("line_width", Severity.WARNING,
                          f"output is {width} characters, {over} over the {max_columns} budget but "
                          f"within tolerance")]
    # The project guideline measures each line the producer chose to break, and only warns.
    result: list[Violation] = []
    for index, segment in enumerate(output.split("\\n")):
        width = display_columns(segment)
        if width > ruleset.max_line_columns:
            result.append(Violation(
                "line_width", Severity.WARNING,
                f"line {index + 1} is {width} columns, over the {ruleset.max_line_columns}-column "
                f"limit"))
    return result


class ValidatorPipeline:
    """Runs the built-in mechanical checks and then the pluggable validators over one output.

    The built-ins are driven by the ``ruleset`` and the ``lexicon`` (established terms are required
    to appear); the pluggables are task-specific components. ``lexicon_limit`` caps how many
    established terms a single output is required to carry, so a long input cannot demand an
    unbounded set. The shared context handed to each pluggable carries the ruleset, the required
    terms, and the record's column budget, so a validator need not recompute them.
    """

    def __init__(self, ruleset: RuleSet, lexicon: list[Entry] | None = None, *,
                 extra: Sequence[Validator] = (), lexicon_limit: int = 12) -> None:
        self.ruleset = ruleset
        self.lexicon = lexicon or []
        self.extra = tuple(extra)
        self.lexicon_limit = lexicon_limit

    def check(self, record: Record, output: str) -> list[Violation]:
        required = tuple(entry.rendering for entry in
                         relevant_entries(record.source, self.lexicon, limit=self.lexicon_limit))
        raw_budget = record.meta.get("max_columns")
        max_columns = raw_budget if isinstance(raw_budget, int) and raw_budget > 0 else None
        violations = check_mechanical(record, output, self.ruleset,
                                      required_terms=required, max_columns=max_columns)
        context = {"ruleset": self.ruleset, "required_terms": required, "max_columns": max_columns}
        for validator in self.extra:
            violations.extend(validator.validate(record, output, context))
        return violations


def blocking(violations: Sequence[Violation]) -> list[Violation]:
    return [v for v in violations if v.blocking]


def partition_on_exhaustion(violations: Sequence[Violation],
                            ruleset: RuleSet) -> tuple[list[Violation], list[Violation]]:
    """Split blocking violations into ``(reject, keep_flagged)`` by the rule set's
    ``keep_flagged_rules``. When the repair budget runs out, a keep-flagged blocker means the
    output is imperfect but still worth keeping (the alternative is showing nothing); a reject
    blocker means the output is unusable."""
    keep: list[Violation] = []
    reject: list[Violation] = []
    for violation in blocking(violations):
        (keep if violation.rule_id in ruleset.keep_flagged_rules else reject).append(violation)
    return reject, keep
