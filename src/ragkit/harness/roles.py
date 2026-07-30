"""Who the personas are and how much room they get, loaded from ``personas.toml``.

A **persona** is an LLM acting under a fixed system prompt — the producer that generates the
output, and the reviewers that judge it. This is distinct from the *harness* (the loop that
sequences them) and from a chat message's ``role`` field; the naming keeps all three apart.

Every persona names the logical model it runs on, so a run can put its cheap reviewers on a
small model and its producer on a larger one; personas naming the same model share it. The panel
consults reviewers in file order and stops at the first objection, so order them
most-decisive-first. A ``from_rules`` reviewer judges against the rule set's advisory criteria
rather than its own instructions, keeping policy in one place.
"""
from __future__ import annotations

import re
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from ragkit.core.config import (
    ConfigError,
    read_bool,
    read_float,
    read_int,
    read_string_list,
    reject_unknown,
    tables,
)
from ragkit.core.ports import SamplingParams

_TOKEN = re.compile(r"\{([a-z_][a-z0-9_]*)\}")
"""A substitution placeholder in an instruction. Deliberately narrow — lowercase identifiers
only — so prose containing braces is not mistaken for one."""

_PERSONA_KINDS = frozenset({"producer", "reviewer"})

_PRODUCER_DEFAULT_TEMPERATURE = 0.3
"""A little warmth for generation, unless the persona overrides it."""
_REVIEWER_DEFAULT_TEMPERATURE = 0.0
"""Determinism for judging: a reviewer's verdict should not wander between runs."""

# Revision-budget defaults, defined once so the Panel field default and the load_panel loader
# reference the same value rather than each re-typing the literal (the SSOT rule the config
# loaders already follow for Leniency).
_DEFAULT_MAX_REVISIONS = 2
_DEFAULT_MAX_REPAIRS = 2
_DEFAULT_REPAIR_TRUNCATED_JSON = True

# Derived from the dataclass, not re-typed: the allowed [persona.sampling] keys are exactly the
# SamplingParams fields, so adding a knob there cannot drift out of sync with what config accepts.
_SAMPLING_KEYS = frozenset(f.name for f in fields(SamplingParams))


def _at_least(value: int, minimum: int, *, what: str) -> None:
    if value < minimum:
        raise ValueError(f"{what} must be >= {minimum}, got {value}")


@dataclass(frozen=True, slots=True)
class Leniency:
    """How tolerant the *console* is of a reviewer's unusable replies before it warns.

    Every unusable reply is always recorded; this bounds only the *printed* warning, so a flaky
    reviewer neither hides a systemic failure nor buries it in noise. Within a rolling window of
    the last ``window`` replies, up to ``max_bad`` unusable ones pass without a console warning.
    """

    window: int = 20
    max_bad: int = 2

    def __post_init__(self) -> None:
        _at_least(self.window, 1, what="leniency window")
        _at_least(self.max_bad, 0, what="leniency max_bad")

    def surfaces(self, bad_in_window: int) -> bool:
        """Whether a bad reply, given the bad count now in the window, reaches the console."""
        return bad_in_window > self.max_bad


@dataclass(frozen=True, slots=True)
class Limits:
    """Output token budgets. The producer's scales with the input; each reviewer has a ceiling."""

    produce_tokens_per_source_char: int = 8
    produce_tokens_floor: int = 256
    produce_tokens_ceiling: int = 1024
    review_tokens: int = 1024

    def __post_init__(self) -> None:
        _at_least(self.produce_tokens_per_source_char, 1, what="produce_tokens_per_source_char")
        _at_least(self.produce_tokens_floor, 1, what="produce_tokens_floor")
        _at_least(self.produce_tokens_ceiling, 1, what="produce_tokens_ceiling")
        _at_least(self.review_tokens, 1, what="review_tokens")
        if self.produce_tokens_floor > self.produce_tokens_ceiling:
            raise ValueError(
                f"produce_tokens_floor ({self.produce_tokens_floor}) exceeds "
                f"produce_tokens_ceiling ({self.produce_tokens_ceiling}), so every budget would "
                f"be the ceiling")

    def produce_budget(self, source_length: int) -> int:
        """Token ceiling for producing from a source of ``source_length`` characters, bounded at
        both ends: the floor keeps a short input able to produce a full answer, the ceiling stops
        an unusually long one asking for an unbounded response."""
        scaled = source_length * self.produce_tokens_per_source_char
        return max(self.produce_tokens_floor, min(self.produce_tokens_ceiling, scaled))

    def review_budget(self, persona: Persona) -> int:
        return persona.max_tokens if persona.max_tokens is not None else self.review_tokens


_LIMITS_DEFAULTS = Limits()
"""One home for the token-budget defaults: the loader reads them from here rather than re-typing
each literal (matching the Leniency loader's ``default.window`` pattern)."""


@dataclass(frozen=True, slots=True)
class Persona:
    """One member of the run: the producer, or a reviewer."""

    id: str
    kind: str
    model: str
    instructions: str = ""
    from_rules: bool = False
    """Reviewer only: build the instructions from the rule set's advisory criteria instead of
    its own, keeping project policy in one file."""
    max_tokens: int | None = None
    leniency: Leniency = Leniency()
    sampling: SamplingParams = field(default_factory=SamplingParams)
    """The persona's per-request decode settings (temperature and the other sampling knobs). The
    loader fills a kind-appropriate default temperature; a directly-constructed persona gets the
    low neutral default."""

    def __post_init__(self) -> None:
        if self.kind not in _PERSONA_KINDS:
            raise ValueError(f"persona {self.id!r}: kind must be one of {sorted(_PERSONA_KINDS)}")
        if self.max_tokens is not None:
            _at_least(self.max_tokens, 1, what="max_tokens")

    @property
    def is_reviewer(self) -> bool:
        return self.kind == "reviewer"


@dataclass(frozen=True, slots=True)
class Panel:
    """The producer, the reviewers that judge it, and the budgets they operate under."""

    producer: Persona
    reviewers: tuple[Persona, ...]
    limits: Limits = field(default_factory=Limits)
    leniency: Leniency = field(default_factory=Leniency)
    max_revisions: int = _DEFAULT_MAX_REVISIONS
    max_repairs: int = _DEFAULT_MAX_REPAIRS
    repair_truncated_json: bool = _DEFAULT_REPAIR_TRUNCATED_JSON

    def __post_init__(self) -> None:
        _at_least(self.max_revisions, 0, what="max_revisions")
        _at_least(self.max_repairs, 0, what="max_repairs")

    def models_in_use(self) -> set[str]:
        """Every logical model named by a persona — the set the pool's thrash guard checks."""
        return {self.producer.model, *(r.model for r in self.reviewers)}


def load_panel(path: Path, substitutions: dict[str, str] | None = None) -> Panel:
    """Load the personas and budgets, filling any ``{placeholder}`` tokens by literal
    replacement (not ``str.format``, because instructions are prose that legitimately contains
    braces). An unfilled placeholder is an error, never a brace reaching the model."""
    if not path.is_file():
        raise ConfigError("personas file not found", path=path)
    try:
        data = tomllib.loads(path.read_text("utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML: {exc}", path=path) from exc

    reject_unknown(data, {"persona", "limits", "revision", "leniency"},
                   label="the personas file", path=path)
    resolved = dict(substitutions or {})

    def fill(text: str, *, where: str) -> str:
        for key, value in resolved.items():
            text = text.replace(f"{{{key}}}", value)
        if remaining := _TOKEN.findall(text):
            raise ConfigError(
                f"{where} refers to unknown placeholder(s) {sorted(set(remaining))}; known "
                f"placeholders are {sorted(resolved)}", path=path)
        return text

    default_leniency = _leniency(data.get("leniency", {}), Leniency(),
                                 label="[leniency]", path=path)
    limits = _limits(data.get("limits", {}), path=path)
    revision = reject_unknown(data.get("revision", {}),
                              {"max_revisions", "max_repairs", "repair_truncated_json"},
                              label="[revision]", path=path)
    producer, reviewers = _personas(data, fill, default_leniency=default_leniency, path=path)
    try:
        return Panel(
            producer=producer, reviewers=reviewers, limits=limits, leniency=default_leniency,
            max_revisions=read_int(revision, "max_revisions", _DEFAULT_MAX_REVISIONS,
                                   label="[revision]", path=path),
            max_repairs=read_int(revision, "max_repairs", _DEFAULT_MAX_REPAIRS,
                                 label="[revision]", path=path),
            repair_truncated_json=read_bool(revision, "repair_truncated_json",
                                            _DEFAULT_REPAIR_TRUNCATED_JSON,
                                            label="[revision]", path=path))
    except ValueError as exc:
        raise ConfigError(f"[revision]: {exc}", path=path) from exc


def _leniency(section: object, default: Leniency, *, label: str, path: Path) -> Leniency:
    table = reject_unknown(section, {"window", "max_bad"}, label=label, path=path)
    try:
        return Leniency(window=read_int(table, "window", default.window, label=label, path=path),
                        max_bad=read_int(table, "max_bad", default.max_bad, label=label, path=path))
    except ValueError as exc:
        raise ConfigError(f"{label}: {exc}", path=path) from exc


def _limits(section: object, *, path: Path) -> Limits:
    table = reject_unknown(section, {"produce_tokens_per_source_char", "produce_tokens_floor",
                                     "produce_tokens_ceiling", "review_tokens"},
                           label="[limits]", path=path)
    d = _LIMITS_DEFAULTS
    try:
        return Limits(
            produce_tokens_per_source_char=read_int(
                table, "produce_tokens_per_source_char", d.produce_tokens_per_source_char,
                label="[limits]", path=path),
            produce_tokens_floor=read_int(table, "produce_tokens_floor", d.produce_tokens_floor,
                                          label="[limits]", path=path),
            produce_tokens_ceiling=read_int(table, "produce_tokens_ceiling",
                                            d.produce_tokens_ceiling, label="[limits]", path=path),
            review_tokens=read_int(table, "review_tokens", d.review_tokens,
                                   label="[limits]", path=path))
    except ValueError as exc:
        raise ConfigError(f"[limits]: {exc}", path=path) from exc


def _personas(data: dict[str, Any], fill: Callable[..., str], *, default_leniency: Leniency,
              path: Path) -> tuple[Persona, tuple[Persona, ...]]:
    producers: list[Persona] = []
    reviewers: list[Persona] = []
    seen: set[str] = set()
    for entry in tables(data, "persona", path=path):
        identifier = entry.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ConfigError("a [[persona]] entry has no 'id'", path=path)
        if identifier in seen:
            raise ConfigError(f"duplicate persona id {identifier!r}", path=path)
        seen.add(identifier)
        reject_unknown(entry, {"id", "kind", "model", "instructions", "from_rules", "max_tokens",
                               "leniency", "sampling"}, label=f"persona {identifier!r}", path=path)
        persona = _one_persona(entry, identifier, fill, default_leniency=default_leniency,
                               path=path)
        (reviewers if persona.is_reviewer else producers).append(persona)

    if len(producers) != 1:
        raise ConfigError(
            f"exactly one persona must have kind='producer'; found {len(producers)}", path=path)
    if not reviewers:
        raise ConfigError(
            "no reviewer personas: every output would be accepted unread. Set "
            "revision.max_revisions to 0 if that is genuinely what you want", path=path)
    return producers[0], tuple(reviewers)


def _one_persona(entry: dict[str, Any], identifier: str, fill: Callable[..., str], *,
                 default_leniency: Leniency, path: Path) -> Persona:
    kind = entry.get("kind")
    if kind not in _PERSONA_KINDS:
        raise ConfigError(
            f"persona {identifier!r}: kind must be one of {sorted(_PERSONA_KINDS)}, got {kind!r}",
            path=path)
    model = entry.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ConfigError(f"persona {identifier!r} needs a 'model' (a logical model name)",
                          path=path)
    from_rules = read_bool(entry, "from_rules", False, label=f"persona {identifier!r}", path=path)
    if from_rules and kind != "reviewer":
        raise ConfigError(f"persona {identifier!r}: from_rules applies only to a reviewer",
                          path=path)
    text = entry.get("instructions", "")
    if not isinstance(text, str):
        raise ConfigError(f"persona {identifier!r}: instructions must be a string", path=path)
    if from_rules and text.strip():
        raise ConfigError(
            f"persona {identifier!r} sets both from_rules and instructions; one would be silently "
            f"ignored", path=path)
    text = fill(text, where=f"persona {identifier!r} instructions")
    if not from_rules and not text.strip():
        raise ConfigError(
            f"persona {identifier!r} has no instructions and does not set from_rules, so it would "
            f"act against nothing", path=path)
    leniency = (_leniency(entry["leniency"], default_leniency,
                          label=f"persona {identifier!r} leniency", path=path)
                if "leniency" in entry else default_leniency)
    max_tokens = (read_int(entry, "max_tokens", 1, label=f"persona {identifier!r}", path=path)
                  if "max_tokens" in entry else None)
    sampling = _sampling(entry.get("sampling", {}), kind, identifier=identifier, path=path)
    try:
        return Persona(id=identifier, kind=kind, model=model, instructions=text.strip(),
                       from_rules=from_rules, max_tokens=max_tokens, leniency=leniency,
                       sampling=sampling)
    except ValueError as exc:
        raise ConfigError(f"persona {identifier!r}: {exc}", path=path) from exc


def _sampling(section: object, kind: str, *, identifier: str, path: Path) -> SamplingParams:
    """Build a persona's decode settings from an optional ``[persona.sampling]`` sub-table. The
    default temperature depends on the role (a producer gets a little warmth, a reviewer none);
    every other knob is unset unless named, so only chosen settings ever reach the server."""
    label = f"persona {identifier!r} sampling"
    table = reject_unknown(section, _SAMPLING_KEYS, label=label, path=path)
    default_temp = (_PRODUCER_DEFAULT_TEMPERATURE if kind == "producer"
                    else _REVIEWER_DEFAULT_TEMPERATURE)

    def opt_float(key: str) -> float | None:
        return read_float(table, key, 0.0, label=label, path=path) if key in table else None

    def opt_int(key: str) -> int | None:
        return read_int(table, key, 0, label=label, path=path) if key in table else None

    try:
        return SamplingParams(
            temperature=read_float(table, "temperature", default_temp, label=label, path=path),
            top_p=opt_float("top_p"), top_k=opt_int("top_k"), min_p=opt_float("min_p"),
            seed=opt_int("seed"), presence_penalty=opt_float("presence_penalty"),
            frequency_penalty=opt_float("frequency_penalty"),
            repeat_penalty=opt_float("repeat_penalty"),
            stop=read_string_list(table, "stop", label=label, path=path))
    except ValueError as exc:
        raise ConfigError(f"{label}: {exc}", path=path) from exc
