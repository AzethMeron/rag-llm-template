"""Strict TOML loading primitives — the single source of truth for how config is parsed.

Every config file in the framework (personas, rules, context, retrieval, models, storage)
and every third-party component's own option block is read through these helpers, so the
guarantees hold uniformly rather than being re-implemented, slightly differently, per
loader:

* **Unknown keys are errors, not silently-ignored typos.** A mistyped ``max_reivsions``
  must not leave its default quietly in force — the operator would never learn their setting
  had no effect. :func:`reject_unknown` refuses it, naming the allowed keys.
* **A ``bool`` where a number is declared is refused, never coerced.** Python makes ``bool``
  a subclass of ``int``, so ``= true`` would otherwise read as ``1`` — turning a token budget
  or a context-window size into a plausible-but-wrong value. :func:`read_int` /
  :func:`read_float` reject it.
* **The ``[[section]]`` vs ``[section]`` mistake is named explicitly**, because the failure
  it otherwise produces — iterating a table yields its *keys*, so each "entry" is a bare
  string — points nowhere near the real error.
* **Range checks are deliberately left to the receiving dataclass**, so a directly
  constructed object is exactly as impossible to make invalid as a loaded one. These helpers
  answer only "is it the right *type* and a known *key*".
"""
from __future__ import annotations

import tomllib
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import Any

from .errors import LocatedError


class ConfigError(LocatedError):
    """A configuration file is missing, malformed, or internally inconsistent.

    Carries the ``path`` and, where known, the ``label`` of the section at fault, so the
    message points the operator at the exact place to fix rather than at a bare traceback.
    """

    def __init__(self, reason: str, *, path: Path | None = None,
                 label: str | None = None) -> None:
        super().__init__(reason, path=path, label=label)
        self.label = label


def load_toml(path: Path, *, what: str = "configuration file") -> dict[str, Any]:
    """Read and parse a TOML file, turning both failure modes into a :class:`ConfigError`.

    A missing file and a syntactically invalid one are different problems and say so, each
    naming the path — neither surfaces as a raw ``FileNotFoundError`` or ``TOMLDecodeError``
    the caller would have to translate.
    """
    if not path.is_file():
        raise ConfigError(f"{what} not found", path=path)
    try:
        return tomllib.loads(path.read_text("utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML: {exc}", path=path) from exc


def as_table(value: object, *, label: str, path: Path | None = None) -> dict[str, Any]:
    """The value as a TOML table (dict), or a :class:`ConfigError` naming what it actually is.

    Answered once, centrally, so the "is this even a table?" question does not resurface as a
    bare ``TypeError`` deep inside a loader when someone writes ``limits = 5`` instead of
    ``[limits]``.
    """
    if not isinstance(value, dict):
        raise ConfigError(
            f"{label} must be a table, got {type(value).__name__} {value!r}",
            path=path, label=label)
    return value


def reject_unknown(section: object, allowed: AbstractSet[str], *, label: str,
                   path: Path | None = None) -> dict[str, Any]:
    """The section as a table, with any key outside ``allowed`` refused.

    Refused, not dropped: a typo that silently left a default in force is the exact silent
    failure the project's rules forbid.
    """
    table = as_table(section, label=label, path=path)
    unknown = set(table) - allowed
    if unknown:
        raise ConfigError(
            f"{label} has unknown key(s) {sorted(unknown)}; allowed keys are {sorted(allowed)}",
            path=path, label=label)
    return table


def tables(data: dict[str, Any], key: str, *, path: Path | None = None) -> list[dict[str, Any]]:
    """The entries of an array-of-tables (``[[key]]``), each validated to be a table.

    Writing ``[key]`` for ``[[key]]`` is the likeliest structural mistake in a config file,
    and it is named explicitly because the failure it used to produce — iterating a table
    yields its keys, so each "entry" was a bare string — pointed nowhere near the cause.
    """
    entries = data.get(key, [])
    if not isinstance(entries, list):
        raise ConfigError(
            f"[[{key}]] must be an array of tables -- write [[{key}]], not [{key}] -- got "
            f"{type(entries).__name__}", path=path, label=f"[[{key}]]")
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ConfigError(
                f"[[{key}]] entry {position} must be a table, got {type(entry).__name__} "
                f"{entry!r}", path=path, label=f"[[{key}]]")
    return entries


def read_int(section: dict[str, Any], key: str, default: int, *, label: str,
             path: Path | None = None) -> int:
    """A TOML integer, refused rather than coerced when mistyped.

    ``bool`` is excluded despite being an ``int`` subclass: ``= true`` reading as ``1`` would
    silently turn "one token" or "one unit of context" into a plausible wrong value. Range is
    left to the dataclass that receives the value, so the bound has one home, not two.
    """
    value = section.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(
            f"{label}.{key} must be an integer, got {type(value).__name__} {value!r}",
            path=path, label=label)
    return value


def read_float(section: dict[str, Any], key: str, default: float, *, label: str,
               path: Path | None = None) -> float:
    """A TOML number, refused rather than coerced when mistyped.

    An ``int`` is accepted and widened (``0`` and ``1`` are natural values for a fraction),
    but ``bool`` is not: ``= true`` reading as ``1.0`` would silently turn a similarity floor
    into "accept only exact matches". Range, as with :func:`read_int`, is the dataclass's.
    """
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            f"{label}.{key} must be a number, got {type(value).__name__} {value!r}",
            path=path, label=label)
    return float(value)


def read_bool(section: dict[str, Any], key: str, default: bool, *, label: str,
              path: Path | None = None) -> bool:
    """A TOML boolean, refused rather than coerced.

    ``bool("false")`` is ``True``, so coercion here could silently *invert* a safety gate;
    a mistyped boolean is a config error, not a truthiness puzzle.
    """
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(
            f"{label}.{key} must be a boolean, got {type(value).__name__} {value!r}",
            path=path, label=label)
    return value


def read_string(section: dict[str, Any], key: str, default: str, *, label: str,
                path: Path | None = None) -> str:
    """A TOML string, refused rather than coerced."""
    value = section.get(key, default)
    if not isinstance(value, str):
        raise ConfigError(
            f"{label}.{key} must be a string, got {type(value).__name__} {value!r}",
            path=path, label=label)
    return value


def read_string_list(section: dict[str, Any], key: str, *, label: str,
                     path: Path | None = None) -> tuple[str, ...]:
    """A TOML array of strings.

    A bare string is refused because ``tuple("abc")`` would silently split it into
    ``("a", "b", "c")`` — one-character entries — rather than failing.
    """
    value = section.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(
            f"{label}.{key} must be an array of strings, got {value!r}", path=path, label=label)
    return tuple(value)
