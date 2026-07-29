"""Translation-specific mechanical validators — the code-decidable checks that are about
*translation* rather than about output in general.

Two ship here, both satisfying the :class:`~ragkit.core.ports.Validator` port and both honouring the
framework's discipline for a blocking check: they **abstain unless there is positive evidence** (a
line too short to judge is never flagged) and are **deterministic** (a blocking check must not
depend on a model's weights).

* :class:`EchoValidator` — the model handed the input back untranslated. A whole-line verbatim echo
  (after masking placeholders and casefolding) is unambiguous, so it is flagged; a partial overlap
  is left to the script check, because some tokens (names, numbers, "OK") legitimately survive.
* :class:`ScriptValidator` — for a different-script pair (Japanese/Chinese/Russian/... into a Latin
  target), the source's own letters surviving into the target is untranslated text. Needs no
  language profile, just the named source script.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from ragkit.core.records import Record
from ragkit.core.rules import Severity, Violation

_PLACEHOLDER = re.compile(r"\[\[\d+\]\]")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

# Code-point ranges per named script. Enough to cover the common different-script pairs; a custom
# range can be added by registering another validator.
_SCRIPTS: dict[str, tuple[tuple[int, int], ...]] = {
    "han": ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF)),
    "hiragana": ((0x3040, 0x309F),),
    "katakana": ((0x30A0, 0x30FF),),
    "japanese": ((0x3040, 0x30FF), (0x4E00, 0x9FFF)),
    "cyrillic": ((0x0400, 0x04FF),),
    "greek": ((0x0370, 0x03FF),),
    "arabic": ((0x0600, 0x06FF),),
    "hebrew": ((0x0590, 0x05FF),),
    "hangul": ((0xAC00, 0xD7A3), (0x1100, 0x11FF)),
    "devanagari": ((0x0900, 0x097F),),
    "thai": ((0x0E00, 0x0E7F),),
}


def available_scripts() -> list[str]:
    return sorted(_SCRIPTS)


def _normalise(text: str) -> str:
    stripped = _PLACEHOLDER.sub(" ", text)
    return " ".join(unicodedata.normalize("NFC", stripped).casefold().split())


class EchoValidator:
    """Flags an output that is a whole-line verbatim echo of the input (untranslated)."""

    CONFIG_KEYS = frozenset({"min_words"})

    def __init__(self, min_words: int = 2) -> None:
        # Below this the "translation" of, say, a proper noun may legitimately equal the source, so
        # abstain rather than flag on too little evidence.
        self._min_words = min_words

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> EchoValidator:
        return cls(min_words=int(options.get("min_words", 2)))

    def validate(self, record: Record, output: str,
                 context: Mapping[str, Any]) -> list[Violation]:  # noqa: ARG002
        source = _normalise(record.source)
        if len(_WORD.findall(source)) < self._min_words:
            return []  # too short to be sure it is an echo rather than a shared token
        if _normalise(output) == source:
            return [Violation("untranslated", Severity.ERROR,
                              "the output is a verbatim copy of the input; it was not translated")]
        return []


class ScriptValidator:
    """Flags source-script characters surviving into a target of a different script."""

    CONFIG_KEYS = frozenset({"source_script", "max_survivors"})

    def __init__(self, source_script: str, max_survivors: int = 0) -> None:
        if source_script not in _SCRIPTS:
            raise ValueError(
                f"unknown source_script {source_script!r}; known: {available_scripts()}")
        self._ranges = _SCRIPTS[source_script]
        self._script = source_script
        # A few stray source characters (a quoted name kept on purpose) are tolerated; more than
        # this is untranslated text.
        self._max = max_survivors

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> ScriptValidator:
        return cls(source_script=str(options.get("source_script", "")),
                   max_survivors=int(options.get("max_survivors", 0)))

    def _in_script(self, char: str) -> bool:
        code = ord(char)
        return any(low <= code <= high for low, high in self._ranges)

    def validate(self, record: Record, output: str,
                 context: Mapping[str, Any]) -> list[Violation]:  # noqa: ARG002
        # Only judge when the source actually is in this script (positive evidence); otherwise the
        # check does not apply to this record and must abstain rather than pass-or-fail blindly.
        if not any(self._in_script(c) for c in record.source):
            return []
        survivors = sum(1 for c in _PLACEHOLDER.sub(" ", output) if self._in_script(c))
        if survivors > self._max:
            return [Violation("untranslated", Severity.ERROR,
                              f"{survivors} {self._script} character(s) from the source remain in "
                              f"the output; it was not fully translated")]
        return []
