"""The placeholder convention shared across the framework.

A source of records may replace the parts of a payload the model must not touch — engine
expressions, conditionals, escapes, inline markup, a column reference — with ``[[0]]``,
``[[1]]`` and so on, and restore them afterwards. The producer only has to preserve the
*set*; it never learns what any placeholder contained, which is what lets one engine serve
inputs whose syntaxes have nothing in common. The mechanical ``placeholders`` validator
enforces that the set is neither changed nor renumbered.
"""
from __future__ import annotations

import re

PLACEHOLDER = re.compile(r"\[\[(\d+)\]\]")


def placeholder_indices(text: str) -> list[int]:
    """Every placeholder index referenced by ``text``, in order of appearance."""
    return [int(match.group(1)) for match in PLACEHOLDER.finditer(text)]
