"""The declarative context builder: an ordered list of blocks, assembled into one prompt passage
under a character budget.

The list of blocks and the budget are config (``context.toml``); the assembler renders each block
in order, drops the ones with no content (no empty sections), and — if the result exceeds the
character budget — trims whole blocks by a configured priority rather than truncating text
mid-section. A block whose kind is not named in ``trim_order`` is never dropped, so the essential
sections survive any budget.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragkit.core.config import (
    ConfigError,
    load_toml,
    read_int,
    read_string_list,
    reject_unknown,
    tables,
)
from ragkit.core.ports import ContextBlock
from ragkit.core.records import Record

from .blocks import CONTEXT_BLOCKS


@dataclass(frozen=True, slots=True)
class _PlacedBlock:
    kind: str
    block: ContextBlock


class ContextAssembler:
    """Renders a record's configured context blocks into one passage, within a character budget.

    ``max_chars = 0`` disables the budget (every block that has content is shown). ``trim_order``
    lists block kinds most-trimmable-first; when the assembled text is over budget, sections of
    those kinds are dropped (last occurrence first) until it fits or none remain to drop.
    """

    def __init__(self, blocks: Sequence[_PlacedBlock], *, max_chars: int = 0,
                 trim_order: Sequence[str] = (), separator: str = "\n\n") -> None:
        self._blocks = tuple(blocks)
        self._max_chars = max_chars
        self._trim_priority = {kind: rank for rank, kind in enumerate(trim_order)}
        self._separator = separator

    def assemble(self, record: Record, context: Mapping[str, Any]) -> str:
        rendered: list[tuple[str, str]] = []
        for placed in self._blocks:
            text = placed.block.render(record, context)
            if text is not None and text.strip():
                rendered.append((placed.kind, text))
        rendered = self._fit(rendered)
        return self._separator.join(text for _kind, text in rendered)

    def _fit(self, rendered: list[tuple[str, str]]) -> list[tuple[str, str]]:
        if self._max_chars <= 0:
            return rendered
        while self._length(rendered) > self._max_chars:
            victim = self._most_trimmable(rendered)
            if victim is None:
                break  # nothing left that may be trimmed; the guaranteed sections stay
            del rendered[victim]
        return rendered

    def _length(self, rendered: list[tuple[str, str]]) -> int:
        if not rendered:
            return 0
        return (sum(len(text) for _kind, text in rendered)
                + len(self._separator) * (len(rendered) - 1))

    def _most_trimmable(self, rendered: list[tuple[str, str]]) -> int | None:
        # The trimmable section with the highest priority (lowest rank); ties broken toward the
        # last occurrence, so a repeated kind sheds its farthest section first.
        best: int | None = None
        best_rank = -1
        for index, (kind, _text) in enumerate(rendered):
            rank = self._trim_priority.get(kind)
            if rank is None:
                continue
            if best is None or rank <= best_rank:
                best, best_rank = index, rank
        return best


def load_context(path: Path) -> ContextAssembler:
    """Build a :class:`ContextAssembler` from a ``context.toml``: ``[[context.block]]`` entries in
    order, each selecting a registered block by ``kind`` and validated against that block's own
    options, plus an optional ``[context.budget]``."""
    data = load_toml(path, what="context file")
    reject_unknown(data, {"context"}, label="the context file", path=path)
    context = reject_unknown(data.get("context", {}), {"block", "budget"},
                             label="[context]", path=path)

    blocks: list[_PlacedBlock] = []
    for entry in tables(context, "block", path=path):
        kind = entry.get("kind")
        if not isinstance(kind, str) or not kind.strip():
            raise ConfigError("a [[context.block]] entry needs a 'kind'", path=path)
        options = {k: v for k, v in entry.items() if k != "kind"}
        block = CONTEXT_BLOCKS.create(kind, options, path=path)
        blocks.append(_PlacedBlock(kind=kind, block=block))
    if not blocks:
        raise ConfigError("no [[context.block]] entries defined", path=path)

    budget = reject_unknown(context.get("budget", {}), {"max_chars", "trim_order"},
                            label="[context.budget]", path=path)
    max_chars = read_int(budget, "max_chars", 0, label="[context.budget]", path=path)
    if max_chars < 0:
        raise ConfigError("[context.budget].max_chars must be >= 0", path=path)
    trim_order = read_string_list(budget, "trim_order", label="[context.budget]", path=path)
    return ContextAssembler(blocks, max_chars=max_chars, trim_order=trim_order)
