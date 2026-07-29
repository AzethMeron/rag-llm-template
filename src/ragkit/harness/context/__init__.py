"""Declarative context building: an ordered list of registered blocks, assembled into one prompt
passage under a character budget. See :mod:`ragkit.harness.context.blocks` for the built-ins and
:mod:`ragkit.harness.context.assembler` for the budgeter and the config loader."""
from __future__ import annotations

from .assembler import ContextAssembler, load_context
from .blocks import (
    CONTEXT_BLOCKS,
    ContextBlockError,
    EstablishedBlock,
    LexiconBlock,
    LiteralBlock,
    NeighboursBlock,
    PreviousAttemptBlock,
    ReadingsBlock,
    RetrievedBlock,
    SchemaBlock,
    SqlRowsBlock,
)

__all__ = [
    "ContextAssembler", "load_context", "CONTEXT_BLOCKS", "ContextBlockError",
    "LiteralBlock", "LexiconBlock", "NeighboursBlock", "EstablishedBlock",
    "RetrievedBlock", "PreviousAttemptBlock", "SqlRowsBlock", "SchemaBlock", "ReadingsBlock",
]
