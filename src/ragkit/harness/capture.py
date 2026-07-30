"""The context-capture sink: records what one :meth:`~ragkit.harness.agents.Harness.process` call
actually retrieved and assembled, without ever triggering a retrieval of its own.

A block or validator that already calls a :class:`~ragkit.core.ports.Retriever` (``RetrievedBlock``,
a recipe's grounding validator) writes its hits here through :func:`capture_retrieved`; nothing new
is fetched for capture's sake. This lives in its own module, separate from :mod:`.agents`, so both
:mod:`.agents` and :mod:`.context.blocks` can import it without a cycle (``agents`` builds the
context assembler, which is built from ``context.blocks``).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ragkit.core.ports import Retrieved


@dataclass(slots=True)
class Capture:
    """A fresh instance is created per :meth:`Harness.process` call and threaded through every
    round (produce, mechanical check); never shared across records, so it needs no lock.

    ``retrieved`` is keyed by ``chunk_id`` and merges across every call within one ``process()``
    rather than resetting per round: retrieval is deterministic on ``record.source``, which does not
    change across repair/revision rounds, so every call within one record contributes the same hits
    and a later write simply confirms an earlier one. ``passage`` instead reflects whichever call
    wrote it *last* — by construction that is always the final :meth:`Harness.produce` of the round
    that produced the returned outcome, since production always precedes settling and review's own
    prompt assembly does not write here (see ``Harness.review``).
    """

    passage: str = ""
    retrieved: dict[str, Retrieved] = field(default_factory=dict)

    def note_passage(self, passage: str) -> None:
        self.passage = passage

    def note_retrieved(self, hits: Sequence[Retrieved]) -> None:
        for hit in hits:
            self.retrieved[hit.chunk_id] = hit


def capture_retrieved(context: Mapping[str, Any], hits: Sequence[Retrieved]) -> None:
    """Record hits a context block or validator already retrieved into the run's capture sink, if
    one is wired into ``context`` under the ``"capture"`` key — a no-op otherwise (no capture wired,
    or no hits). Never performs a retrieval of its own."""
    capture = context.get("capture")
    if isinstance(capture, Capture) and hits:
        capture.note_retrieved(hits)
