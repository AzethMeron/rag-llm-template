"""What every :class:`~ragkit.core.ports.PairingStore` driver shares: the structured error type
and the display-text convention. Kept in one place so two drivers of the same port cannot compute a
hit's display text differently — the swap property (``[pairings].driver`` sqlite <-> duckdb with
identical retrieval behaviour) depends on it.
"""
from __future__ import annotations

from ragkit.core.errors import RagkitError


class PairingStoreError(RagkitError):
    """A pairing-store operation failed, or the pairings database is misconfigured."""


def pairing_display(source: str, target: str) -> str:
    """The text a retriever shows for a pairing: ``"source -> target"`` once the pairing has a
    target, else the source alone (a reference entry imported before it has one)."""
    return f"{source} -> {target}" if target else source
