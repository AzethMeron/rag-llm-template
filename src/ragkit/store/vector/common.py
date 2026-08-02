"""What every :class:`~ragkit.core.ports.VectorIndex` driver shares: the structured error type and
the ``upsert`` argument contract. Kept in one place, like
:mod:`ragkit.store.pairings.common`, so two drivers of the same port cannot disagree about what a
valid call looks like — the swap property (``[vector].driver`` lancedb <-> qdrant with identical
behaviour) depends on the *rejections* matching, not only the successes.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from ragkit.core.errors import RagkitError


class VectorIndexError(RagkitError):
    """A vector-index operation failed, or the driver's backend is unavailable."""


def validate_upsert(ids: Sequence[str], vectors: Sequence[Sequence[float]],
                    metas: Sequence[Mapping[str, Any]], *, dim: int) -> None:
    """Check one ``upsert`` call's arguments, raising :class:`VectorIndexError` on the first
    problem. Every driver calls this before touching its backend, so a bad call is refused at the
    port with the same message whichever driver is configured.

    The duplicate-id rule is the interesting one. ``upsert`` maps each id to *one* vector, so a
    batch naming the same id twice has no defined answer — and the two drivers resolved that
    ambiguity differently: Qdrant's deterministic point id made the last occurrence silently win,
    while LanceDB's ``merge_insert`` refuses it outright. Silently picking one is the worse half
    of that split (an id whose vector depends on batch order is unreproducible), so both refuse.
    """
    if not (len(ids) == len(vectors) == len(metas)):
        raise VectorIndexError(
            f"upsert got mismatched lengths: {len(ids)} ids, {len(vectors)} vectors, "
            f"{len(metas)} metas")
    for vector in vectors:
        if len(vector) != dim:
            raise VectorIndexError(
                f"a vector has dimension {len(vector)}, but the index is {dim}-d")
    repeated = sorted(chunk_id for chunk_id, seen in Counter(ids).items() if seen > 1)
    if repeated:
        raise VectorIndexError(
            f"upsert got {len(repeated)} id(s) more than once in a batch of {len(ids)}, so which "
            f"vector wins is undefined (e.g. {repeated[:5]}). De-duplicate the batch first.")
