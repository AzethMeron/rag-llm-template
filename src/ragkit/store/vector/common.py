"""What every :class:`~ragkit.core.ports.VectorIndex` driver shares: the structured error type and
the ``upsert`` argument contract. Kept in one place, like
:mod:`ragkit.store.pairings.common`, so two drivers of the same port cannot disagree about what a
valid call looks like — the swap property (``[vector].driver`` lancedb <-> qdrant with identical
behaviour) depends on the *rejections* matching, not only the successes.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

from ragkit.core.errors import RagkitError


class VectorIndexError(RagkitError):
    """A vector-index operation failed, or the driver's backend is unavailable."""


def reconcile_against(authoritative: Iterable[str], indexed: Iterator[str],
                      delete: Callable[[Sequence[str]], None]) -> set[str]:
    """Drop orphan vectors and report which authoritative ids the index is missing.

    Shared by both drivers, and written to hold **one** id set rather than two. The obvious form
    -- ``set(authoritative) - indexed_ids()`` plus ``indexed_ids() - set(authoritative)`` -- builds
    the full authoritative set *and* the full indexed set before diffing them, so a from-scratch
    reconcile of a 7.1M-row corpus (after a wipe, a migration, or a crash that lost a large tail)
    held roughly twice the necessary memory at peak. Here the indexed side is consumed as a
    stream: each id either strikes itself off the authoritative set or is collected as an orphan,
    and what remains in the set at the end is exactly what is missing. Orphans are normally few,
    so the second list is small in practice.

    ``delete`` is the driver's own; it is called once, with the orphans sorted, so the operation
    is deterministic.
    """
    missing = set(authoritative)
    orphans = [chunk_id for chunk_id in indexed if not _seen(missing, chunk_id)]
    if orphans:
        delete(sorted(orphans))
    return missing


def _seen(missing: set[str], chunk_id: str) -> bool:
    """True if ``chunk_id`` was in ``missing`` — removing it, since an id the index already holds
    is by definition not missing. False marks it an orphan."""
    if chunk_id in missing:
        missing.discard(chunk_id)
        return True
    return False


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
