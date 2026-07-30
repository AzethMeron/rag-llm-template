"""Fold a finished run's outputs into the reference memory as new pairings.

A deliberate, separate post-run step (never automatic, per the storage-overhaul plan): a run is
over, its :class:`~ragkit.core.ports.RunStore` holds the results, and write-back reads them,
keeps the ones trustworthy enough to become a future run's worked example, and writes them through
a :class:`~ragkit.core.ports.Sink` — closing the loop from a produced ``(source, context, target)``
triple back into what a later run can retrieve.
"""
from __future__ import annotations

from dataclasses import replace

from ragkit.core.ports import PairingStore, RunStore, Sink, VectorIndex
from ragkit.core.records import Status

from ..retrieve.embedding import EmbeddingClient
from .reference import reconcile_vector
from ..store.pairings.sink import PairingSink


def write_back(run_store: RunStore, sink: Sink, pairing_store: PairingStore, *,
               vector: VectorIndex | None = None, embedder: EmbeddingClient | None = None) -> int:
    """Read ``run_store.results()``, keep the ``VERIFIED`` ones — the conservative bar for what is
    trusted enough to teach a later run; ``PRODUCED`` covers an output kept *despite* a flagged
    rule or an incomplete review, not something write-back should reinforce — fold each result's
    captured context into its record's meta (``Sink.write``'s signature is ``Iterable[Record]``, a
    task-agnostic core port; ``context_passage`` lives on ``RunResult``, not ``Record``, so it
    travels this way — see :class:`~ragkit.store.pairings.sink.PairingSink`), and hand them to
    ``sink``.

    ``sink`` and ``pairing_store`` are both required — a generic :class:`Sink` is the injectable
    reinjection seam (``write`` returns nothing to report what changed), while ``pairing_store`` is
    needed directly here to measure how many pairings were actually added and to drive
    ``reconcile``; the caller must pass a ``sink`` that writes into this same ``pairing_store``, or
    the count reported is meaningless.

    If ``vector``/``embedder`` are given, :func:`~ragkit.ingest.reference.reconcile_vector` runs
    afterward, so a later dense/hybrid retrieval sees the new pairings too. Returns the number of
    pairings actually added.
    """
    if vector is not None and embedder is None:
        raise ValueError("a vector index needs an embedder to reconcile after write-back")
    before = pairing_store.count()
    enriched = (
        replace(result.record, meta={**result.record.meta,
                                     PairingSink.CONTEXT_META_KEY: result.context_passage})
        for result in run_store.results() if result.record.status is Status.VERIFIED)
    sink.write(enriched)
    added = pairing_store.count() - before
    if vector is not None and embedder is not None:
        reconcile_vector(pairing_store, vector, embedder)
    return added
