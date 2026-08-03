"""Import a reference JSONL corpus into a :class:`~ragkit.core.ports.PairingStore`, and build
retrievers over it.

Streams the corpus, batched, into one *co-located* :class:`~ragkit.core.ports.PairingStore` — the
row and its search entry are written in the same transaction and so can never drift apart (see
:mod:`ragkit.store.pairings.sqlite`) — with the store's own row count as the resumable floor. The
vector index, when configured, still lives separately (co-locating an ANN index is a different,
harder problem) and is kept in sync via :meth:`~ragkit.core.ports.VectorIndex.reconcile`.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from ragkit.core.errors import RagkitError
from ragkit.core.ports import Pairing, PairingStore, VectorIndex

from ..retrieve.embedding import EmbeddingClient, dedup_embed
from ..retrieve.hybrid import HybridRetriever
from ..retrieve.rerank import RerankClient
from ..retrieve.retrievers import DenseRetriever, LexicalRetriever


class ReferenceImportError(RagkitError):
    """The reference JSONL corpus is malformed."""


def reference_pairings(path: Path, *, index_field: str = "source", target_field: str = "target",
                       skip: int = 0) -> Iterator[Pairing]:
    """Yield one :class:`~ragkit.core.ports.Pairing` per non-blank JSONL line beyond ``skip``,
    numbered ``ref-<line>`` — a fetch script's own numbering, so citations, gold, and
    :mod:`ragkit.eval.retrieval` line up. ``skip`` fast-forwards past already-imported lines
    without parsing them, the resume path's mechanism. A line with no ``index_field`` value is
    skipped (blank source), never yielded as an empty pairing."""
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if line_no <= skip:
                continue
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReferenceImportError(
                    f"line {line_no}: invalid JSON in reference corpus: {exc}", path=path) from exc
            raw_source = record.get(index_field)
            # A JSON null (or absent) source is a blank source -- skipped, never yielded as "None".
            # str(None) used to be "None" (truthy), so the guard below never fired and a null field
            # was silently stored as the 4-char string "None".
            if raw_source is None:
                continue
            if not isinstance(raw_source, str):
                raise ReferenceImportError(
                    f"line {line_no}: {index_field!r} must be a string, "
                    f"got {type(raw_source).__name__}", path=path)
            if not raw_source:
                continue
            raw_target = record.get(target_field)
            if raw_target is None:  # absent or null -> a lexical-only entry (empty target)
                target = ""
            elif isinstance(raw_target, str):
                target = raw_target
            else:
                raise ReferenceImportError(
                    f"line {line_no}: {target_field!r} must be a string, "
                    f"got {type(raw_target).__name__}", path=path)
            yield Pairing(chunk_id=f"ref-{line_no}", source=raw_source, target=target, meta=record)


def import_reference(path: Path, pairing_store: PairingStore, *, index_field: str = "source",
                     target_field: str = "target", vector: VectorIndex | None = None,
                     embedder: EmbeddingClient | None = None, batch_size: int = 1000) -> int:
    """Stream ``path`` into ``pairing_store`` in batches, never holding the whole corpus in RAM.

    **Resumable**: the floor is ``pairing_store.count()`` (the durable state of the one co-located
    store — no separate stores to reconcile a tail against, unlike the legacy split-store path).
    A finished on-disk store resumes to a no-op; an in-memory one is always empty and rebuilds from
    scratch.

    If ``vector``/``embedder`` are given, each batch is embedded and upserted as it is imported,
    and :meth:`~ragkit.core.ports.VectorIndex.reconcile` runs once at the end — even on a no-op
    resume — so a gap a previous crash left between the pairings and the vector index (an upsert
    that never happened) is closed by re-embedding whatever ``reconcile`` reports missing.
    Returns the number of pairings actually added (excludes an already-imported line on resume,
    and a duplicate ``chunk_id`` within the corpus itself — see ``PairingStore.add``).
    """
    if vector is not None and embedder is None:
        raise ValueError("a vector index needs an embedder to import the reference corpus")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    floor = pairing_store.count()
    total = 0
    batch: list[Pairing] = []
    for pairing in reference_pairings(path, index_field=index_field, target_field=target_field,
                                      skip=floor):
        batch.append(pairing)
        if len(batch) >= batch_size:
            total += _flush(batch, pairing_store, vector, embedder)
            batch = []
    if batch:
        total += _flush(batch, pairing_store, vector, embedder)
    if vector is not None and embedder is not None:
        reconcile_vector(pairing_store, vector, embedder, batch_size=batch_size)
    return total


def _flush(batch: Sequence[Pairing], pairing_store: PairingStore, vector: VectorIndex | None,
          embedder: EmbeddingClient | None) -> int:
    added = pairing_store.add(batch)
    if vector is not None and embedder is not None:
        embed_and_upsert(batch, vector, embedder)
    return added


def embed_and_upsert(pairings: Sequence[Pairing], vector: VectorIndex,
                     embedder: EmbeddingClient) -> None:
    """Embed each pairing's source text (de-duplicated) and upsert into ``vector`` under its
    ``chunk_id``. Shared by the importer's per-batch upsert and by :func:`reconcile_vector` (and
    reused by :mod:`ragkit.ingest.writeback` for the same re-embed-what's-missing step).

    **No metadata is written to the vector index**, deliberately. This used to copy each
    pairing's entire JSON record across, and nothing ever read it: a search returns
    ``(chunk_id, score)``, and the retriever resolves display text and metadata through
    ``pairing_store.document()`` — which is the point of the co-located store, and what keeps
    one authoritative copy of a row rather than two that can drift. At corpus scale the copy was
    pure disk and write bandwidth (7.1M full records duplicated into the ANN store).

    The consequence to know about: a driver that *can* filter on metadata (Qdrant) has none to
    filter on through this path. Nothing in the framework passes ``where`` to a vector search
    today; a caller that wants payload filtering populates the index itself rather than paying
    for a copy on every import that needs it.
    """
    embedded = dedup_embed(embedder, [p.source for p in pairings])
    ids = [p.chunk_id for p in pairings]
    vecs = [embedded[p.source] for p in pairings]
    vector.upsert(ids, vecs, [{} for _ in pairings])


def reconcile_vector(pairing_store: PairingStore, vector: VectorIndex, embedder: EmbeddingClient,
                     *, batch_size: int = 1000,
                     on_batch: Callable[[int, int], None] | None = None,
                     compact_every: int | None = None,
                     on_compact: Callable[[], None] | None = None) -> int:
    """Close any gap between ``pairing_store`` (authoritative) and ``vector`` (derived): drop
    orphan vectors and re-embed whatever the store has that the index is missing. Safe to call
    even when nothing changed (an empty reconcile is a no-op) -- callers use it after any batch of
    additions so a previous crash's gap is always eventually closed. Returns how many ids were
    missing (and thus re-embedded).

    ``missing`` is embedded and upserted in chunks of ``batch_size``, not as one call: after a
    normal per-batch-synced import this set is small, but a caller reconciling from a wipe, a
    crash that lost a large tail, or a from-scratch backend migration can have ``missing`` be the
    *entire* corpus -- embedding and upserting millions of rows in one call would hold every one of
    their vectors in RAM at once, the same shape of bug as :meth:`LanceVectorIndex.indexed_ids`
    once did on the read side.

    **Memory profile of the reconcile itself.** The one unavoidable non-streamed allocation is the
    id set ``VectorIndex.reconcile`` returns, which at 7.1M short ids is a few hundred MB. Only
    one such set is built: the drivers consume their indexed ids as a stream and strike them off
    (see ``store.vector.common.reconcile_against``), where the obvious set-difference form used to
    hold the authoritative *and* indexed sets at once, roughly doubling the peak. ``sorted()`` on
    top adds a pointer array (~8 bytes per id, not another copy of the strings) and buys
    deterministic, resumable batch order, which is worth it.

    ``on_batch``, if given, is called with ``(done, total)`` once up front (``done=0``, so a
    long-running caller can report the full scope before any work happens) and again after each
    batch -- the one seam a caller needs to report progress on a multi-hour reconcile without this
    function reimplementing its own chunking loop just to add printing (see
    ``tools/embed_reference.sh``).

    ``compact_every``, if given, calls ``vector.compact()`` (silently skipped if ``vector`` has no
    such method -- a LanceDB-specific maintenance operation, not part of the general VectorIndex
    port) after every ``compact_every`` batches, announced via ``on_compact`` first if given. A
    table upserted in many small batches over a long run accumulates one on-disk fragment per
    batch without bound; confirmed directly at real corpus scale that this alone can cost multiple
    GB of RSS per subsequent batch once fragments number in the thousands. Periodic compaction
    during the run, not just once at the end, is what keeps that bounded. If ``compact()`` itself
    raises, the exception propagates uncaught -- the whole call aborts rather than silently
    skipping a failed compaction. Every batch up to that point already committed (embed-then-upsert
    happens before the compaction check), so nothing already embedded is lost, and a rerun resumes
    from exactly the remaining gap -- but the caller sees the failure immediately, not a silently
    degraded (uncompacted) run."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if compact_every is not None and compact_every < 1:
        raise ValueError(f"compact_every must be >= 1, got {compact_every}")
    missing = sorted(vector.reconcile(pairing_store.all_ids()))
    if on_batch is not None:
        on_batch(0, len(missing))
    compact = getattr(vector, "compact", None)
    for batches_done, i in enumerate(range(0, len(missing), batch_size), start=1):
        chunk_ids = missing[i:i + batch_size]
        pairings = [p for chunk_id in chunk_ids if (p := pairing_store.get(chunk_id)) is not None]
        if pairings:
            embed_and_upsert(pairings, vector, embedder)
        if on_batch is not None:
            on_batch(min(i + batch_size, len(missing)), len(missing))
        if compact is not None and compact_every is not None and batches_done % compact_every == 0:
            if on_compact is not None:
                on_compact()
            compact()
    return len(missing)


class PairingRetrievers:
    """Builds retrievers over a :class:`~ragkit.core.ports.PairingStore` (+ optional vector index/
    embedder). :func:`import_reference` is a module-level function, not a method here: a pairing
    store is already one complete store, so there is no separate lexical/document write path to
    batch together."""

    def __init__(self, pairing_store: PairingStore, *, vector: VectorIndex | None = None,
                embedder: EmbeddingClient | None = None) -> None:
        if vector is not None and embedder is None:
            raise ValueError("a vector index needs an embedder to build retrievers over it")
        self._pairings = pairing_store
        self._vector = vector
        self._embedder = embedder

    def resolve(self, chunk_id: str) -> str | None:
        doc = self._pairings.document(chunk_id)
        return doc[0] if doc is not None else None

    def resolve_meta(self, chunk_id: str) -> Mapping[str, Any]:
        doc = self._pairings.document(chunk_id)
        return doc[1] if doc is not None else {}

    def lexical_retriever(self, *, default_k: int = 10) -> LexicalRetriever:
        return LexicalRetriever(self._pairings, self.resolve, meta=self.resolve_meta,
                                default_k=default_k)

    def dense_retriever(self) -> DenseRetriever:
        if self._vector is None or self._embedder is None:
            raise ValueError("this store has no vector index/embedder; build it with both")
        return DenseRetriever(self._vector, self._embedder, self.resolve, meta=self.resolve_meta)

    def hybrid_retriever(self, *, reranker: RerankClient | None = None, candidate_pool: int = 40,
                         mmr_lambda: float = 0.7, lexical_min_score: float = 0.30,
                         dense_min_score: float = 0.55) -> HybridRetriever:
        return HybridRetriever(
            self.lexical_retriever(), self.dense_retriever(), reranker=reranker,
            candidate_pool=candidate_pool, mmr_lambda=mmr_lambda,
            lexical_min_score=lexical_min_score, dense_min_score=dense_min_score)
