"""The default vector index: LanceDB — a real embedded vector database with true ANN.

LanceDB gives genuine ANN indexing, on-disk columnar storage with versioning, metadata filtering
and hybrid search, all in-process with no server or daemon. It is the framework's default because
it satisfies "a real vector database" while keeping the self-contained rule (pip-only, nothing to
launch).

``lancedb`` and ``pyarrow`` are imported **lazily, inside this one module** (vectors are handled as
plain Python lists, so no numpy), so the core import path never pulls them and
``tests/test_boundaries.py`` can confine the allowance here. The
cosine metric returns a *distance* (``0`` identical, ``2`` opposite); the
:class:`~ragkit.core.ports.VectorIndex` port promises a higher-is-better score toward ``[0, 1]``,
so the conversion happens here.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import timedelta
from typing import Any

from ragkit.core.ports import Filter

from ..filters import to_sql
from .common import VectorIndexError, reconcile_against, validate_upsert

_NPROBE_FRACTION = 0.05
"""Share of the IVF partitions a query probes by default. 5% is the usual operating point where
IVF recall is close to exact while still reading a small slice of the corpus; see
:meth:`LanceVectorIndex.search_nprobes` for why this is a fraction and not a fixed count."""

_MIN_NPROBES = 20
"""Floor for the derived default, matching LanceDB's own default, so adding this never probes a
small table *less* than not having it would have."""


__all__ = ["LanceVectorIndex", "VectorIndexError"]  # VectorIndexError re-exported from .common


_FILTERABLE_FIELDS = frozenset({"id"})
"""The only real columns this table has, besides the vector itself. Metadata lives in one opaque
JSON string (``meta``), so there is nothing for a predicate on a metadata key to resolve against."""


def _supported(where: Filter) -> Filter:
    """``where``, or a structured refusal — at the boundary, as :class:`VectorIndex` requires.

    A predicate on a metadata key used to compile to a column this table does not have, and
    LanceDB then failed at query time with a raw engine error about an unknown field. Refusing
    here says what is actually wrong and what to do about it. Qdrant *can* filter on metadata (it
    stores each key as a payload field), so this is a genuine capability difference between the
    drivers — made explicit rather than discovered mid-query.

    Not fixable by storing metadata as columns: which keys exist varies per record, and LanceDB
    needs a fixed schema at table creation. LanceDB also offers no JSON-path filter to reach into
    the ``meta`` string (checked directly: ``json_extract``/``get_json_object`` are unavailable).
    """
    unsupported = sorted({p.field for p in where if p.field not in _FILTERABLE_FIELDS})
    if unsupported:
        raise VectorIndexError(
            f"the LanceDB vector index cannot filter on {unsupported}: metadata is stored as one "
            f"opaque JSON column, so only {sorted(_FILTERABLE_FIELDS)} can be filtered. Use the "
            f"'qdrant' driver, which stores each metadata key as a filterable payload field.")
    return where


def _ivf_partitions(row_count: int) -> int:
    """The standard IVF heuristic: ``~sqrt(n)`` partitions, so each holds ``~sqrt(n)`` vectors.
    One home for it, because the default ``nprobes`` is a fraction of what ``create_index``
    actually built — if the two ever disagreed, search would silently under- or over-probe."""
    return max(1, int(row_count ** 0.5))


class LanceVectorIndex:
    """A :class:`~ragkit.core.ports.VectorIndex` over an embedded LanceDB table."""

    CONFIG_KEYS = frozenset({"path", "table", "dim", "metric", "nprobes"})

    def __init__(self, path: str, *, table: str = "chunks", dim: int = 0,
                 metric: str = "cosine", nprobes: int | None = None) -> None:
        if dim < 1:
            raise VectorIndexError(f"a vector index needs a positive dim, got {dim}")
        if nprobes is not None and nprobes < 1:
            raise VectorIndexError(f"nprobes must be >= 1 when set, got {nprobes}")
        self._metric = metric
        self._dim = dim
        self._nprobes = nprobes
        lancedb, pa = _require_lancedb()
        self._pa = pa
        try:
            db = lancedb.connect(path)
            schema = pa.schema([pa.field("id", pa.string()),
                                pa.field("vector", pa.list_(pa.float32(), dim)),
                                pa.field("meta", pa.string())])
            # exist_ok opens the table if it is already there (a resumed run) and creates it
            # otherwise, in one call -- more robust than a list_tables() check, whose result can
            # lag a just-written table across connections.
            self._table = db.create_table(table, schema=schema, exist_ok=True)
        except Exception as exc:
            raise VectorIndexError(f"could not open the LanceDB table {table!r}: {exc}") from exc

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> LanceVectorIndex:
        path = str(options.get("path", ""))
        if not path:
            raise VectorIndexError("the LanceDB vector index needs a 'path'")
        raw_nprobes = options.get("nprobes")
        return cls(path=path, table=str(options.get("table", "chunks")),
                   dim=int(options.get("dim", 0)), metric=str(options.get("metric", "cosine")),
                   nprobes=None if raw_nprobes is None else int(raw_nprobes))

    def upsert(self, ids: Sequence[str], vectors: Sequence[Sequence[float]],
               metas: Sequence[Mapping[str, Any]]) -> None:
        """Replace the rows for ``ids`` and insert the rest, as **one** transaction.

        Via LanceDB's native ``merge_insert``, not delete-then-add. The old two-statement form
        left a window in which a crash had removed the old vectors without adding the new, and
        wrote two fragments per call instead of one -- the fragment growth ``compact()`` exists to
        clean up. It also had no structural defence against writing the same id twice; that is
        exactly the shape of this table's real 5,000-duplicate-row corruption incident (see
        ``compact``). ``merge_insert`` joins on ``id``, so a matched row is *updated in place*
        rather than deleted and re-added, and a duplicate cannot be created by this path.
        """
        validate_upsert(ids, vectors, metas, dim=self._dim)
        if not ids:
            return
        rows = [{"id": i, "vector": list(v), "meta": json.dumps(dict(m))}
                for i, v, m in zip(ids, vectors, metas, strict=True)]
        try:
            (self._table.merge_insert("id")
             .when_matched_update_all()
             .when_not_matched_insert_all()
             .execute(rows))
        except Exception as exc:
            raise VectorIndexError(f"could not upsert {len(rows)} row(s): {exc}") from exc

    def search(self, vector: Sequence[float], *, k: int,
               where: Filter = ()) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        row_count = self.count()
        if row_count == 0:
            return []
        if len(vector) != self._dim:
            raise VectorIndexError(
                f"the query vector has dimension {len(vector)}, but the index is {self._dim}-d")
        builder = (self._table.search(list(vector)).metric(self._metric).limit(k)
                   .nprobes(self.search_nprobes(row_count)))
        predicate = to_sql(_supported(where))
        if predicate:
            builder = builder.where(predicate)
        return [(row["id"], _distance_to_score(row["_distance"], self._metric))
                for row in builder.to_list()]

    def search_nprobes(self, row_count: int) -> int:
        """How many IVF partitions a query probes: the configured ``nprobes``, or a default scaled
        to the table.

        This has to be set explicitly. LanceDB's own default probes a small fixed number of
        partitions, which is fine for a table with a handful of them and quietly lossy for one
        with thousands: ``create_index`` builds ``~sqrt(n)`` partitions, so a 7.1M-row table has
        ~2,650 and the default would read well under 1% of it per query. The observable symptom is
        a recall *drop* on the very tables the ANN index was added to speed up — measured as a
        Recall@20 dip on legal_procurement right after the index was first built.

        The default probes :data:`_NPROBE_FRACTION` of the partitions (floored at
        :data:`_MIN_NPROBES` so a small table is never probed less than LanceDB would have). Query
        cost is therefore a roughly fixed *fraction* of the corpus rather than a fixed count — the
        recall side of the trade is held constant as the table grows, and latency is what scales.
        Deliberately uncapped: an upper bound would silently reintroduce the recall cliff on the
        largest tables, which is the bug being fixed. Set ``nprobes`` in ``[vector]`` to override —
        necessary if ``create_index`` was given an explicit ``num_partitions``, since the default
        assumes the ``sqrt(n)`` heuristic both sides share via :func:`_ivf_partitions`.

        Harmless when no index exists (the scan is exact and ignores it) and when it exceeds the
        partition count (every partition is probed, i.e. exact).
        """
        if self._nprobes is not None:
            return self._nprobes
        return max(_MIN_NPROBES, math.ceil(_NPROBE_FRACTION * _ivf_partitions(row_count)))

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        quoted = ", ".join("'" + str(i).replace("'", "''") + "'" for i in ids)
        self._table.delete(f"id IN ({quoted})")

    def count(self) -> int:
        return int(self._table.count_rows())

    def iter_indexed_ids(self) -> Iterator[str]:
        # table.to_arrow() takes no column argument -- it would materialize every column (the
        # full 1024-d vector + meta JSON) for every row just to throw all but "id" away. At
        # millions of rows that is tens of GB for a single call. search().select(["id"]) projects
        # at the scan level so only the id column is ever read, and to_batches() streams it
        # instead of building one giant pyarrow Table.
        for batch in self._table.search().select(["id"]).limit(None).to_batches():
            yield from batch.column("id").to_pylist()

    def reconcile(self, chunk_ids: Iterable[str]) -> set[str]:
        return reconcile_against(chunk_ids, self.iter_indexed_ids(), self.delete)

    def close(self) -> None:
        # LanceDB holds no long-lived handle that needs explicit release for a local table; the
        # method exists for interface symmetry with the other stores.
        return None

    def create_index(self, *, num_partitions: int | None = None, replace: bool = True) -> None:
        """Build an IVF_FLAT ANN index on the vector column.

        ``search()`` (see above) silently falls back to an O(n) brute-force scan of every row
        whenever no index exists -- correct, but at millions of rows a single query then reads the
        entire vector column (confirmed: a 7.1M-row/1024-d table without an index made a 956-query
        eval CPU-bound and multi-hour, at 0% GPU use, because the "GPU-backed" retrieval pipeline
        never got past a full in-process table scan per query). IVF_FLAT (not IVF_PQ) is used
        because it needs no minimum training-set size and loses no precision to quantization --
        IVF_PQ requires >=256 rows per partition to train and this store's smallest real tables
        (recipe unit tests) have far fewer. ``num_partitions`` defaults to ``sqrt(row count)``, the
        standard IVF heuristic balancing per-partition scan cost against the number of partitions
        probed.

        **The index makes search approximate, and how approximate is a query-side decision.** An
        IVF query probes only some of the partitions; the rest of the table is never looked at, so
        a neighbour in an unprobed partition is simply missed. How many are probed is
        :meth:`search_nprobes`, which defaults to a fixed *fraction* of the count built here --
        that coupling is why both sides derive from :func:`_ivf_partitions`. Passing an explicit
        ``num_partitions`` breaks it, so pair that with an explicit ``nprobes`` in ``[vector]``.
        More partitions means faster queries at the same nprobes but a coarser cell each, so a
        matching nprobes rise is needed to hold recall.

        Not built automatically on upsert(): training needs a representative sample of the already-
        written data and is itself an expensive batch operation, so callers build it once after a
        corpus is (mostly) loaded, not on every write -- same reasoning as compact() below.

        Do not call this while another process is concurrently writing to the table -- same
        concurrent-access hazard as compact() (see its docstring): this table's real corruption
        incident came from exactly this class of concurrent access.
        """
        # See the module docstring: lancedb is imported lazily, only inside this module.
        from lancedb.index import IvfFlat  # type: ignore[import-untyped]

        row_count = self.count()
        if row_count == 0:
            raise VectorIndexError("cannot build a vector index on an empty table")
        partitions = num_partitions if num_partitions is not None else _ivf_partitions(row_count)
        try:
            self._table.create_index(
                "vector", config=IvfFlat(distance_type=self._metric, num_partitions=partitions),
                replace=replace)
        except Exception as exc:
            raise VectorIndexError(f"could not build the vector index: {exc}") from exc

    def compact(self) -> None:
        """Consolidate the small fragments left by many incremental ``upsert()``/``delete()``
        calls (each is a separate write transaction) into a few large ones, and prune old
        versions. Not a LanceVectorIndex/VectorIndex-port method other drivers need to implement
        (Qdrant's HNSW index has no on-disk fragment-file model to compact) — a driver-specific
        maintenance operation, called explicitly by ``tools/compact_vector_store.sh``, not on any
        read/write path.

        Confirmed necessary, not speculative: a table reconciled/upserted in many small batches
        over a long, repeatedly-resumed embedding job accumulates one fragment per batch without
        bound. legal_procurement's table reached 3,717 versions / 1,858 fragments for 2M rows, at
        which point simply *opening and reconciling against it* -- before embedding a single new
        row -- cost multiple GB of RSS per subsequent batch, because every read/write re-scans the
        whole growing fragment list. Compacting it to 2 fragments fixed that immediately.

        **Confirmed unsafe against a concurrent writer -- this actually corrupted data, once.**
        Calling this (or even just running two independent ``upsert()``-driven jobs) while a
        *different process* is mid-``upsert()``/``delete()`` against the same on-disk table is not
        just untested, it is confirmed broken: this session hit that scenario by accident (two
        embed-job workers briefly writing to the same table after a bash-wrapper-vs-child-process
        kill mistake) and it silently produced 5,000 duplicate rows -- same id, two rows each,
        `count_rows()` inflated by exactly that many, `reconcile()`'s set-based orphan/missing
        logic blind to it since it compares distinct ids, not row counts. No error was ever
        raised; it was only caught later by an exact ``pairings.count() == vector.count()`` audit
        after a job finished. "No error surfaced" is not evidence of no corruption with this
        write pattern -- always audit row counts after any concurrent-access incident, don't take
        a clean exit as proof nothing broke. Only call this from the same process that owns the
        table's writes (as :func:`~ragkit.ingest.reference.reconcile_vector`'s ``compact_every``
        does, sequentially inside its own batch loop), or when no writer is active.
        """
        try:
            import lance  # noqa: F401 -- to_lance()/optimize() need pylance; fail with our own
                          # message naming it, not a bare "No module named 'lance'" from deep
                          # inside lancedb's internals.
        except ImportError as exc:
            raise VectorIndexError(
                "compacting a LanceDB table needs 'pylance' (a separate package from 'lancedb'), "
                "which is not installed (pip install pylance; it ships in requirements.txt)."
            ) from exc
        try:
            self._table.optimize(cleanup_older_than=timedelta(0))
        except Exception as exc:
            raise VectorIndexError(f"could not compact the LanceDB table: {exc}") from exc


def _distance_to_score(distance: float, metric: str) -> float:
    """Convert a LanceDB distance to a higher-is-better score. Cosine distance is ``1 - cos_sim``
    over ``[0, 2]``; the score is the cosine similarity clamped into ``[0, 1]``. For L2 (unbounded)
    a bounded, order-preserving ``1/(1+d)`` is used."""
    if metric == "cosine":
        return max(0.0, min(1.0, 1.0 - distance))
    return 1.0 / (1.0 + max(0.0, distance))


def _require_lancedb() -> tuple[Any, Any]:
    try:
        import lancedb  # type: ignore[import-untyped]
        import pyarrow as pa  # type: ignore[import-untyped]
    except ImportError as exc:
        raise VectorIndexError(
            "the LanceDB vector index needs 'lancedb' and 'pyarrow', which are not installed "
            "(pip install lancedb; they ship in requirements.txt). Use a lexical-only retrieval "
            "config if you do not want the dense/vector path.") from exc
    return lancedb, pa
