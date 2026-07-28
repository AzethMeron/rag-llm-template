"""The default vector index: LanceDB — a real embedded vector database with true ANN.

LanceDB gives genuine ANN indexing, on-disk columnar storage with versioning, metadata filtering
and hybrid search, all in-process with no server or daemon. It is the framework's default because
it satisfies "a real vector database" while keeping the self-contained rule (pip-only, nothing to
launch).

``lancedb``, ``pyarrow`` and ``numpy`` are imported **lazily, inside this one module**, so the core
import path never pulls them and ``tests/test_boundaries.py`` can confine the allowance here. The
cosine metric returns a *distance* (``0`` identical, ``2`` opposite); the
:class:`~ragkit.core.ports.VectorIndex` port promises a higher-is-better score toward ``[0, 1]``,
so the conversion happens here.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ragkit.core.errors import RagkitError
from ragkit.core.ports import Filter

from ..filters import to_sql


class VectorIndexError(RagkitError):
    """A vector-index operation failed, or LanceDB/pyarrow is unavailable."""


class LanceVectorIndex:
    """A :class:`~ragkit.core.ports.VectorIndex` over an embedded LanceDB table."""

    CONFIG_KEYS = frozenset({"path", "table", "dim", "metric"})

    def __init__(self, path: str, *, table: str = "chunks", dim: int = 0,
                 metric: str = "cosine") -> None:
        if dim < 1:
            raise VectorIndexError(f"a vector index needs a positive dim, got {dim}")
        self._metric = metric
        self._dim = dim
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
        return cls(path=path, table=str(options.get("table", "chunks")),
                   dim=int(options.get("dim", 0)), metric=str(options.get("metric", "cosine")))

    def upsert(self, ids: Sequence[str], vectors: Sequence[Sequence[float]],
               metas: Sequence[Mapping[str, Any]]) -> None:
        if not (len(ids) == len(vectors) == len(metas)):
            raise VectorIndexError(
                f"upsert got mismatched lengths: {len(ids)} ids, {len(vectors)} vectors, "
                f"{len(metas)} metas")
        if not ids:
            return
        for vector in vectors:
            if len(vector) != self._dim:
                raise VectorIndexError(
                    f"a vector has dimension {len(vector)}, but the index is {self._dim}-d")
        self.delete(ids)  # upsert = replace any existing rows for these ids
        rows = [{"id": i, "vector": list(v), "meta": json.dumps(dict(m))}
                for i, v, m in zip(ids, vectors, metas, strict=True)]
        self._table.add(rows)

    def search(self, vector: Sequence[float], *, k: int,
               where: Filter = ()) -> list[tuple[str, float]]:
        if k <= 0 or self.count() == 0:
            return []
        if len(vector) != self._dim:
            raise VectorIndexError(
                f"the query vector has dimension {len(vector)}, but the index is {self._dim}-d")
        builder = self._table.search(list(vector)).metric(self._metric).limit(k)
        predicate = to_sql(where)
        if predicate:
            builder = builder.where(predicate)
        return [(row["id"], _distance_to_score(row["_distance"], self._metric))
                for row in builder.to_list()]

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        quoted = ", ".join("'" + str(i).replace("'", "''") + "'" for i in ids)
        self._table.delete(f"id IN ({quoted})")

    def count(self) -> int:
        return int(self._table.count_rows())

    def indexed_ids(self) -> set[str]:
        return {row["id"] for row in self._table.to_arrow().select(["id"]).to_pylist()}

    def reconcile(self, chunk_ids: Iterable[str]) -> set[str]:
        authoritative = set(chunk_ids)
        indexed = self.indexed_ids()
        orphans = indexed - authoritative
        if orphans:
            self.delete(sorted(orphans))
        return authoritative - indexed  # missing: the caller's to re-embed or refuse

    def close(self) -> None:
        # LanceDB holds no long-lived handle that needs explicit release for a local table; the
        # method exists for interface symmetry with the other stores.
        return None


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
