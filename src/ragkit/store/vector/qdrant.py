"""A second real vector database behind the ``VectorIndex`` port: Qdrant, in embedded local mode.

Ships so swapping the vector store is a config edit (``driver = "lancedb"`` -> ``"qdrant"``) with no
change to any other code — the vector-DB counterpart of the SQLite->DuckDB SQL swap. Qdrant is a
real vector database with HNSW ANN and native metadata filtering; its client's *local mode*
(``QdrantClient(path=...)`` / ``":memory:"``) runs entirely in-process with no server, so it keeps
the self-contained rule. The plan names Qdrant as the production tier too — the same driver code
points at a Qdrant server by URL.

``qdrant_client`` is imported **lazily inside this one module**, so the core/store import path never
pulls it and ``tests/test_boundaries.py`` confines the allowance here. Cosine distance in Qdrant is
returned as a similarity score (``1`` identical), already the port's higher-is-better convention; it
is clamped into ``[0, 1]`` to match the other drivers exactly. Chunk ids are strings, but a Qdrant
point id must be an int or UUID, so the id is stored in the payload and the point id is a
deterministic UUID5 of it (so an upsert of the same chunk id overwrites in place).
"""
from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ragkit.core.ports import Filter, FilterOp, Predicate

from .common import VectorIndexError, validate_upsert

_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")  # fixed: deterministic point ids

__all__ = ["QdrantVectorIndex", "VectorIndexError"]  # VectorIndexError re-exported from .common


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


class QdrantVectorIndex:
    """A :class:`~ragkit.core.ports.VectorIndex` over an embedded (or remote) Qdrant collection."""

    CONFIG_KEYS = frozenset({"path", "url", "collection", "dim"})

    def __init__(self, path: str = ":memory:", *, url: str = "", collection: str = "chunks",
                 dim: int = 0) -> None:
        if dim < 1:
            raise VectorIndexError(f"a vector index needs a positive dim, got {dim}")
        self._dim = dim
        self._collection = collection
        client_cls, models = _require_qdrant()
        self._m = models
        try:
            # A url points at a Qdrant server; else local mode -- on-disk at path, or in-memory.
            if url:  # pragma: no cover - a real Qdrant server, not exercised by the unit suite
                self._client = client_cls(url=url)
            elif path == ":memory:":
                self._client = client_cls(location=":memory:")
            else:
                self._client = client_cls(path=path)
            if not self._client.collection_exists(collection):
                self._client.create_collection(
                    collection, vectors_config=models.VectorParams(
                        size=dim, distance=models.Distance.COSINE))
        except Exception as exc:
            raise VectorIndexError(f"could not open the Qdrant collection {collection!r}: "
                                   f"{exc}") from exc

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> QdrantVectorIndex:
        return cls(path=str(options.get("path", ":memory:")), url=str(options.get("url", "")),
                   collection=str(options.get("collection", "chunks")),
                   dim=int(options.get("dim", 0)))

    def upsert(self, ids: Sequence[str], vectors: Sequence[Sequence[float]],
               metas: Sequence[Mapping[str, Any]]) -> None:
        validate_upsert(ids, vectors, metas, dim=self._dim)
        if not ids:
            return
        points = [self._m.PointStruct(id=_point_id(cid), vector=list(vector),
                                      payload={**dict(meta), "_cid": cid})
                  for cid, vector, meta in zip(ids, vectors, metas, strict=True)]
        self._client.upsert(self._collection, points=points)

    def search(self, vector: Sequence[float], *, k: int,
               where: Filter = ()) -> list[tuple[str, float]]:
        if k <= 0 or self.count() == 0:
            return []
        if len(vector) != self._dim:
            raise VectorIndexError(
                f"the query vector has dimension {len(vector)}, but the index is {self._dim}-d")
        hits = self._client.query_points(
            self._collection, query=list(vector), limit=k,
            query_filter=self._to_filter(where), with_payload=True).points
        return [(hit.payload["_cid"], max(0.0, min(1.0, hit.score))) for hit in hits]

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        self._client.delete(self._collection, points_selector=self._m.FilterSelector(
            filter=self._m.Filter(must=[self._m.FieldCondition(
                key="_cid", match=self._m.MatchAny(any=list(ids)))])))

    def count(self) -> int:
        return int(self._client.count(self._collection).count)

    def indexed_ids(self) -> set[str]:
        found: set[str] = set()
        offset = None
        while True:
            points, offset = self._client.scroll(
                self._collection, limit=256, offset=offset, with_payload=True, with_vectors=False)
            found.update(point.payload["_cid"] for point in points)
            if offset is None:
                return found

    def reconcile(self, chunk_ids: Iterable[str]) -> set[str]:
        authoritative = set(chunk_ids)
        indexed = self.indexed_ids()
        orphans = indexed - authoritative
        if orphans:
            self.delete(sorted(orphans))
        return authoritative - indexed  # missing: the caller's to re-embed or refuse

    def close(self) -> None:
        self._client.close()

    def _to_filter(self, where: Filter) -> Any:
        """Compile the framework's backend-neutral :class:`~ragkit.core.ports.Filter` (a conjunction
        of predicates) into a Qdrant filter. A predicate Qdrant cannot honour is refused here, at
        the boundary, rather than silently returning wrong results."""
        if not where:
            return None
        must, must_not = [], []
        for predicate in where:
            must_not.append(self._condition(predicate)) if predicate.op is FilterOp.NE \
                else must.append(self._condition(predicate))
        return self._m.Filter(must=must or None, must_not=must_not or None)

    def _condition(self, predicate: Predicate) -> Any:
        models, field = self._m, predicate.field
        if predicate.op in (FilterOp.EQ, FilterOp.NE):
            return models.FieldCondition(key=field, match=models.MatchValue(value=predicate.value))
        if predicate.op is FilterOp.IN:
            values = list(predicate.value) if isinstance(predicate.value, (list, tuple)) else None
            if not values:
                raise VectorIndexError(f"the IN filter on {field!r} needs a non-empty list")
            return models.FieldCondition(key=field, match=models.MatchAny(any=values))
        bounds = {FilterOp.LT: "lt", FilterOp.LE: "lte", FilterOp.GT: "gt", FilterOp.GE: "gte"}
        return models.FieldCondition(key=field,
                                     range=models.Range(**{bounds[predicate.op]: predicate.value}))


def _require_qdrant() -> tuple[Any, Any]:
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        raise VectorIndexError(
            "the Qdrant vector index needs 'qdrant-client', which is not installed "
            "(pip install qdrant-client; it ships in requirements.txt). Use the 'lancedb' driver, "
            "or a lexical-only retrieval config, if you do not want it.") from exc
    return QdrantClient, models
