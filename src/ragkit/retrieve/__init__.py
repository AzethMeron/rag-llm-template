"""Retrieval: first-stage lexical and dense retrievers over the storage indexes, the pure fusion
math (RRF, MMR), an optional cross-encoder reranker, and the hybrid retriever that composes them.

The stdlib-only fusion math needs nothing beyond the interpreter; the dense path adds numpy and a
local embedding endpoint, the rerank path a local rerank endpoint — each optional and behind an
injectable client. Depends on the store and core layers, never on the harness.
"""
from __future__ import annotations

from ragkit.core.ports import Retriever
from ragkit.core.registry import Registry

from .embedding import DEFAULT_EMBEDDING_MIN_SCORE, EmbeddingClient, EmbeddingError
from .fusion import RRF_K, best_possible_rrf, mmr_order, rrf
from .hybrid import HybridError, HybridRetriever
from .rerank import SCORE_SCALES, RerankClient, RerankError, sigmoid
from .retrievers import DenseRetriever, LexicalRetriever, trigram_similarity
from .tuning import RetrievalSettings, load_retrieval

RETRIEVERS: Registry[Retriever] = Registry(
    "retriever", Retriever,  # type: ignore[type-abstract]
    entry_point_group="ragkit.retrievers")

__all__ = [
    "RETRIEVERS", "RetrievalSettings", "load_retrieval",
    "LexicalRetriever", "DenseRetriever", "HybridRetriever", "HybridError",
    "EmbeddingClient", "EmbeddingError", "DEFAULT_EMBEDDING_MIN_SCORE",
    "RerankClient", "RerankError", "SCORE_SCALES", "sigmoid",
    "rrf", "mmr_order", "best_possible_rrf", "RRF_K", "trigram_similarity",
]
