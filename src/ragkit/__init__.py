"""rag-llm-template: a configurable RAG + LLM task framework.

The public contract lives in :mod:`ragkit.core` (stdlib-only). The layers above it — storage,
ingestion, retrieval, the model layer, and the harness — depend on that contract and on each
other only downward. See ``docs/architecture.md``.
"""
from __future__ import annotations

__version__ = "0.1.0"
