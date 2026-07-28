"""The model layer: a transport client, per-family request-shaping backends, and a pool that
routes each persona's model over one connection per endpoint.

Depends only on the core contract and ``httpx``. Everything that varies between models lives in
a :class:`~ragkit.llm.backends.BaseBackend`; the client owns only the transport, retry, error
taxonomy, and usage accounting.
"""
from __future__ import annotations

from ragkit.core.ports import Message, StructuredRequest

from .backends import (
    BackendError,
    BaseBackend,
    JsonObjectBackend,
    SchemaBackend,
    available_backends,
    get_backend,
    register_backend,
    resolve_backend,
    suggest_backend,
)
from .client import LlmClient, ServerConfig, UsageStats
from .errors import (
    LlmContentError,
    LlmError,
    LlmIncompleteJsonError,
    LlmRefusalError,
    LlmTruncationError,
)
from .pool import (
    EndpointSpec,
    ModelPool,
    ModelPoolError,
    ModelSpec,
    load_models,
)

__all__ = [
    # transport
    "LlmClient", "ServerConfig", "UsageStats", "Message", "StructuredRequest",
    # backends
    "BaseBackend", "SchemaBackend", "JsonObjectBackend", "BackendError",
    "register_backend", "get_backend", "resolve_backend", "available_backends", "suggest_backend",
    # errors (typed by blast radius)
    "LlmError", "LlmContentError", "LlmRefusalError", "LlmTruncationError",
    "LlmIncompleteJsonError",
    # pool
    "ModelPool", "ModelSpec", "EndpointSpec", "ModelPoolError", "load_models",
]
