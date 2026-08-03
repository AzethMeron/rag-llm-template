"""The model pool: one connection per endpoint, per-persona model routing, and a thrash guard.

Each persona (a reviewer, the producer) names a logical model; several personas may name the
same one, and different ones may name different models. The pool owns **one HTTP connection per
endpoint** and hands out one :class:`~ragkit.llm.client.LlmClient` per logical model over it, so
personas sharing a model share a client (and, on a router-mode server, share the loaded weights).

The correction to the common assumption is enforced here. A router (llama.cpp router mode,
Ollama) keeps only so many distinct models resident — ``resident_max`` — and evicts the rest by
LRU. With more distinct models in use than the cap, a round-robin over personas makes every
request evict the model the next one needs: silent thrash. :meth:`ModelPool.check_capacity`
turns that into a **load-time refusal**, naming the models and the two fixes, before the server
is ever contacted. It counts *distinct models*, not personas — persona count costs KV-cache
slots, not weight memory.
"""
from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import httpx

from ragkit.core.config import (
    ConfigError,
    as_table,
    load_toml,
    read_bool,
    read_float,
    read_int,
    read_string,
    read_string_list,
    reject_unknown,
)
from ragkit.core.errors import RagkitError

from .backends import resolve_backend
from .client import LlmClient, ServerConfig

_PROVIDERS = frozenset({"llamacpp-router", "ollama", "openai-compatible"})
_KINDS = frozenset({"chat", "embedding", "rerank"})


class ModelPoolError(RagkitError):
    """The model configuration is inconsistent, or its capacity would thrash."""


@dataclass(frozen=True, slots=True)
class EndpointSpec:
    """One inference endpoint: where it is, and how many models it holds resident."""

    name: str
    provider: str = "llamacpp-router"
    base_url: str = "http://127.0.0.1:8080/v1"
    resident_max: int = 4
    """How many distinct models stay resident before the server evicts by LRU (llama.cpp
    ``--models-max``, Ollama ``OLLAMA_MAX_LOADED_MODELS``). The thrash guard's ceiling."""
    parallel: int = 2
    """Server request slots; must be >= the harness concurrency or in-flight requests serialise."""
    vram_budget_mb: int = 0
    """Optional VRAM ceiling for the byte-level guard. 0 disables it (count-only)."""
    timeout_seconds: float = 300.0
    max_retries: int = 4
    retry_backoff_seconds: float = 2.0
    enable_reasoning: bool = False
    server_args: tuple[str, ...] = ()
    """Launch-time flags for the server that hosts this endpoint — the settings that are *not*
    per-request (GPU offload, KV-cache type, rope scaling, flash attention). Per-request decode
    settings live on the persona instead. ``serve_models.sh`` reads these so ``models.toml`` is the
    single source of truth for both routing and serving; the pool itself never launches a server,
    so it only carries them."""

    def __post_init__(self) -> None:
        if self.provider not in _PROVIDERS:
            raise ModelPoolError(
                f"endpoint {self.name!r}: unknown provider {self.provider!r}; known: "
                f"{sorted(_PROVIDERS)}")
        for value, what in ((self.resident_max, "resident_max"), (self.parallel, "parallel")):
            if value < 1:
                raise ModelPoolError(f"endpoint {self.name!r}: {what} must be >= 1, got {value}")
        if self.vram_budget_mb < 0:
            raise ModelPoolError(
                f"endpoint {self.name!r}: vram_budget_mb must be >= 0, got {self.vram_budget_mb}")


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One logical model a persona can name: which endpoint serves it, its served id, and how to
    shape a structured request for it."""

    name: str
    endpoint: str
    model_id: str
    backend: str = "auto"
    kind: str = "chat"
    context_window: int = 0
    approx_vram_mb: int = 0
    """Optional resident weight size, in MB, for the byte-level thrash guard. 0 means undeclared —
    the guard then counts models only and says so, rather than implying a check it did not do."""

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ModelPoolError(
                f"model {self.name!r}: unknown kind {self.kind!r}; known: {sorted(_KINDS)}")
        if self.context_window < 0:
            raise ModelPoolError(
                f"model {self.name!r}: context_window must be >= 0, got {self.context_window}")
        if self.approx_vram_mb < 0:
            raise ModelPoolError(
                f"model {self.name!r}: approx_vram_mb must be >= 0, got {self.approx_vram_mb}")


# One home for each spec's field defaults: the loaders read them from these instances rather than
# re-typing every literal, so a default and its loader cannot drift (the Leniency loader pattern).
# The required id fields are placeholders -- only the defaulted fields are read.
_ENDPOINT_DEFAULTS = EndpointSpec(name="")
_MODEL_DEFAULTS = ModelSpec(name="", endpoint="", model_id="")


# Builds the httpx.Client for an endpoint, given its base_url and timeout. Injected so a test
# drives an in-memory transport with no server; a real caller lets each endpoint own a real one.
ClientFactory = Callable[[str, float], httpx.Client]


class ModelPool:
    """Owns one HTTP connection per endpoint and routes each logical model over it.

    ``client_factory`` builds the ``httpx.Client`` for an endpoint (given its ``base_url`` and
    ``timeout``); a test injects one wrapping a ``MockTransport``. The pool closes every client it
    opened; an injected transport's lifecycle is the test's.

    **Thread-safe lazily.** ``run_batch`` starts ``concurrency`` workers that all call
    :meth:`client_for` for the same persona model at once, on a cold pool. The two caches are
    therefore built under a lock: an unguarded check-then-set let several threads each build an
    ``LlmClient`` + ``httpx.Client`` for the same key, and every loser was overwritten in the dict
    and so never closed by :meth:`close` — a leaked connection pool per race, plus usage stats
    split across the discarded clients. Reentrant because :meth:`client_for` builds an endpoint's
    HTTP client while already holding it.
    """

    def __init__(self, endpoints: Mapping[str, EndpointSpec], models: Mapping[str, ModelSpec], *,
                 client_factory: ClientFactory | None = None) -> None:
        for model in models.values():
            if model.endpoint not in endpoints:
                raise ModelPoolError(
                    f"model {model.name!r} names endpoint {model.endpoint!r}, which is not "
                    f"defined; known endpoints: {sorted(endpoints)}")
        self._endpoints = dict(endpoints)
        self._models = dict(models)
        self._client_factory = client_factory
        self._http: dict[str, httpx.Client] = {}
        self._clients: dict[str, LlmClient] = {}
        self._build_lock = threading.RLock()

    def check_capacity(self, active_models: Iterable[str]) -> None:
        """Refuse, before any server is contacted, a set of in-use models that would thrash.

        For each endpoint, the distinct models among ``active_models`` must not exceed
        ``resident_max``. When every such model declares ``approx_vram_mb`` and the endpoint sets
        ``vram_budget_mb``, their sum must also fit. The refusal names the models and both fixes,
        and states whether it checked bytes or only the count — never implying a check it skipped.
        """
        by_endpoint: dict[str, set[str]] = {}
        for name in active_models:
            spec = self._models.get(name)
            if spec is None:
                raise ModelPoolError(
                    f"unknown model {name!r}; defined models: {sorted(self._models)}")
            by_endpoint.setdefault(spec.endpoint, set()).add(name)

        for endpoint_name, model_names in by_endpoint.items():
            endpoint = self._endpoints[endpoint_name]
            distinct = {self._models[n].model_id for n in model_names}
            if len(distinct) > endpoint.resident_max:
                raise ModelPoolError(
                    f"endpoint {endpoint_name!r} would hold {len(distinct)} distinct models "
                    f"({sorted(distinct)}) but resident_max is {endpoint.resident_max}; every "
                    f"request would evict the model the next one needs. Raise resident_max (the "
                    f"server's --models-max / OLLAMA_MAX_LOADED_MODELS), or point these personas "
                    f"at fewer distinct models.")
            self._check_vram(endpoint, model_names, distinct)

    def _check_vram(self, endpoint: EndpointSpec, model_names: set[str],
                    distinct_ids: set[str]) -> None:
        if endpoint.vram_budget_mb <= 0:
            return
        # One spec per distinct served id (personas sharing a model share its weights).
        by_id = {self._models[n].model_id: self._models[n] for n in model_names}
        declared = [by_id[mid].approx_vram_mb for mid in distinct_ids]
        if not all(declared):
            # A byte check is only honest if every resident model declares its size; otherwise the
            # count check above is all that ran, and the guard must not imply otherwise.
            return
        total = sum(declared)
        if total > endpoint.vram_budget_mb:
            raise ModelPoolError(
                f"endpoint {endpoint.name!r}: the resident models need ~{total} MB of VRAM but "
                f"vram_budget_mb is {endpoint.vram_budget_mb}; they will not fit together. Use "
                f"smaller/more-quantised models, raise vram_budget_mb, or use fewer distinct "
                f"models.")

    def client_for(self, model_name: str) -> tuple[LlmClient, str]:
        """The ``(client, served_model_id)`` for a logical chat model, building it once per model
        (personas sharing a model share the client) over the endpoint's shared connection."""
        spec = self._models.get(model_name)
        if spec is None:
            raise ModelPoolError(f"unknown model {model_name!r}; defined: {sorted(self._models)}")
        if spec.kind != "chat":
            raise ModelPoolError(
                f"model {model_name!r} is a {spec.kind!r} model, not a chat model; the persona "
                f"pool serves chat models (embedding/rerank models are used by the retrieval "
                f"layer)")
        with self._build_lock:
            if model_name not in self._clients:
                endpoint = self._endpoints[spec.endpoint]
                http = self._http_for(endpoint)
                config = ServerConfig(
                    base_url=endpoint.base_url, model=spec.model_id,
                    timeout_seconds=endpoint.timeout_seconds, max_retries=endpoint.max_retries,
                    retry_backoff_seconds=endpoint.retry_backoff_seconds,
                    context_window=spec.context_window, enable_reasoning=endpoint.enable_reasoning)
                self._clients[model_name] = LlmClient(
                    config, backend=resolve_backend(spec.backend, spec.model_id), client=http)
            return self._clients[model_name], spec.model_id

    def _http_for(self, endpoint: EndpointSpec) -> httpx.Client:
        with self._build_lock:
            if endpoint.name not in self._http:
                if self._client_factory is not None:
                    http = self._client_factory(endpoint.base_url, endpoint.timeout_seconds)
                else:
                    http = httpx.Client(timeout=endpoint.timeout_seconds)
                self._http[endpoint.name] = http
            return self._http[endpoint.name]

    def model_spec(self, model_name: str) -> ModelSpec:
        spec = self._models.get(model_name)
        if spec is None:
            raise ModelPoolError(f"unknown model {model_name!r}; defined: {sorted(self._models)}")
        return spec

    def endpoint(self, name: str) -> EndpointSpec:
        spec = self._endpoints.get(name)
        if spec is None:
            raise ModelPoolError(
                f"unknown endpoint {name!r}; defined: {sorted(self._endpoints)}")
        return spec

    def models_of_kind(self, kind: str) -> dict[str, ModelSpec]:
        """Every model of a given kind, for the retrieval layer to find its embedding/rerank
        models."""
        if kind not in _KINDS:
            raise ModelPoolError(f"unknown kind {kind!r}; known: {sorted(_KINDS)}")
        return {name: spec for name, spec in self._models.items() if spec.kind == kind}

    def close(self) -> None:
        """Close every HTTP connection the pool opened. The LlmClients share these, so they are
        not closed individually (each was given an injected client it does not own)."""
        with self._build_lock:
            for http in self._http.values():
                http.close()
        self._http.clear()
        self._clients.clear()

    def __enter__(self) -> ModelPool:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def load_models(path: Path, *, client_factory: ClientFactory | None = None) -> ModelPool:
    """Build a :class:`ModelPool` from a ``models.toml``. Endpoints are ``[endpoint.<name>]``
    tables, logical models ``[model.<name>]`` tables. Unknown keys and mistyped values are
    refused, per the framework's config rules; range checks live on the dataclasses."""
    data = load_toml(path, what="models file")
    reject_unknown(data, {"endpoint", "model"}, label="the models file", path=path)

    endpoints = _load_endpoints(data.get("endpoint", {}), path=path)
    if not endpoints:
        raise ConfigError("no [endpoint.<name>] tables defined", path=path)
    models = _load_model_specs(data.get("model", {}), path=path)
    if not models:
        raise ConfigError("no [model.<name>] tables defined", path=path)
    try:
        return ModelPool(endpoints, models, client_factory=client_factory)
    except ModelPoolError as exc:
        raise ConfigError(str(exc), path=path) from exc


_ENDPOINT_KEYS = {"provider", "base_url", "resident_max", "parallel", "vram_budget_mb",
                  "timeout_seconds", "max_retries", "retry_backoff_seconds", "enable_reasoning",
                  "server_args"}
_MODEL_KEYS = {"endpoint", "model_id", "backend", "kind", "context_window", "approx_vram_mb"}


def _load_endpoints(section: object, *, path: Path) -> dict[str, EndpointSpec]:
    table = as_table(section, label="[endpoint]", path=path)
    endpoints: dict[str, EndpointSpec] = {}
    for name, raw in table.items():
        body = reject_unknown(raw, _ENDPOINT_KEYS, label=f"[endpoint.{name}]", path=path)
        d, label = _ENDPOINT_DEFAULTS, f"[endpoint.{name}]"
        try:
            endpoints[name] = EndpointSpec(
                name=name,
                provider=read_string(body, "provider", d.provider, label=label, path=path),
                base_url=read_string(body, "base_url", d.base_url, label=label, path=path),
                resident_max=read_int(body, "resident_max", d.resident_max, label=label, path=path),
                parallel=read_int(body, "parallel", d.parallel, label=label, path=path),
                vram_budget_mb=read_int(body, "vram_budget_mb", d.vram_budget_mb,
                                        label=label, path=path),
                timeout_seconds=read_float(body, "timeout_seconds", d.timeout_seconds,
                                           label=label, path=path),
                max_retries=read_int(body, "max_retries", d.max_retries, label=label, path=path),
                retry_backoff_seconds=read_float(body, "retry_backoff_seconds",
                                                 d.retry_backoff_seconds, label=label, path=path),
                enable_reasoning=read_bool(body, "enable_reasoning", d.enable_reasoning,
                                           label=label, path=path),
                server_args=read_string_list(body, "server_args", label=label, path=path))
        except ModelPoolError as exc:
            raise ConfigError(str(exc), path=path) from exc
    return endpoints


def _load_model_specs(section: object, *, path: Path) -> dict[str, ModelSpec]:
    table = as_table(section, label="[model]", path=path)
    models: dict[str, ModelSpec] = {}
    for name, raw in table.items():
        body = reject_unknown(raw, _MODEL_KEYS, label=f"[model.{name}]", path=path)
        model_id = read_string(body, "model_id", "", label=f"[model.{name}]", path=path)
        if not model_id:
            raise ConfigError(f"[model.{name}] needs a non-empty model_id", path=path)
        endpoint = read_string(body, "endpoint", "", label=f"[model.{name}]", path=path)
        if not endpoint:
            raise ConfigError(f"[model.{name}] needs an endpoint", path=path)
        d, label = _MODEL_DEFAULTS, f"[model.{name}]"
        try:
            models[name] = ModelSpec(
                name=name, endpoint=endpoint, model_id=model_id,
                backend=read_string(body, "backend", d.backend, label=label, path=path),
                kind=read_string(body, "kind", d.kind, label=label, path=path),
                context_window=read_int(body, "context_window", d.context_window,
                                        label=label, path=path),
                approx_vram_mb=read_int(body, "approx_vram_mb", d.approx_vram_mb,
                                        label=label, path=path))
        except ModelPoolError as exc:
            raise ConfigError(str(exc), path=path) from exc
    return models
