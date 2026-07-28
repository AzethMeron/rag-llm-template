"""Synchronous client for a local, OpenAI-compatible inference server.

Model-agnostic on purpose: everything that varies between models — above all how a structured
request must be shaped — lives in a :class:`~ragkit.llm.backends.BaseBackend`, which this client
consults rather than deciding for itself. The client owns only the transport: the HTTP
connection, retry with backoff, the error taxonomy, and usage accounting.

Prompts are assembled most-static-part-first by callers (system rules, then context, then the
input) so the server's prefix cache is reused across requests — the dominant throughput factor
for a large job on one loaded model.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import httpx

from ragkit.core.ports import Message, SamplingParams

from .backends import DEFAULT_BACKEND, BaseBackend, SchemaBackend, get_backend, suggest_backend
from .errors import (
    LlmContentError,
    LlmError,
    LlmIncompleteJsonError,
    LlmRefusalError,
    LlmTruncationError,
)

logger = logging.getLogger(__name__)

_DEFAULT_SAMPLING = SamplingParams()
"""The decode settings a request uses when a caller passes none — a low temperature, every other
knob left to the server's default. Personas normally supply their own."""

CONTEXT_WARN_FRACTION = 0.8
"""How full the context window may get before a proactive warning: prompt plus output budget
crossing this fraction means the run is approaching truncation. The server's own check is the
hard stop; this is the early warning before it."""


class ServerConfig:
    """Connection and decoding settings for the local server — purely about the transport. How a
    model wants a structured request shaped is a backend concern, not a field here."""

    __slots__ = (
        "base_url",
        "context_window",
        "enable_reasoning",
        "max_retries",
        "model",
        "retry_backoff_seconds",
        "timeout_seconds",
    )

    def __init__(self, *, base_url: str = "http://127.0.0.1:8080/v1", model: str = "local",
                 timeout_seconds: float = 300.0, max_retries: int = 4,
                 retry_backoff_seconds: float = 2.0, context_window: int = 0,
                 enable_reasoning: bool = False) -> None:
        if max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {max_retries}")
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds}")
        if retry_backoff_seconds < 0:
            # A negative backoff would make time.sleep raise on the first retry -- a latent crash
            # on the error path, exactly where it must not.
            raise ValueError(f"retry_backoff_seconds must be >= 0, got {retry_backoff_seconds}")
        if context_window < 0:
            raise ValueError(f"context_window must be >= 0, got {context_window}")
        self.base_url = base_url
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.context_window = context_window
        self.enable_reasoning = enable_reasoning


@dataclass
class UsageStats:
    """Cumulative token and latency accounting, for throughput reporting. One client is shared by
    every worker in a concurrent run, so all mutation is guarded: a bare ``+=`` loses updates."""

    requests: int = 0
    prompt_tokens: int = 0
    peak_prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    retries: int = 0
    refusals: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, usage: dict[str, Any], elapsed: float) -> None:
        with self._lock:
            self.requests += 1
            # The usage block comes from the server, so its shape is not ours to trust: a null or
            # non-numeric count must not sink a request that otherwise succeeded. Degrade to zero.
            prompt = _token_count(usage.get("prompt_tokens"))
            self.prompt_tokens += prompt
            self.peak_prompt_tokens = max(self.peak_prompt_tokens, prompt)
            self.completion_tokens += _token_count(usage.get("completion_tokens"))
            self.seconds += elapsed

    def record_retry(self) -> None:
        with self._lock:
            self.retries += 1

    def record_refusal(self) -> None:
        with self._lock:
            self.refusals += 1

    @property
    def completion_tokens_per_second(self) -> float:
        return self.completion_tokens / self.seconds if self.seconds else 0.0


class LlmClient:
    """Synchronous chat client with retry and backend-directed structured output.

    Owns its HTTP connection; use as a context manager so the socket is released
    deterministically. The backend is fixed for the client's lifetime — one loaded model, one
    request shape. ``client`` is injectable so a test drives an in-memory transport with no
    server.
    """

    def __init__(self, config: ServerConfig | None = None, *, backend: BaseBackend | None = None,
                 client: httpx.Client | None = None) -> None:
        self.config = config or ServerConfig()
        self.backend = backend or get_backend(DEFAULT_BACKEND)
        self.stats = UsageStats()
        self._client = client or httpx.Client(timeout=self.config.timeout_seconds)
        self._owns_client = client is None
        # The prompt-budget warning fires once per run, not per request; the check-and-set is
        # guarded because one client is shared by every worker.
        self._context_warned = False
        self._context_lock = threading.Lock()

    def __enter__(self) -> LlmClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the HTTP client if this object opened it; an injected one is the injector's."""
        if self._owns_client:
            self._client.close()

    def health(self) -> bool:
        """True when the server answers a models listing."""
        try:
            response = self._client.get(f"{self.config.base_url}/models", timeout=10.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    def complete(self, messages: Sequence[Message], *, role: str = "<none>",
                 sampling: SamplingParams | None = None, max_tokens: int = 1024,
                 schema: dict[str, Any] | None = None, model: str | None = None) -> str:
        """Run one chat completion, retrying transient failures. When ``schema`` is given, the
        backend decides how the request asks for conforming JSON. ``model`` overrides the config's
        model id (the model pool passes a per-persona model over one shared client). ``sampling``
        carries the per-request decode settings (the persona's); ``max_tokens`` is the separate
        computed output budget."""
        params = sampling or _DEFAULT_SAMPLING
        if schema is not None:
            request = self.backend.structured_request(messages, schema)
            sent = [{"role": m.role, "content": m.content} for m in request.messages]
            response_format: dict[str, Any] | None = request.response_format
        else:
            sent = [{"role": m.role, "content": m.content} for m in messages]
            response_format = None

        payload: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": sent,
            "max_tokens": max_tokens,
            **params.payload(),
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if not self.config.enable_reasoning:
            # Suppress chain-of-thought. A reasoning model otherwise spends hidden tokens against
            # max_tokens and truncates the answer; for a bounded task the reasoning buys nothing.
            # Sent as the chat-template kwarg the reasoning families read; a template that does not
            # declare it ignores the kwarg, so this is inert for other models.
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        self._warn_if_context_tight(sent, max_tokens, role)

        last: Exception | None = None
        for attempt in range(self.config.max_retries):
            if attempt:
                self.stats.record_retry()
                time.sleep(self.config.retry_backoff_seconds * (2 ** (attempt - 1)))
            try:
                started = time.monotonic()
                response = self._client.post(
                    f"{self.config.base_url}/chat/completions", json=payload)
                elapsed = time.monotonic() - started
            except httpx.HTTPError as exc:
                last = LlmError(f"transport failure: {exc}", role=role)
                continue

            if response.status_code >= 500:
                last = LlmError(self._server_error_reason(schema, response.text),
                                role=role, status=response.status_code, body=response.text)
                continue
            if response.status_code != 200:
                # 4xx indicates a malformed request; retrying cannot help.
                raise LlmError(self._server_error_reason(schema, response.text),
                               role=role, status=response.status_code, body=response.text)

            try:
                body = response.json()
                choice = body["choices"][0]
                content = choice["message"]["content"]
            except (json.JSONDecodeError, KeyError, IndexError) as exc:
                last = LlmError(f"malformed response: {exc}", role=role, body=response.text)
                continue

            if content is None:
                self.stats.record_refusal()
                raise LlmRefusalError("model returned no content", role=role)

            self.stats.record(body.get("usage", {}), elapsed)
            if choice.get("finish_reason") == "length":
                raise LlmTruncationError(
                    f"generation stopped at the {max_tokens}-token ceiling; the model did not "
                    f"finish (usually a repetition loop)", role=role, body=content[-200:])
            return cast("str", content)

        raise last or LlmError("exhausted retries with no recorded error", role=role)

    def _warn_if_context_tight(self, sent: list[dict[str, str]], max_tokens: int,
                               role: str) -> None:
        """Warn once when the estimated prompt plus its output budget nears the context window. A
        no-op unless ``config.context_window`` is set; the server's own overflow rejection stays
        the hard stop."""
        window = self.config.context_window
        if window <= 0:
            return
        estimated = sum(_estimate_tokens(message["content"]) for message in sent)
        projected = estimated + max_tokens
        if projected < window * CONTEXT_WARN_FRACTION:
            return
        with self._context_lock:
            if self._context_warned:
                return
            self._context_warned = True
        logger.warning(
            "prompt budget is tight: role %r is an estimated ~%d input tokens + up to %d output "
            "= ~%d, about %.0f%% of the %d-token context window. Approaching truncation -- reduce "
            "context, shorten the instructions, or raise the server's context size. Further budget "
            "warnings this run are suppressed.",
            role, estimated, max_tokens, projected, 100 * projected / window, window)

    def _server_error_reason(self, schema: dict[str, Any] | None, body: str) -> str:
        """A rejection message that names the likely cause when it is known — a context overflow,
        or a build that cannot compile a strict schema grammar — rather than degrading silently."""
        base = "request rejected"
        lowered = body.lower()
        if ("exceed_context" in lowered or "context size" in lowered
                or "context window" in lowered or "n_ctx" in lowered):
            return (f"{base}: the prompt is larger than the server's context window. Reduce the "
                    f"context blocks or instructions, or raise the server's context size "
                    f"(llama.cpp -c / --ctx-size, and match --parallel so each slot still fits)")
        if schema is None or not isinstance(self.backend, SchemaBackend):
            return base
        if "sampler" in lowered or "grammar" in lowered or "json_schema" in lowered:
            hint = suggest_backend(self.config.model) or "bielik"
            return (f"{base}: the server rejected schema-constrained decoding, which some model "
                    f"builds cannot compile. Retry with a json_object backend (--backend {hint})")
        return base

    def complete_json(self, messages: Sequence[Message], schema: dict[str, Any], *,
                      role: str = "<none>", sampling: SamplingParams | None = None,
                      max_tokens: int = 1024, model: str | None = None) -> dict[str, Any]:
        """Chat completion that must yield a JSON object matching ``schema``'s top-level shape.

        A ``json_object`` backend (whose server does not constrain decoding) is held to the same
        shape as a grammar-constrained one. A reply that fails is an :class:`LlmContentError`,
        recorded against this record rather than shipped. A reply whose envelope was merely
        truncated is recovered and raised as :class:`LlmIncompleteJsonError` for the caller to
        treat as an unverified candidate.
        """
        text = self.complete(messages, role=role, sampling=sampling,
                             max_tokens=max_tokens, schema=schema, model=model)
        stripped = _strip_code_fence(text)
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            recovered = _close_truncated_json_object(stripped)
            if recovered is not None:
                try:
                    _check_schema_shape(recovered, schema, role=role, body=text)
                except LlmContentError:
                    pass  # Doesn't satisfy the schema either -- fall through to the plain error.
                else:
                    raise LlmIncompleteJsonError(
                        f"response JSON envelope was truncated: {exc}", role=role, body=text,
                        recovered=recovered) from exc
            raise LlmContentError(f"response was not valid JSON: {exc}",
                                  role=role, body=text) from exc
        _check_schema_shape(parsed, schema, role=role, body=text)
        return cast("dict[str, Any]", parsed)


def _estimate_tokens(text: str) -> int:
    """A rough, tokenizer-free token estimate, for a budget *warning* -- not a gate. CJK/wide
    characters count ~1 token each; everything else ~4 chars per token. Errs toward over-counting
    so the warning fires a little early. The server's own check remains authoritative."""
    wide = sum(1 for char in text if unicodedata.east_asian_width(char) in ("W", "F"))
    return wide + (len(text) - wide + 3) // 4


def _token_count(value: Any) -> int:
    """A token count from an untrusted usage block, or 0 if it is not a plain integer (bool
    excluded, being an int subclass a ``true`` in the field would otherwise read as 1)."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


_JSON_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def _json_type_matches(value: Any, declared: Any) -> bool:
    """Whether ``value`` has the JSON type ``declared`` names (a single name, or a union list).
    A type name this does not recognise is accepted, so the check tightens the common flat case
    without reimplementing JSON Schema."""
    if isinstance(declared, list):
        return any(_json_type_matches(value, item) for item in declared)
    check = _JSON_TYPE_CHECKS.get(declared) if isinstance(declared, str) else None
    return check is None or check(value)


def _check_schema_shape(parsed: Any, schema: dict[str, Any], *, role: str, body: str) -> None:
    """Raise :class:`LlmContentError` unless ``parsed`` is an object matching ``schema``'s
    top-level shape. Shared by the ordinary parse path and the truncated-envelope recovery path,
    so a recovered object is held to exactly the same standard as one that parsed cleanly."""
    if not isinstance(parsed, dict):
        raise LlmContentError(f"expected a JSON object, got {type(parsed).__name__}",
                              role=role, body=body)
    missing = set(schema.get("required", [])) - parsed.keys()
    if missing:
        raise LlmContentError(f"response missing required fields {sorted(missing)}",
                              role=role, body=body)
    mistyped = [
        f"{name!r} should be {spec['type']}, got {type(parsed[name]).__name__}"
        for name, spec in schema.get("properties", {}).items()
        if name in parsed and "type" in spec and not _json_type_matches(parsed[name], spec["type"])
    ]
    if mistyped:
        raise LlmContentError(f"response fields have the wrong type: {'; '.join(mistyped)}",
                              role=role, body=body)


def _close_truncated_json_object(text: str) -> dict[str, Any] | None:
    """Recover a flat JSON object from a reply that stopped one or two characters short of a valid
    envelope. **The recovered value is not known to be complete** -- appending ``"`` closes the
    string at an arbitrary cut point, so a value severed mid-clause yields as valid an object as
    one the model finished. That is exactly why the caller must treat the result as an unverified
    candidate (re-checked, re-reviewed, never certified), and why recovery is reported by raising
    :class:`LlmIncompleteJsonError`, never by returning into the normal path.

    Deliberately not a general repair: unbalanced brackets, a truncated nested container, or a
    trailing comma all still fail both attempts and return ``None``."""
    for suffix in ("}", '"}'):
        try:
            candidate = json.loads(text + suffix)
        except json.JSONDecodeError:
            continue
        # Both suffixes end in "}", and the only JSON document ending in "}" is an object.
        return cast("dict[str, Any]", candidate)
    return None


def _strip_code_fence(text: str) -> str:
    """Tolerate a model that wraps JSON in a markdown fence despite instructions. The opening
    ``` is dropped always; the closing fence only when actually present, so a fenced-and-truncated
    reply is not stranded with the fence attached (which would make the envelope recovery
    unreachable)."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) == 1:
        inner = stripped[3:]
        return (inner[:-3] if inner.endswith("```") else inner).strip()
    body = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
    return "\n".join(body)
