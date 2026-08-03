# ragkit.llm API Reference

`ragkit.llm` is the model-serving/request layer: a synchronous HTTP client for a local,
OpenAI-compatible inference server (`ragkit.llm.client.LlmClient`), the per-model-family request
shaping that client consults rather than deciding for itself (`ragkit.llm.backends`), a pool that
routes each persona's logical model over one shared connection per endpoint while guarding against
resident-model thrash (`ragkit.llm.pool`), the error taxonomy that lets a caller distinguish "the
server is broken, stop the run" from "this one reply is bad, record and continue"
(`ragkit.llm.errors`), the shared classification of which HTTP failures are transient enough to
retry (`ragkit.llm.http`), and the launch-flag rendering that keeps `models.toml` the single source
of truth for both routing and serving (`ragkit.llm.serveargs`). It depends only on the core contract
(`ragkit.core`) and `httpx` — everything above it (the harness, recipes) talks to a model through
this layer, never directly to an HTTP endpoint.

## ragkit.llm.backends

Request-shaping for one family of local model, and the small registry that selects it. "The same
request" does not mean "the same bytes on the wire" across models served over an OpenAI-compatible
endpoint — the sharpest difference is structured output: Qwen accepts a strict `json_schema`
grammar, while some builds (Bielik, EuroLLM) reject it ("Failed to initialize samplers") but honour
`json_object` with the shape described in the prompt. A `BaseBackend` is one object per family that
shapes the request accordingly; the client never changes when the model does.

This module keeps a dedicated registry rather than using `ragkit.core.registry.Registry`, because
backends are stateless singletons carrying model *hints* and selection has a bespoke `"auto"` mode
(pick the backend whose hints match the served model id) that the general registry has no
equivalent for. This is the codec-registry pattern the general registry's own docstring names as
its bounded exception: written only at import, collision-refusing, holding no per-run state. User
extensibility is preserved — `register_backend` is public, and a config may also name a custom
backend by dotted path.

#### `DEFAULT_BACKEND`

`"generic"` — the backend used when a caller names none: grammar-constrained decoding, correct for
Qwen and most local servers. A model that rejects it selects another by name.

### BackendError

A backend was requested that does not exist, or two were registered under one name. Subclasses
`ragkit.core.errors.RagkitError` directly (no custom `__init__`); carries the inherited
`reason`/`context`.

### BaseBackend (ABC)

Request-shaping for one model family. Stateless and shared: one instance is registered under its
name (and aliases) and reused for every request. Constructed as `BaseBackend(name: str, *,
aliases: Sequence[str] = (), model_hints: Sequence[str] = ())` — note `BaseBackend` itself is
abstract (`structured_request` has no body); only its subclasses `SchemaBackend` and
`JsonObjectBackend` are instantiated.

**Attributes:**
- `name` (`str`): the backend's registered name; the constructor rejects an empty/whitespace-only name.
- `aliases` (`tuple[str, ...]`): additional names it also answers to.
- `model_hints` (`tuple[str, ...]`): lower-cased substrings of a served model id that point to this backend, used by `"auto"` selection (`resolve_backend`, `suggest_backend`) and the diagnostic hint when a structured request is rejected.

#### `structured_request(self, messages: Sequence[Message], schema: dict[str, Any]) -> StructuredRequest` (abstract)

Shape a request for output conforming to `schema` for this model.

**Args:**
- `messages` (`Sequence[Message]`): the conversation so far.
- `schema` (`dict[str, Any]`): the JSON Schema the reply must conform to.

**Returns:** `StructuredRequest` — the (possibly rewritten) messages and `response_format` fragment.

**Raises:** not implemented on the base class (`NotImplementedError` via `abstractmethod` if called directly on a subclass that doesn't override it).

#### `matches_model(self, model: str) -> bool`

**Args:**
- `model` (`str`): a served model id.

**Returns:** `bool` — whether any of `self.model_hints` occurs as a substring of `model.lower()`.

**Raises:** none.

### SchemaBackend

Grammar-constrained decoding: `response_format = json_schema`, strict — the strongest guarantee and
the default backend's implementation (`DEFAULT_BACKEND = "generic"` is registered as a
`SchemaBackend`). Messages are sent unchanged.

#### `structured_request(self, messages: Sequence[Message], schema: dict[str, Any]) -> StructuredRequest`

**Args:**
- `messages` (`Sequence[Message]`): the conversation so far.
- `schema` (`dict[str, Any]`): the JSON Schema to constrain decoding to.

**Returns:** `StructuredRequest` — `messages` unchanged, `response_format = {"type": "json_schema", "json_schema": {"name": "response", "strict": True, "schema": schema}}`.

**Raises:** none.

### JsonObjectBackend

Valid-JSON-only mode: `response_format = json_object`, with the schema's shape described in prose
and appended to the prompt. The reply is still validated against the schema by the client
(`LlmClient.complete_json`), so this is a weaker *constraint*, not a weaker *check*. Used for the
`bielik`, `eurollm`, and `gemma` built-in profiles.

#### `structured_request(self, messages: Sequence[Message], schema: dict[str, Any]) -> StructuredRequest`

**Args:**
- `messages` (`Sequence[Message]`): the conversation so far.
- `schema` (`dict[str, Any]`): the JSON Schema whose shape (field names and JSON types) is described in an appended instruction.

**Returns:** `StructuredRequest` — messages with a JSON-shape instruction appended to the last `"user"` turn (or as a new trailing user turn if none exists), `response_format = {"type": "json_object"}`.

**Raises:** none.

#### `register_backend(backend: BaseBackend) -> BaseBackend`

Register `backend` under its name and every alias. A name collision with a *different* backend is
an error, not a silent overwrite — two models answering to one name would make selection depend on
import order; re-registering the same instance is a no-op.

**Args:**
- `backend` (`BaseBackend`): the backend instance to register.

**Returns:** `BaseBackend` — `backend`, unchanged (for convenience/chaining).

**Raises:**
- `BackendError`: `backend.name` or one of its aliases is already registered to a *different* backend instance.

**Side effects:** mutates the module-level `_REGISTRY` dict.

#### `get_backend(name: str) -> BaseBackend`

Look up a backend by name or alias.

**Args:**
- `name` (`str`): a registered backend name or alias (case-insensitive).

**Returns:** `BaseBackend`.

**Raises:**
- `BackendError`: `name` is not registered; lists what is available.

#### `available_backends() -> list[str]`

**Args:** none.

**Returns:** `list[str]` — the distinct registered backend names, sorted.

**Raises:** none.

#### `suggest_backend(model: str) -> str | None`

**Args:**
- `model` (`str`): a served model id.

**Returns:** `str | None` — the name of the first registered backend whose `matches_model(model)` is `True`, or `None` if none match.

**Raises:** none.

#### `resolve_backend(name: str, model: str) -> BaseBackend`

Select a backend by `name`, or by the served `model` id when `name` is `"auto"`. `"auto"` is what
hardens a model switch: it picks the backend whose hints match the model, falling back to
`DEFAULT_BACKEND` when none do. The caller is expected to report which backend was chosen, so the
selection is never silent.

**Args:**
- `name` (`str`): a registered backend name, or `"auto"`.
- `model` (`str`): the served model id, consulted only when `name == "auto"`.

**Returns:** `BaseBackend`.

**Raises:**
- `BackendError`: (via `get_backend`) the resolved name is not registered.

## ragkit.llm.providers

The inference *providers* — the endpoint kinds the transport can talk to — and the registry that
selects one. Every provider speaks OpenAI-compatible HTTP, so one transport serves them all; the
provider carries what the transport cannot infer from the wire: which capabilities the endpoint
offers, and which request extras are safe to send. Concrete implementation of the
`ragkit.core.ports.Provider` port; the same codec-registry pattern as `ragkit.llm.backends`.

**Module constants:**
- `DEFAULT_PROVIDER` (`str`, `"llamacpp-router"`): the provider assumed when none is named.
- `CAPABILITIES` (`frozenset[str]`, `{"chat", "embedding", "rerank"}`): the roles an endpoint may fill.

Three built-ins are registered at import: `llamacpp-router` (all capabilities, sends
`chat_template_kwargs`), `ollama` (chat + embedding only, no template kwarg), `openai-compatible`
(all capabilities, no template kwarg — a strict server would reject it).

### ProviderError

`RagkitError` subclass: a provider was requested that does not exist, two were registered under one
name, or a provider was declared with an unknown capability.

### LlmProvider

One endpoint kind: its `name`, the `capabilities` it can serve, and its request quirks. Stateless
and shared, `slots`-based; satisfies the `Provider` port structurally. Constructed as
`LlmProvider(name: str, *, capabilities: frozenset[str] | set[str], sends_template_kwargs: bool)`.

**Attributes:**
- `name` (`str`): the registered name.
- `capabilities` (`frozenset[str]`): the served roles, a subset of `CAPABILITIES`.

**Raises (at construction):**
- `ProviderError`: the name is blank, or a declared capability is not in `CAPABILITIES`.

#### `supports(self, capability: str) -> bool`

Whether this endpoint kind can serve `capability`. An unknown capability name raises `ProviderError`
(a typo would otherwise read as an unsupported feature) rather than returning `False`.

#### `chat_payload_extras(self, *, enable_reasoning: bool) -> Mapping[str, Any]`

The extra chat-completion payload fields this endpoint kind understands. `llamacpp-router` returns
`{"chat_template_kwargs": {"enable_thinking": False}}` when `enable_reasoning` is `False` (and `{}`
when it is `True`); the others always return `{}`.

### register_provider(provider: LlmProvider) -> LlmProvider

Register `provider` under its name; returns it. A name collision with a *different* provider is a
`ProviderError`, not a silent overwrite; re-registering the same instance is a no-op.

### get_provider(name: str) -> LlmProvider

Look up a provider by name (case-insensitive), listing what is available when it is missing.

**Raises:** `ProviderError` — no provider registered under `name`.

### available_providers() -> list[str]

The registered provider names, sorted.

## ragkit.llm.client

Synchronous client for a local, OpenAI-compatible inference server. Model-agnostic on purpose:
everything that varies between models — above all how a structured request must be shaped — lives
in a `BaseBackend`, which this client consults rather than deciding for itself. The client owns
only the transport: the HTTP connection, retry with backoff, the error taxonomy, and usage
accounting. Prompts are assembled most-static-part-first by callers (system rules, then context,
then the input) so the server's prefix cache is reused across requests — the dominant throughput
factor for a large job on one loaded model.

#### `CONTEXT_WARN_FRACTION`

`0.8` — how full the context window may get (as a fraction of `ServerConfig.context_window`)
before `LlmClient` proactively warns: prompt plus output budget crossing this fraction means the
run is approaching truncation. The server's own overflow rejection is the hard stop; this is the
early warning before it.

### ServerConfig

Connection and decoding settings for the local server — purely about the transport. How a model
wants a structured request shaped is a backend concern, not a field here. A plain class using
`__slots__` (not a dataclass; the AST-derived checklist reports no dataclass fields for it), whose
`__init__` validates its own ranges.

**Attributes:**
- `base_url` (`str`, default `"http://127.0.0.1:8080/v1"`): the server's OpenAI-compatible base URL.
- `model` (`str`, default `"local"`): the model id sent when a call does not override it.
- `timeout_seconds` (`float`, default `300.0`): per-request HTTP timeout; must be `> 0`.
- `max_retries` (`int`, default `4`): attempts before giving up; must be `>= 1`.
- `retry_backoff_seconds` (`float`, default `2.0`): base backoff between retries (doubled each attempt); must be `>= 0` (a negative value would make `time.sleep` raise on the first retry).
- `context_window` (`int`, default `0`): the server's context size, for the proactive budget warning; `0` disables the warning. Must be `>= 0`.
- `enable_reasoning` (`bool`, default `False`): whether to let a reasoning model emit chain-of-thought. When `False` **and** the provider honours it, `complete` adds the provider's reasoning-suppression extras (llama.cpp's `chat_template_kwargs: {"enable_thinking": False}`); a provider that does not honour the field adds nothing.
- `provider` (`Provider`, default `get_provider("llamacpp-router")`): the endpoint kind. Consulted by `complete` for provider-specific request extras (see `chat_payload_extras`), so a llama.cpp-only field never reaches a server that would reject it.

**Raises (at construction):**
- `ValueError`: `max_retries < 1`; `timeout_seconds <= 0`; `retry_backoff_seconds < 0`; or `context_window < 0`.

### UsageStats

Cumulative token and latency accounting, for throughput reporting. One client is shared by every
worker in a concurrent run, so all mutation is guarded by a lock — a bare `+=` would lose updates.
A (non-frozen) dataclass.

**Attributes:**
- `requests` (`int`, default `0`): total completed requests recorded.
- `prompt_tokens` (`int`, default `0`): cumulative prompt tokens across all recorded requests.
- `peak_prompt_tokens` (`int`, default `0`): the largest single request's prompt token count seen.
- `completion_tokens` (`int`, default `0`): cumulative completion tokens.
- `seconds` (`float`, default `0.0`): cumulative wall-clock time across recorded requests.
- `retries` (`int`, default `0`): number of retry attempts recorded.
- `refusals` (`int`, default `0`): number of model refusals recorded.
- `_lock` (`threading.Lock`, default `field(default_factory=threading.Lock, repr=False, compare=False)`): internal — guards all mutation; excluded from `repr`/equality.

#### `record(self, usage: dict[str, Any], elapsed: float) -> None`

**Args:**
- `usage` (`dict[str, Any]`): the server's `"usage"` block (untrusted — a `null`/non-numeric count degrades to `0` rather than raising, since a malformed usage block must not sink a request that otherwise succeeded).
- `elapsed` (`float`): wall-clock seconds the request took.

**Returns:** `None`.

**Raises:** none.

**Side effects:** mutates `self.requests`, `self.prompt_tokens`, `self.peak_prompt_tokens`, `self.completion_tokens`, `self.seconds` under `self._lock`.

#### `record_retry(self) -> None`

**Args:** none. **Returns:** `None`. **Raises:** none.

**Side effects:** increments `self.retries` under `self._lock`.

#### `record_refusal(self) -> None`

**Args:** none. **Returns:** `None`. **Raises:** none.

**Side effects:** increments `self.refusals` under `self._lock`.

#### `completion_tokens_per_second(self) -> float` (property)

**Args:** none.

**Returns:** `float` — `completion_tokens / seconds`, or `0.0` if `seconds` is `0`.

**Raises:** none.

### LlmClient

Synchronous chat client with retry and backend-directed structured output. Owns its HTTP
connection; use as a context manager (`with LlmClient(...) as client:`) so the socket is released
deterministically. The backend is fixed for the client's lifetime — one loaded model, one request
shape. Constructed as `LlmClient(config: ServerConfig | None = None, *, backend: BaseBackend |
None = None, client: httpx.Client | None = None)`; `config` defaults to `ServerConfig()`, `backend`
to `get_backend(DEFAULT_BACKEND)`, and `client` is injectable (e.g. wrapping an
`httpx.MockTransport`) so a test drives the client with no real server — when injected, `close()`
does not close it (the injector owns it).

#### `close(self) -> None`

Release the HTTP client if this object opened it; an injected one is the injector's.

**Args:** none.

**Returns:** `None`.

**Raises:** none.

**Side effects:** closes `self._client` iff `self._owns_client` (i.e. no `client` was injected at construction).

#### `health(self) -> bool`

**Args:** none.

**Returns:** `bool` — `True` when a `GET {base_url}/models` (10s timeout) returns HTTP 200; `False` on any `httpx.HTTPError` or a non-200 status.

**Raises:** none (transport errors are caught and turned into `False`).

**Side effects:** issues one HTTP GET request.

#### `complete(self, messages: Sequence[Message], *, role: str = "<none>", sampling: SamplingParams | None = None, max_tokens: int = 1024, schema: dict[str, Any] | None = None, model: str | None = None) -> str`

Run one chat completion, retrying transient failures. When `schema` is given, the backend decides
how the request asks for conforming JSON.

**Args:**
- `messages` (`Sequence[Message]`): the conversation to send.
- `role` (`str`, default `"<none>"`): a label for the caller's persona/role, carried into error context and the budget-warning log line.
- `sampling` (`SamplingParams | None`, default `None`): per-request decode settings (the persona's); defaults to a module-level `SamplingParams()` (temperature `0.2`, nothing else set) when omitted.
- `max_tokens` (`int`, default `1024`): the output token budget for this call.
- `schema` (`dict[str, Any] | None`, default `None`): a JSON Schema; when given, `self.backend.structured_request` shapes the request. `None` sends a plain chat completion.
- `model` (`str | None`, default `None`): overrides `self.config.model` for this call (the model pool passes a per-persona model over one shared client).

**Returns:** `str` — the model's reply text (`choices[0].message.content`).

**Raises:**
- `LlmRefusalError`: the server returns `content: null` (also records a refusal via `self.stats.record_refusal()`).
- `LlmTruncationError`: `finish_reason == "length"` — generation hit the `max_tokens` ceiling before finishing (usually a repetition loop); not retried, since the ceiling is a property of this prompt.
- `LlmError`: a non-transient HTTP status (a deterministic 4xx — 400/401/403/404/422 — not in `ragkit.llm.http.TRANSIENT_HTTP_STATUS`; retrying cannot help, so it is raised immediately); or every retry attempt is exhausted (accumulated from transport failures, transient-status responses in `TRANSIENT_HTTP_STATUS` — 429 and 408 as well as the 5xx family — or malformed response bodies — missing `choices`/`message`/`content` or invalid JSON).

**Side effects:** issues one or more HTTP POSTs to `{base_url}/chat/completions`; sleeps between retries (`retry_backoff_seconds * 2**(attempt-1)`); records retries/usage into `self.stats`; may log one budget warning (`logger.warning`, at most once per client instance, guarded by `self._context_lock`) when the estimated prompt + `max_tokens` nears `CONTEXT_WARN_FRACTION` of `config.context_window`.

#### `complete_json(self, messages: Sequence[Message], schema: dict[str, Any], *, role: str = "<none>", sampling: SamplingParams | None = None, max_tokens: int = 1024, model: str | None = None) -> dict[str, Any]`

Chat completion that must yield a JSON object matching `schema`'s top-level shape. A
`json_object` backend (whose server does not constrain decoding) is held to the same shape
check as a grammar-constrained one.

**Args:**
- `messages` (`Sequence[Message]`): the conversation to send.
- `schema` (`dict[str, Any]`): the JSON Schema the parsed reply's top level must satisfy (`required` fields present, declared `type`s matched via `ragkit.core.jsonshape.json_type_matches`).
- `role` (`str`, default `"<none>"`): label for error context and logging.
- `sampling` (`SamplingParams | None`, default `None`): per-request decode settings.
- `max_tokens` (`int`, default `1024`): output token budget.
- `model` (`str | None`, default `None`): overrides the configured model id.

**Returns:** `dict[str, Any]` — the parsed JSON object.

**Raises:**
- `LlmContentError`: the reply (after stripping a markdown code fence) is not valid JSON and cannot be recovered as a truncated-but-closeable object either; the parsed value is not a `dict`; required fields are missing; or a field's value does not match its schema-declared type. Also everything `complete` can raise, propagated unchanged.
- `LlmIncompleteJsonError`: the JSON envelope was cut short (end-of-sequence inside or just after the output value) but appending `"}"` or `'"}'`  closes it into an object that *does* satisfy the schema shape check. Carries `.recovered` — an **unverified candidate** the caller must re-check/re-review, never certify, since the closing character was chosen arbitrarily and the content itself may still be truncated mid-clause.

**Side effects:** same as `complete` (one call to it).

#### `_estimate_tokens`, `_token_count`, `_check_schema_shape`, `_close_truncated_json_object`, `_strip_code_fence`

Module-private helpers behind `complete`/`complete_json` (not part of the public API, so not
individually documented as top-level entries): a tokenizer-free token estimate for the budget
warning (over-counts on purpose); a defensive int-or-zero reader for the server's untrusted usage
block; the shared schema-shape check used by both the clean-parse and truncated-recovery paths;
the truncated-envelope recovery (`text + "}"` / `text + '"}'`); and markdown-code-fence stripping
tolerant of a fence that is itself truncated.

## ragkit.llm.errors

The model-layer error taxonomy, typed by **blast radius** — the single most important distinction
in the engine, because it decides whether one bad request stops a whole run. `LlmError` means the
transport or server is broken, so every remaining request would fail the same way: it propagates
and the run stops with its journal intact. `LlmContentError` means *this* request's output is
unusable (unparseable JSON, a repetition loop, a refusal): it says nothing about the next request,
so the caller records the failure and continues. Turning the second into the first once burned
every remaining record on a run that still exited 0 — the hierarchy is load-bearing, not cosmetic.
All subclass `ragkit.core.errors.RagkitError`, so a caller can also catch "any framework error."

### LlmError

A request to the inference server failed or returned unusable output. The base of the taxonomy
and the *broad* blast radius: absent a more specific subclass, the server or transport is broken
and the run should stop.

**Attributes:**
- `role` (`str`, default `"<none>"`): the caller's persona/role label.
- `status` (`int | None`, default `None`): the HTTP status code, when the failure came from a response.
- `body` (`str`, default `""`): the full response body as given to the constructor (note: the `context` passed to `RagkitError`/rendered in the message is truncated to the first 400 characters; `self.body` itself is the untruncated value).

#### `LlmError(reason: str, *, role: str = "<none>", status: int | None = None, body: str = "") -> None`

**Args:**
- `reason` (`str`): what went wrong.
- `role` (`str`, default `"<none>"`): the persona/role label.
- `status` (`int | None`, default `None`): HTTP status, if applicable.
- `body` (`str`, default `""`): response body text.

**Returns:** none (constructor).

### LlmContentError

The server answered, but *this* request's output is unusable. Distinguished from its base by blast
radius: a bare `LlmError` means every remaining request would fail alike; this means the model
produced garbage for one particular input. No custom `__init__` — inherits `LlmError`'s
`(reason, *, role, status, body)` and attributes unchanged.

### LlmRefusalError

The model declined to produce output. Counted separately from other malformed output (via
`UsageStats.record_refusal`) so a caller can distinguish a refusal from a repetition loop. No
custom `__init__` — inherits `LlmContentError`'s.

### LlmTruncationError

Generation hit the token ceiling before the model finished. Its own type because the fix differs:
the text is not wrong, it is unfinished — usually a repetition loop on a hard input. Not retried,
since the ceiling is a property of this prompt. No custom `__init__` — inherits
`LlmContentError`'s.

### LlmIncompleteJsonError

The reply's JSON envelope was cut short, but its content parses once closed. A narrow, verified
failure shape: the model emitted end-of-sequence inside or just after the output value, before the
object's closing brace. The envelope is repaired and carried in `.recovered`, but **the content may
still be cut off mid-clause**, and nothing at this layer can tell the two apart — so the caller
must treat `.recovered` as an unverified *candidate* (re-check and re-review it, never certify it).
Because it is still an `LlmContentError`, a caller that does nothing special still gets the safe
behaviour (record and continue).

**Attributes:**
- `recovered` (`dict[str, Any]`): the repaired JSON object; unverified beyond matching the schema's declared shape.
- (inherits `role`, `status`, `body` from `LlmError`.)

#### `LlmIncompleteJsonError(reason: str, *, role: str = "<none>", body: str = "", recovered: dict[str, Any]) -> None`

**Args:**
- `reason` (`str`): what went wrong.
- `role` (`str`, default `"<none>"`): the persona/role label.
- `body` (`str`, default `""`): the raw (truncated) response text.
- `recovered` (`dict[str, Any]`, required keyword): the repaired-but-unverified JSON object.

**Returns:** none (constructor).

## ragkit.llm.http

Transient-vs-deterministic HTTP failure classification, shared by every client that talks to a
model-serving endpoint — chat (`ragkit.llm.client`), embeddings and rerank (`ragkit.retrieve`). One
home so all three retry the *same* failures the same way, rather than each client deciding
differently (the chat client used to raise on a 429 the embedding client retried, and the rerank
client retried nothing at all). Depends only on `httpx`.

#### `TRANSIENT_HTTP_STATUS`

A `frozenset[int]` of the HTTP status codes worth retrying with backoff: `{408, 429, 500, 502, 503,
504}` — rate-limiting (429), request-timeout (408), and the transient 5xx a busy/overloaded server
returns (llama.cpp answers 503 when every `--parallel` slot is in use; a cloud OpenAI-compatible
endpoint answers 429 under load). A genuinely deterministic 4xx (400/401/403/404/422) is
deliberately excluded — retrying it cannot help.

#### `is_transient_http_error(exc: httpx.HTTPError) -> bool`

Whether `exc` is a transient failure worth retrying with backoff: a timeout, a dropped connection,
or a `TRANSIENT_HTTP_STATUS` response (from `raise_for_status`). A deterministic status error or any
other `httpx.HTTPError` returns `False`.

**Args:**
- `exc` (`httpx.HTTPError`): the transport or status error to classify.

**Returns:** `bool` — `True` for an `httpx.HTTPStatusError` whose `response.status_code` is in `TRANSIENT_HTTP_STATUS`, or for any `httpx.TimeoutException`/`httpx.TransportError`; `False` for any other `httpx.HTTPError`.

**Raises:** none.

## ragkit.llm.pool

The model pool: one connection per endpoint, per-persona model routing, and a thrash guard. Each
persona (a reviewer, the producer) names a logical model; several personas may name the same one,
and different ones may name different models. The pool owns one HTTP connection per endpoint and
hands out one `LlmClient` per logical model over it, so personas sharing a model share a client
(and, on a router-mode server, share the loaded weights).

The correction to the common assumption is enforced here: a router (llama.cpp router mode, Ollama)
keeps only so many distinct models resident (`resident_max`) and evicts the rest by LRU. With more
distinct models in use than the cap, a round-robin over personas makes every request evict the
model the next one needs — silent thrash. `ModelPool.check_capacity` turns that into a load-time
refusal, naming the models and the two fixes, before the server is ever contacted. It counts
*distinct models*, not personas — persona count costs KV-cache slots, not weight memory.

### ModelPoolError

The model configuration is inconsistent, or its capacity would thrash. Subclasses
`ragkit.core.errors.RagkitError` directly (no custom `__init__`); carries the inherited
`reason`/`context`.

### EndpointSpec

One inference endpoint: where it is, and how many models it holds resident. Frozen, `slots`-based
dataclass; `__post_init__` validates its own invariants.

**Attributes:**
- `name` (`str`): the endpoint's name (matches its `[endpoint.<name>]` table).
- `provider` (`str`, default `"llamacpp-router"`): the endpoint kind, a registered provider name (`"llamacpp-router"`, `"ollama"`, `"openai-compatible"`, or a custom one registered via `register_provider`). Validated at construction; resolved to its `LlmProvider` by `provider_profile`, which drives capability gating and request quirks.
- `base_url` (`str`, default `"http://127.0.0.1:8080/v1"`): the endpoint's OpenAI-compatible base URL.
- `resident_max` (`int`, default `4`): how many distinct models stay resident before the server evicts by LRU (llama.cpp `--models-max`, Ollama `OLLAMA_MAX_LOADED_MODELS`) — the thrash guard's ceiling. Must be `>= 1`.
- `parallel` (`int`, default `2`): server request slots; must be `>= 1`, and must be `>=` the harness concurrency or in-flight requests serialise.
- `vram_budget_mb` (`int`, default `0`): optional VRAM ceiling for the byte-level guard; `0` disables it (count-only). Must be `>= 0`.
- `timeout_seconds` (`float`, default `300.0`): per-request HTTP timeout for clients built against this endpoint.
- `max_retries` (`int`, default `4`): retry ceiling for clients built against this endpoint.
- `retry_backoff_seconds` (`float`, default `2.0`): retry backoff base for clients built against this endpoint.
- `enable_reasoning` (`bool`, default `False`): whether models on this endpoint may emit chain-of-thought.
- `server_args` (`tuple[str, ...]`, default `()`): launch-time flags for the server hosting this endpoint (GPU offload, KV-cache type, rope scaling, flash attention, ...) — not per-request; `serve_models.sh` reads these via `ragkit.llm.serveargs` so `models.toml` is the single source of truth for both routing and serving. The pool itself never launches a server.

**Properties:**
- `provider_profile` (`LlmProvider`): the resolved provider for this endpoint's `provider` name — its capabilities and request quirks. A registry lookup validated at construction, so it cannot fail for a constructed spec.

**Raises (at construction):**
- `ModelPoolError`: `provider` is not a registered provider name; `resident_max < 1`; `parallel < 1`; or `vram_budget_mb < 0`.

### ModelSpec

One logical model a persona can name: which endpoint serves it, its served id, and how to shape a
structured request for it. Frozen, `slots`-based dataclass; `__post_init__` validates its own
invariants.

**Attributes:**
- `name` (`str`): the logical model name a persona refers to.
- `endpoint` (`str`): the `EndpointSpec.name` that serves this model.
- `model_id` (`str`): the id sent to the server (the served model identifier).
- `backend` (`str`, default `"auto"`): a registered `ragkit.llm.backends` name, or `"auto"` to pick by `model_id`.
- `kind` (`str`, default `"chat"`): one of `"chat"`, `"embedding"`, `"rerank"`.
- `context_window` (`int`, default `0`): the model's context size, passed to `ServerConfig.context_window` for the budget warning. Must be `>= 0`.
- `approx_vram_mb` (`int`, default `0`): optional resident weight size in MB, for the byte-level thrash guard; `0` means undeclared — the guard then counts models only and says so, rather than implying a check it did not do. Must be `>= 0`.

**Raises (at construction):**
- `ModelPoolError`: `kind` is not one of `{"chat", "embedding", "rerank"}`; `context_window < 0`; or `approx_vram_mb < 0`.

### ModelPool

Owns one HTTP connection per endpoint and routes each logical model over it. Constructed as
`ModelPool(endpoints: Mapping[str, EndpointSpec], models: Mapping[str, ModelSpec], *,
client_factory: ClientFactory | None = None)`; `client_factory` builds the `httpx.Client` for an
endpoint (given its `base_url` and `timeout`), letting a test inject one wrapping a
`MockTransport`. The pool closes every client it opened; an injected transport's lifecycle is the
test's. Thread-safe lazily: concurrent workers may all call `client_for` for the same persona
model at once on a cold pool, so both internal caches (`_http`, `_clients`) are built under a
reentrant lock (`threading.RLock`) — an unguarded check-then-set previously let several threads
each build a client for the same key, leaking every loser's connection pool and splitting usage
stats across discarded clients.

**Raises (at construction):**
- `ModelPoolError`: any `model.endpoint` in `models` is not a key of `endpoints`.

#### `check_capacity(self, active_models: Iterable[str]) -> None`

Refuse, before any server is contacted, a set of in-use models that would thrash. For each
endpoint, the distinct models among `active_models` must not exceed `resident_max`. When every
such model declares `approx_vram_mb` and the endpoint sets `vram_budget_mb`, their sum must also
fit.

**Args:**
- `active_models` (`Iterable[str]`): the logical model names about to be used concurrently (e.g. every persona's model in a run).

**Returns:** `None`.

**Raises:**
- `ModelPoolError`: a name in `active_models` is not a defined model; an endpoint would hold more distinct `model_id`s than its `resident_max` (names the models and both fixes: raise `resident_max`, or use fewer distinct models); or the endpoint's `vram_budget_mb` is set, every resident model on it declares `approx_vram_mb`, and their sum exceeds the budget.

#### `client_for(self, model_name: str) -> tuple[LlmClient, str]`

The `(client, served_model_id)` for a logical chat model, building it once per model (personas
sharing a model share the client) over the endpoint's shared connection.

**Args:**
- `model_name` (`str`): a logical model name defined in `models.toml`.

**Returns:** `tuple[LlmClient, str]` — the shared client for this model, and its served `model_id`.

**Raises:**
- `ModelPoolError`: `model_name` is not a defined model; it is defined with `kind != "chat"` (the persona pool serves chat models only — embedding/rerank models are used by the retrieval layer); or its endpoint's provider does not serve chat.

**Side effects:** lazily builds and caches an `httpx.Client` per endpoint and an `LlmClient` per model, under `self._build_lock`. The built client's `ServerConfig` carries the endpoint's resolved provider, so provider-specific request extras are applied.

#### `model_spec(self, model_name: str) -> ModelSpec`

**Args:**
- `model_name` (`str`): a logical model name.

**Returns:** `ModelSpec`.

**Raises:**
- `ModelPoolError`: `model_name` is not defined.

#### `endpoint(self, name: str) -> EndpointSpec`

**Args:**
- `name` (`str`): an endpoint name.

**Returns:** `EndpointSpec`.

**Raises:**
- `ModelPoolError`: `name` is not a defined endpoint.

#### `models_of_kind(self, kind: str) -> dict[str, ModelSpec]`

Every model of a given kind, for the retrieval layer to find its embedding/rerank models.

**Args:**
- `kind` (`str`): one of `"chat"`, `"embedding"`, `"rerank"`.

**Returns:** `dict[str, ModelSpec]` — name to spec, for every model whose `kind` matches.

**Raises:**
- `ModelPoolError`: `kind` is not one of the known kinds.

#### `close(self) -> None`

Close every HTTP connection the pool opened. The `LlmClient`s share these, so they are not closed
individually (each was given an injected client it does not own).

**Args:** none. **Returns:** `None`. **Raises:** none.

**Side effects:** closes and clears every cached `httpx.Client` and clears the `LlmClient` cache, under `self._build_lock`. Also usable via the `with ModelPool(...) as pool:` context-manager protocol, which calls this on exit.

#### `load_models(path: Path, *, client_factory: ClientFactory | None = None) -> ModelPool`

Build a `ModelPool` from a `models.toml`. Endpoints are `[endpoint.<name>]` tables, logical models
`[model.<name>]` tables. Unknown keys and mistyped values are refused, per the framework's config
rules; range checks live on the dataclasses.

**Args:**
- `path` (`Path`): the `models.toml` file.
- `client_factory` (`ClientFactory | None`, default `None`): passed through to `ModelPool`.

**Returns:** `ModelPool`.

**Raises:**
- `ConfigError`: `path` is missing/invalid TOML (via `load_toml`); the top level has a key other than `endpoint`/`model`; the `[endpoint]`/`[model]` section is not a table; any `[endpoint.<name>]`/`[model.<name>]` table has an unknown key or a mistyped value (via the `read_*` helpers); a `[model.<name>]` is missing `model_id` or `endpoint`; no `[endpoint.<name>]` tables are defined at all; no `[model.<name>]` tables are defined at all; or constructing the resulting `EndpointSpec`/`ModelSpec`/`ModelPool` raises `ModelPoolError` (caught and re-raised as `ConfigError` naming the file).

**Side effects:** reads and parses `path`.

## ragkit.llm.serveargs

Render the `llama-server` launch command for an endpoint from `models.toml`, so the serve script
and the model pool read one source of truth. The pool consumes an endpoint's *routing* facts
(`base_url`, `resident_max`); the serve script needs the *launch* facts (host, port, models-max,
and the endpoint's `server_args`) — both come from the same `EndpointSpec`, so a change to
`models.toml` moves both together rather than drifting between a config file and a hand-written
command line. Invoked as `python -m ragkit.llm.serveargs --config models.toml --endpoint local`,
which prints the flags, one per line, for `serve_models.sh` to read into an array. Router mode
launches with *no* model, so only host/port/models-dir/models-max plus the passthrough flags are
emitted; the model files themselves live in `--models-dir`.

### ServeArgsError

The requested endpoint cannot be turned into a launch command. Subclasses
`ragkit.core.errors.RagkitError` directly (no custom `__init__`); carries the inherited
`reason`/`context`.

#### `host_port(base_url: str) -> tuple[str, int]`

The host and port a server must bind to answer `base_url`. A URL without an explicit port is
refused rather than guessed: serving on the wrong port silently is exactly the class of failure
the framework rejects.

**Args:**
- `base_url` (`str`): the endpoint's base URL, e.g. `"http://127.0.0.1:8080/v1"`.

**Returns:** `tuple[str, int]` — `(hostname, port)`.

**Raises:**
- `ServeArgsError`: `base_url` has no parseable host; or has no explicit port.

#### `render_flags(endpoint: EndpointSpec, *, models_dir: str) -> list[str]`

The `llama-server` router-mode flags for `endpoint`: bind address, resident cap, models directory,
then the endpoint's own launch `server_args` verbatim.

**Args:**
- `endpoint` (`EndpointSpec`): the endpoint to render flags for.
- `models_dir` (`str`): directory of GGUF model files, passed through as `--models-dir`.

**Returns:** `list[str]` — flags in order: `--host`, `--port`, `--models-dir`, `--models-max`, `--jinja`, then `endpoint.server_args` appended verbatim.

**Raises:**
- `ServeArgsError`: `endpoint.provider` is not `"llamacpp-router"` (this renders a llama-server command; an Ollama or hosted OpenAI-compatible endpoint is not launched here); or (via `_validate_server_args_files`) `endpoint.server_args` contains a file-taking flag (currently `--models-preset`) with no following value, or whose value does not exist as a file relative to the current working directory; or (via `host_port`) `endpoint.base_url` has no host or no explicit port.

#### `flags_for(config: Path, endpoint_name: str, *, models_dir: str) -> list[str]`

**Args:**
- `config` (`Path`): the `models.toml` file to load.
- `endpoint_name` (`str`): the `[endpoint.<name>]` to render flags for.
- `models_dir` (`str`): directory of GGUF model files.

**Returns:** `list[str]` — same as `render_flags`, for the named endpoint.

**Raises:**
- `ServeArgsError`: `endpoint_name` is not defined in `config` (any other `RagkitError` from `load_models`/`pool.endpoint` is caught and re-raised as `ServeArgsError`); or anything `render_flags` raises.
- `ConfigError`: propagated unchanged from `load_models` for a malformed `config` file (not caught by the `except RagkitError` re-wrap only because... note: `ConfigError` *is* a `RagkitError`, so it is in fact caught and re-raised as `ServeArgsError` here too — `flags_for` does not distinguish the two, it treats every `RagkitError` from loading/resolving the config as a `ServeArgsError`).

#### `main(argv: list[str] | None = None) -> int`

CLI entry point: `python -m ragkit.llm.serveargs --config PATH --endpoint NAME --models-dir DIR`.
Prints the rendered flags, one per line, to stdout.

**Args:**
- `argv` (`list[str] | None`, default `None`): argument vector; `None` means use `sys.argv[1:]` (via `argparse`).

**Returns:** `int` — `0` on success, `1` if a `RagkitError` (including `ServeArgsError`/`ConfigError`) was raised while computing the flags.

**Raises:** none explicitly — `argparse` itself may exit the process (`SystemExit`) on bad CLI arguments (e.g. missing `--config`); any `RagkitError` computing the flags is caught and turned into an `error: ...` line on stderr plus return code `1` rather than propagating.

**Side effects:** prints to stdout on success, or an `error: ...` line to stderr on failure; does not itself call `sys.exit` (the module's `if __name__ == "__main__":` guard does, via `raise SystemExit(main())`).
