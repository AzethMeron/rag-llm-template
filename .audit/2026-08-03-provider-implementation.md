# 2026-08-03 — The `Provider` seam, made real

Closes the standing item from `2026-08-02-deep-audit.md` ("Interface / swappability verdict": *the
weakest seam*). Prompted by the owner: **"What about ollama, OpenAI support? ... If something isn't
supported (like rerank) in certain paths, it should raise error. Provider must be implemented."**

## What was wrong (verified against the source, not memory)

- **`Provider` (`core/ports.py`) was a dead Protocol** — a `chat(...) -> str` shape nothing
  implemented (`LlmClient` has a different, richer surface) and nothing called.
- **`EndpointSpec.provider` was validated then ignored.** `pool.py` checked it against a local
  frozenset and never read it again; `client_for` built an identical `LlmClient` for every
  provider. `ollama` / `openai-compatible` behaved bit-for-bit like `llamacpp-router`.
- **A llama.cpp-only field leaked to every server.** `client.complete` sent
  `chat_template_kwargs: {"enable_thinking": False}` whenever reasoning was off — a llama.cpp
  `--jinja` extension a strict OpenAI-compatible server rejects with a `400` on the whole request.
- **No capability was enforced.** `docs/config.md` merely *said* "Ollama cannot rerank"; nothing
  refused it, so a reranker pointed at Ollama would fail with a puzzling `404` mid-run.
- **`serveargs` rendered llama-server flags for any provider** — `--jinja`/router flags emitted even
  for an endpoint that is not a llama.cpp server we launch.

## The change

Implemented `Provider` as a real port with a concrete registry, mirroring the existing `Backend`
port / `backends.py` registry pattern exactly (the codec-registry exception the general registry's
own docstring sanctions). **Not** a full alternative transport: every real provider speaks the same
OpenAI-compatible HTTP, so a second transport would be speculative (YAGNI). The provider carries
only what the transport cannot infer from the wire — two axes:

| provider | capabilities | sends `chat_template_kwargs` |
|---|---|---|
| `llamacpp-router` (default) | chat, embedding, rerank | yes |
| `ollama` | chat, embedding | no |
| `openai-compatible` | chat, embedding, rerank | no |

- `llm/providers.py` — `LlmProvider` (declares `capabilities ⊆ {chat, embedding, rerank}` and the
  request quirk), a collision-refusing registry (`get_provider` / `register_provider` /
  `available_providers`), and the three built-ins. Extensible by dotted name like any component.
- `core/ports.py` — the `Provider` Protocol redefined to `capabilities` + `supports()` +
  `chat_payload_extras()`.
- `ServerConfig` carries a `Provider` (default `llamacpp-router`, so existing behaviour is
  unchanged); `client.complete` asks the provider for payload extras, so the client no longer knows
  the llama.cpp field exists — the per-family quirk lives on the provider.
- `EndpointSpec` validates via the registry (dropping the duplicated frozenset — one source of
  truth for the valid set) and exposes `provider_profile`; the pool threads it into every client.

**Unsupported paths raise at build time, not `404` later:**

- rerank build site (`cli/app.py`) refuses a provider without `rerank` (the real case — Ollama);
- embedding and chat build sites refuse a provider that cannot serve them (no built-in triggers
  these, but a custom provider can; the guards complete the capability model rather than leaving a
  rerank-only slice, and their refusal branches are covered by a synthetic provider that doubles as
  the registration-seam test);
- `serveargs.render_flags` refuses a non-`llamacpp-router` endpoint with a message pointing at how
  each other kind is actually reached (`ollama serve`; `base_url` for a hosted endpoint).

## Deliberately not done

- **No native Ollama / OpenAI transports.** All three are OpenAI-compatible HTTP; the differences
  are capabilities and payload quirks, which the profile captures. A separate transport would be
  speculative generality.
- **`openai-compatible` keeps `rerank`.** The bucket covers Jina/TEI/Cohere-style rerank servers
  (as `rerank.py`'s own docstring already says); only Ollama genuinely lacks the endpoint. A strict
  OpenAI endpoint with no reranker will surface that as a normal request error, not a config lie.

## Verification

Full suite green (1424 tests), 100% statement + branch coverage, `ruff` + `mypy --strict`
clean. New/updated tests: `tests/llm/test_providers.py` (registry + capability queries + the payload
split), and capability-refusal tests at every build site (`test_pool.py`, `test_app.py`,
`test_serveargs.py`) plus the client-side payload-extras split (`test_client.py`).
