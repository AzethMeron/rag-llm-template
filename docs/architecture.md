# Architecture

## What this is

A task-agnostic framework for building RAG + LLM applications on local models. It is the
generalisation of a translation engine whose durable, resumable *produce → mechanical check →
review panel → revise* harness turned out to have nothing to do with translation specifically.
The framework keeps that harness and makes every part around it — the databases, the retrieval
stack, the model layer — a replaceable component behind a typed interface.

The design goal, stated as a testable property: **you can replace any component's internals —
the LLM model, the retrieval algorithm, the vector database — by editing configuration, with
no change to any other part of the code**, and you can supply your *own* component the same
way without changing framework code at all.

## The layers, and the direction that must hold

```
cli  →  harness  →  {retrieve, llm, store, core}
                     retrieve → {llm, store, core}
                     ingest   → {llm, store, core}
                     llm → core        store → core        core → stdlib only
```

- **`core`** is the contract: the ports (Protocols), the `Record` and its durable
  catalogue/journal, the component registry, structured errors, the strict config loaders,
  and the stdlib primitives (display width, placeholders, terminology). It depends on nothing
  outside the standard library, so it can never break because a machine-learning or storage
  dependency was upgraded — or is absent.
- Each layer above may depend on the layers below it and never on a layer above. No layer
  imports a concrete driver *by name*; every driver is reached through a registry.

`tests/test_boundaries.py` enforces all of this by walking each module's imports, and imports
`core` in a subprocess with the optional dependencies made unimportable — the strongest form
of "the contract needs none of them".

## The one extension mechanism

Everything swappable is a **component** resolved through a `Registry` (one per port). A
component is selected in config three ways, in precedence order: an explicit dotted path
(`"mypkg.retrievers:MyRetriever"`), a published entry point (`ragkit.retrievers`), or a
registered built-in name. Each component declares its own allowed option keys, so a typo in a
*third-party* component's options is refused just as loudly as one in a built-in's. A registry
holds only component *types and their option schemas* — never per-run state — and resolution
is a pure function of `(registry, spec, options)`, which is how a module-level registry stays
compatible with the no-global-mutable-state rule. See `src/ragkit/core/registry.py`.

## The `Record`, and durability

A `Record` is one task instance — a line to translate, a question to turn into SQL, a form to
fill — carrying its input, its produced output, a status lifecycle
(`PENDING → PRODUCED/VERIFIED/REJECTED/SKIPPED`), provenance for reinjection, and a free-form
`meta` mapping for anything task-specific. Catalogues and journals are JSON Lines: streamable,
appendable, and resumable. `write_catalog` is atomic (temp sibling → fsync → rename → fsync
dir); `read_journal` tolerates a torn *final* record (a process killed mid-write) but refuses a
malformed one anywhere else, because skipping that would discard a completed result while
reporting success. See `src/ragkit/core/records.py`.

## Where each LLM setting lives

A decode setting is configured where it takes effect, so there is one obvious home for each:

- **Per-request settings → the persona.** Temperature and the other sampling knobs (`top_p`,
  `top_k`, `min_p`, `seed`, `presence_penalty`, `frequency_penalty`, `repeat_penalty`, `stop`) are
  applied per call, and the right value is a property of the *role* — a producer may want a little
  warmth, a reviewer wants determinism — so each `[[persona]]` carries its own optional
  `[persona.sampling]` sub-table. The default temperature depends on the role (producer `0.3`,
  reviewer `0.0`); every other knob is unset unless named, so only chosen settings reach the
  server and a strict-OpenAI endpoint is never handed a llama.cpp-only knob. These become a
  `SamplingParams` (a frozen value type in `core/ports.py`) threaded to the client. `max_tokens` is
  deliberately *not* a sampling knob: it is a computed output budget (scaled to the input for the
  producer, a ceiling for a reviewer), kept separate from decode *style*.
- **Launch-time settings → `models.toml`.** Anything a server can only apply when it starts (GPU
  offload, context size, KV-cache type, flash attention, rope scaling) is not per-request, so it
  lives on the endpoint that hosts the model: `[endpoint.<name>].server_args` is a verbatim flag
  list. `tools/serve_models.sh --config models.toml --endpoint <name>` reads the same
  `EndpointSpec` the pool uses (via `python -m ragkit.llm.serveargs`), so `models.toml` is the
  single source of truth for both routing and serving rather than a config file and a hand-written
  command line that can drift.

## Evaluation

Two separable layers, both in `ragkit.eval`, both built so a circular configuration is *refused*
rather than reported (the hard-won lesson recorded in `.audit/`): **a metric must be independent of
what it ranks.**

- **Retrieval** — `evaluate_retrieval` scores each system with `Recall@k`, `MRR`, `MAP`, and
  `NDCG@k` against `Qrels` (ground truth carrying a `source` label). It raises rather than run if
  the ground truth's source is a system under evaluation (it would score `1.0` by construction), or
  if an evaluated query has no gold judgments.
- **Output** — `evaluate_ab` is a blinded A/B judge: the LLM judge sees neutral "Output 1/2" in an
  injected deterministic order (no hidden RNG), never the system names, and A/B-ing a system against
  itself is refused.

## Storage: two roles, two real drivers per port

Two database *roles* are kept apart: the framework's own writable store (records/chunks/metadata),
and an **external, read-only** task data source (the DB that NL→SQL queries or form-autofill reads
— a write through it is refused at the port, before the database). Each storage port ships **two
real, interchangeable drivers** — `SqlStore` = `sqlite` | `duckdb`, `VectorIndex` = `lancedb` |
`qdrant`, `LexicalIndex` = `fts5` — so swapping a database is a one-line `storage.toml` edit,
proven by a conformance suite that runs every implementation through identical operations. A third
party's own driver is selected the same way, by dotted path.

## Where to look next

- **`docs/tutorial.md`** — build a pipeline from scratch and extend every seam, with runnable
  examples. Start here to *do* something.
- **`docs/config.md`** — the per-file configuration reference (every key, its type, default, and
  meaning). Wrong types and unknown keys are errors, not silent defaults.
- **`recipes/<task>/README.md`** — each of the four worked recipes (translation, NL→SQL, form
  autofill, predictive-maintenance decisions), with the real dataset its `fetch.sh` pulls and the
  eval it runs.
- **`.audit/`** — the design investigations kept across sessions, including the circular-metric
  lesson the evaluation layer encodes.
