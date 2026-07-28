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

## Still to come (built milestone by milestone)

The storage layer (a real vector database — LanceDB by default — plus SQLite/FTS5 and a
separate read-only external data source), the ingestion and retrieval stack (RRF fusion,
reranking, MMR), the local model layer (llama.cpp router mode, a model pool with a load-time
thrash guard), the persona/panel/validator harness, and three fully-worked task recipes each
with a script that fetches real data on demand. This document will grow with each.
