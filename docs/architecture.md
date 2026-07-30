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

- **`core`** is the contract: the ports (Protocols), the `Record` and its JSON Lines
  catalogue/journal (the import/export format at a run's edges — durability itself lives in
  `store`'s `RunStore`), the component registry, structured errors, the strict config loaders,
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
`meta` mapping for anything task-specific.

A run's durable state lives in a `RunStore` (see *Storage*, below), not in a file: `ragkit
import` loads a fetch script's JSONL catalogue into it once; `ragkit run` reads
`pending()`/writes `append_result()` against it, one ACID commit per result, so
`completed_ids()` after a crash reflects exactly what was actually committed — no torn write, no
replay; `ragkit export` writes the store's results back out as a `journal.jsonl`-compatible file
for a recipe's `eval.py`. JSON Lines is therefore the **import/export format at a run's edges**,
not the thing a run reads and writes while it executes. `write_catalog`/`read_journal` (in
`src/ragkit/core/records.py`) still do the file I/O for that bridge: `write_catalog` is atomic
(temp sibling → fsync → rename → fsync dir); `read_journal` tolerates a torn *final* record (a
process killed mid-write) but refuses a malformed one anywhere else, because skipping that would
discard a completed result while reporting success.

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

## Storage: three roles, real drivers per port

Three database *roles* are kept apart, never mixed behind one port:

- **External task data** — `SqlStore`, **read-only** (the DB an NL→SQL query or a form-autofill
  lookup reads; a write through it is refused at the port, before the database). Two real,
  interchangeable drivers: `sqlite` | `duckdb`. Unrelated to the framework's own state and never
  written by it.
- **Reference memory** — `PairingStore`, the framework's writable "what has been established"
  store: one row per `(source, target, context)` pairing plus metadata, retrieved by lexical/
  dense/hybrid search. See the next section for why this is co-located rather than split. Two
  real drivers: `sqlite` | `duckdb`; a `VectorIndex` (`lancedb` | `qdrant`), when configured, is a
  separate ANN store kept in sync via `VectorIndex.reconcile`.
- **Run state** — `RunStore`, the record catalogue and append-only result history a run reads and
  writes while it executes (WAL SQLite; see *The `Record`, and durability*, above).

A fourth, smaller DB-native store, `LexiconStore` (established terminology — a term/rendering
mapping, a different shape and key than a pairing), is usually co-located in the same physical
database file as `PairingStore` as its own table, though nothing requires that. Every port is
resolved through its own registry and bound by `storage.toml` (`[sql]`, `[pairings]`, `[vector]`,
`[run]`, `[lexicon]`); a `path` puts a store on disk, no `path` gives an in-memory store for tests
and small corpora. A conformance suite runs every driver of every port — including an in-memory
reference implementation — through identical operations, so a config-only driver swap (`sqlite`
↔ `duckdb` for `[pairings]`, say) is behaviourally proven, not just type-checked. A third party's
own driver is selected the same way, by dotted path.

## Reference memory: rows and search index, co-located

A search index answers only *which ids matched*: FTS5's BM25 index and the vector ANN each return
`(chunk_id, score)`, never text — resolving an id back into its display text + metadata is what
**`PairingStore.document()`** does, an indexed primary-key lookup, so a corpus of any size lives in
the database rather than a RAM map. What makes `PairingStore` DB-*native* rather than a derived
cache: the row and the FTS5 entry that indexes it are **one database, one transaction** — an
FTS5 *external-content* table kept in sync by `AFTER INSERT/UPDATE/DELETE` triggers inside the
same statement's implicit transaction as the row write (`store/pairings/sqlite.py`). There is no
window where a row exists without its index entry or vice versa, and an aborted write leaves
neither behind — the co-location invariant a *split* row store + search index (a relational row
table plus a separate FTS5 index, kept in sync only by write ordering) could only approximate,
never guarantee. `PairingStore` satisfies `SearchIndex` (`search`) directly, so `LexicalRetriever`
/ `DenseRetriever` / `HybridRetriever` (`retrieve/retrievers.py`, `retrieve/hybrid.py`) are reused
verbatim over it — there is exactly one retrieval code path, not one per store shape.

Import is built to match: `ingest/reference.py`'s `import_reference` streams a recipe's
`[reference].file` JSONL into the pairing store in batches, resumable from `PairingStore.count()`
as the durable floor — an interrupted multi-hour import costs seconds to resume, not a restart.
When a `[vector]` store is configured, each batch is embedded and upserted alongside, and
`VectorIndex.reconcile` closes any gap a crash left between the two. This is what lets the
recipes below ingest and query a multi-GB, multi-million-row corpus at a few tens of MB of
process RSS. `[reference].file` needs a `[pairings]` store configured — there is no in-memory or
split-store fallback; `tools/migrate_storage.sh` folds a pre-retirement recipe's on-disk artifacts
(a legacy row store, a `lexicon.jsonl`, a catalogue+journal pair) into the current stores directly,
without re-reading the original corpus.

## Where to look next

- **`docs/tutorial.md`** — build a pipeline from scratch and extend every seam, with runnable
  examples. Start here to *do* something.
- **`docs/config.md`** — the per-file configuration reference (every key, its type, default, and
  meaning). Wrong types and unknown keys are errors, not silent defaults.
- **`recipes/<task>/README.md`** — each of the worked recipes (translation, NL→SQL, form autofill,
  predictive-maintenance decisions, and the two large-corpus retrieval recipes `med_evidence` and
  `legal_procurement`), with the real dataset its `fetch.sh` pulls and the eval it runs.
- **`.audit/`** — the design investigations kept across sessions, including the circular-metric
  lesson the evaluation layer encodes.
