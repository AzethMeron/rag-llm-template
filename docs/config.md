# Configuration reference

A run is described entirely by the TOML files in one config directory. Every component — model,
persona, validator, context block, store, output schema — is resolved through a registry, so your
own driver named by dotted path is selected exactly like a built-in.

**Wrong types and unknown keys are errors, not silent defaults.** Every loader rejects a key it does
not recognise (naming the allowed set), refuses a `bool` where a number is expected (TOML's
`true` would otherwise read as `1`), and names the `[[section]]`-vs-`[section]` mistake. Range checks
live on the value objects, so a directly-constructed object is as impossible to make invalid as a
loaded one. `required` below means there is no default.

The files: [`models.toml`](#modelstoml) · [`personas.toml`](#personastoml) ·
[`rules.toml`](#rulestoml) · [`context.toml`](#contexttoml) · [`storage.toml`](#storagetoml) ·
[`recipe.toml`](#recipetoml) · [`retrieval.toml`](#retrievaltoml). A recipe's directory holds the
subset it needs.

---

## How to read these files

These configs use five TOML table forms, and every section header below is one of them. Knowing
which is which tells you whether a section is singular, repeatable, or something you name.

| Form | Written | Means | Example |
|---|---|---|---|
| **Table** | `[limits]` | A single named section; appears at most once. | `[limits]`, `[revision]`, `[context.budget]` |
| **Keyed table** | `[endpoint.<name>]` | A table whose last segment is a **name you choose**; repeat with different names, then reference the name elsewhere. | `[endpoint.local]`, `[model.author]` |
| **Nested / sub-table** | `[parent.child]` | A table that belongs to its parent. For a keyed parent, `[model.author]` then `[persona.sampling]`. Attaches to the most recently opened parent. | `[retrieval.dense]`, `[persona.sampling]` |
| **Array of tables** | `[[persona]]` | A **repeatable** entry; each `[[persona]]` block is one more item in an ordered list. **File order is meaningful** (e.g. reviewer consultation order). | `[[persona]]`, `[[forbidden]]`, `[[advisory]]`, `[[context.block]]`, `[[validator]]` |
| **Inline table** | `leniency = {window = 20, max_bad = 2}` | A small table written on one line, for a value that is itself a few fields. | `leniency`, `output_schema_options` |

Three conventions run through all of them:

- **`<name>` is yours to pick.** In `[endpoint.<name>]` / `[model.<name>]`, the `<name>` is a label
  you invent (`local`, `author`, `reviewer`); other files refer back to it (a persona's `model = "author"`
  points at `[model.author]`; a model's `endpoint = "local"` points at `[endpoint.local]`).
- **`kind` selects a component; the remaining keys are *its* options.** Wherever a section carries a
  `kind` (context blocks) or `driver` (stores) or a bare validator `kind`, that value chooses a
  registered component, and every other key in the section is validated against *that component's*
  own option set — an unknown option is rejected naming the component. A dotted path
  (`kind = "mypkg.blocks:MyBlock"`, `driver = "mypkg:MyStore"`) or an entry-point name selects a
  third-party component the same way, with no framework change.
- **Type notation in the tables below.** `int ≥ 1`, `float in [0,1]`, `array of strings`, and
  `A \| B \| C` (one of a fixed set) are constraints the loader enforces; `required` means there is
  no default. A range violation is refused whether the object is loaded from TOML or constructed
  directly.

The repeatable (`[[...]]`) tags read as ordered lists. A minimal shape for each:

```toml
# personas.toml — panel is consulted top-to-bottom, stopping at the first objection
[[persona]]
id = "author"
kind = "producer"
model = "author"
instructions = "..."

[[persona]]                     # a second block = the next panel member
id = "grammar"
kind = "reviewer"
model = "reviewer"
instructions = "..."

# rules.toml — each block is one pattern / one prose criterion
[[forbidden]]
pattern = '^\s*Note:'
reason  = "no translator's notes"

[[advisory]]
id          = "register"
description = "keep the source's formality level"

# context.toml — blocks render in the order written, subject to the budget
[[context.block]]
kind = "literal"
text = "Translate the line below."

[[context.block]]
kind = "retrieved"
k    = 5
```

---

## models.toml

Logical models and the endpoints that serve them. The pool owns one connection per endpoint and
routes each model over it; personas naming the same model share it.

### `[endpoint.<name>]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `provider` | `"llamacpp-router" \| "ollama" \| "openai-compatible"` | `"llamacpp-router"` | Server family. Ollama cannot rerank; vLLM is unsupported. |
| `base_url` | string | `http://127.0.0.1:8080/v1` | OpenAI-compatible base URL (must include a port for the serve script). |
| `resident_max` | int ≥ 1 | `4` | Distinct models kept resident before LRU eviction (the thrash guard's ceiling; mirrors `--models-max`). |
| `parallel` | int ≥ 1 | `2` | Server request slots; must be ≥ the harness concurrency. |
| `vram_budget_mb` | int ≥ 0 | `0` | Optional VRAM ceiling for the byte-level guard; `0` disables it (count-only). |
| `timeout_seconds` | float > 0 | `300.0` | Per-request timeout. |
| `max_retries` | int ≥ 1 | `4` | Transient-failure retries. |
| `retry_backoff_seconds` | float ≥ 0 | `2.0` | Base for exponential backoff. |
| `enable_reasoning` | bool | `false` | Leave a reasoning model's chain-of-thought on (usually off for bounded tasks). |
| `server_args` | array of strings | `[]` | Verbatim launch flags for `serve_models.sh --config` (GPU offload, KV-cache type, ...). Launch-time only — not per-request. |

### `[model.<name>]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `endpoint` | string | required | Which `[endpoint.<name>]` serves it. |
| `model_id` | string | required | The id the server exposes (pin by repo + quantisation + revision). |
| `backend` | `"auto" \| "generic" \| "bielik" \| "eurollm" \| "gemma"` (or a dotted path) | `"auto"` | Model-family profile that shapes a structured request. `generic` (aliases `openai`/`json_schema`) constrains decoding with a JSON-Schema grammar (Qwen and most llama.cpp/vLLM builds); `bielik`/`eurollm`/`gemma` describe the shape in the prompt with a `json_object` response for builds that cannot compile a grammar. `auto` picks by the model id. |
| `kind` | `"chat" \| "embedding" \| "rerank"` | `"chat"` | Role; personas use chat models, retrieval uses the others. |
| `context_window` | int ≥ 0 | `0` | Tokens, for the proactive budget warning; `0` disables it. |
| `approx_vram_mb` | int ≥ 0 | `0` | Resident weight size for the VRAM guard; `0` means undeclared (guard counts models only). |

---

## personas.toml

The producer, the reviewer panel (file order = consultation order), and the budgets they run under.
Exactly one `kind = "producer"`; at least one reviewer (or set `revision.max_revisions = 0`).

### `[revision]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `max_revisions` | int ≥ 0 | `2` | Panel-driven revise rounds before the output is kept as-is. |
| `max_repairs` | int ≥ 0 | `2` | Retries for a malformed/truncated reply. |
| `repair_truncated_json` | bool | `true` | Whether a truncated JSON envelope may be recovered as an unverified candidate. |

### `[limits]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `produce_tokens_per_source_char` | int ≥ 1 | `8` | Producer budget scales with input length. |
| `produce_tokens_floor` | int ≥ 1 | `256` | Lower bound on the producer budget. |
| `produce_tokens_ceiling` | int ≥ 1 | `1024` | Upper bound (must be ≥ the floor). |
| `review_tokens` | int ≥ 1 | `1024` | Default per-reviewer output budget. |

### `[leniency]` (top-level default; also per-persona)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `window` | int ≥ 1 | `20` | Rolling window of a reviewer's replies. |
| `max_bad` | int ≥ 0 | `2` | Unusable replies tolerated in the window before a **console** warning (every one is still logged). |

### `[[persona]]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `id` | string | required | Unique persona id. |
| `kind` | `"producer" \| "reviewer"` | required | Its role in the loop. |
| `model` | string | required | Logical model from `models.toml`. |
| `instructions` | string | `""` | System prompt; `{placeholder}` tokens are filled from the recipe's substitutions. Required unless `from_rules`. |
| `from_rules` | bool | `false` | Reviewer only: judge the rule set's `[[advisory]]` criteria instead of own instructions (mutually exclusive with `instructions`). |
| `max_tokens` | int ≥ 1 | (uses `review_tokens`) | Override this reviewer's output budget. |
| `leniency` | inline table | (top-level) | Per-persona `{window, max_bad}`. |
| `sampling` | sub-table | (role default) | Per-request decode settings — see below. |

### `[persona.sampling]`

All optional. Default temperature is role-based (producer `0.3`, reviewer `0.0`); every other knob
is unset unless named, so only chosen settings reach the server.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `temperature` | float ≥ 0 | `0.3` producer / `0.0` reviewer | Sampling temperature. |
| `top_p` | float in `[0,1]` | unset | Nucleus sampling. |
| `top_k` | int ≥ 0 | unset | Top-k truncation (`0` disables). |
| `min_p` | float in `[0,1]` | unset | Min-p truncation. |
| `seed` | int | unset | Decode seed (reproducibility). |
| `presence_penalty` | float in `[-2,2]` | unset | Presence penalty. |
| `frequency_penalty` | float in `[-2,2]` | unset | Frequency penalty. |
| `repeat_penalty` | float > 0 | unset | Repetition penalty (`1.0` = none). |
| `stop` | array of strings | `[]` | Stop sequences. |

---

## rules.toml

The three-tier policy: numeric (`[limits]`, code-checked), pattern (`[[forbidden]]`, code-checked),
and prose (`[[advisory]]`, judged by a `from_rules` reviewer).

### `[limits]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `max_line_columns` | int | `110` | Max display columns per line (width-aware). |
| `max_columns_tolerance` | float | `0.12` | Fractional slack before the column limit blocks. |
| `require_nonempty` | bool | `true` | Refuse an empty output. |
| `keep_flagged_rules` | array of strings | `[]` | Rule ids that mark "imperfect but usable" — kept (not rejected) when the repair budget runs out. |

### `[style]`, `[lexicon]`, `[[forbidden]]`, `[[advisory]]`

| Location | Key | Type | Meaning |
|---|---|---|---|
| `[style]` | `directives` | array of strings | Free-form style directives shown to reviewers. |
| `[lexicon]` | `severity` | `"error" \| "warning"` | Severity for a lexicon (terminology) violation. Default `warning`. |
| `[[forbidden]]` | `pattern`, `reason` | string, string | A forbidden regex and why — a blocking pattern violation. |
| `[[advisory]]` | `id`, `description` | string, string | A prose criterion a `from_rules` reviewer judges. |

---

## context.toml

An ordered list of context blocks and an optional budget. Each `[[context.block]]` names a `kind`
and passes the rest as that block's options; unknown options are refused per block.

### `[[context.block]]` — common built-in kinds

| kind | Options | Renders |
|---|---|---|
| `literal` | `text` (required), `heading` | A fixed instruction. |
| `lexicon` | `heading`, `limit` | Terminology entries matching the input (`limit` caps how many, default 12). |
| `neighbours` | `before`, `after`, `heading` | Surrounding source lines as one passage. |
| `retrieved` | `k` (≥1), `min_score` (`[0,1]`), `heading` | Reference examples from the wired retriever (the "memory"). |
| `established` | `before`, `after`, `heading` | Already-produced outputs for neighbouring inputs (needs memory). |
| `previous_attempt` | `heading` | On revision, the attempt being fixed and its issues. |
| `sql_rows` | `query` (required), `param_keys`, `limit`, `heading` | Rows from the read-only external DB, `param_keys` bound positionally from `record.meta`. |
| `schema` | `heading` | The introspected external-DB schema (needs an introspector). |
| `readings` | `keys`, `heading` | Selected `record.meta` fields (sensor readings, fault codes) — reads the record itself. |

### `[context.budget]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `max_chars` | int ≥ 0 | `0` | Character budget; `0` disables it. Over budget, whole blocks are dropped by priority (never truncated mid-section). |
| `trim_order` | array of strings | `[]` | Block kinds most-trimmable-first. A kind not listed is never dropped. |

---

## storage.toml

Bindings for the framework's stores and the external data source. Each `[<port>]` table names a
`driver` and passes the rest as its options. A relative `path` is resolved against the config
directory, so a config is portable.

| Table | Driver (built-in) | Key options |
|---|---|---|
| `[sql]` | `sqlite` \| `duckdb` | `path`, `read_only`, `schema_sql` (the external data source is `read_only = true`; `schema_sql` initialises the framework's own writable store and is refused on a read-only binding). Swapping `sqlite`↔`duckdb` is a one-line config edit — both are real embedded SQL engines and pass the same conformance suite. |
| `[vector]` | `lancedb` \| `qdrant` | `path`, `dim` (plus `table`, `metric` for `lancedb`; `collection` for `qdrant`). Two real embedded vector DBs behind one port — swapping `lancedb`↔`qdrant` is a one-line edit; both pass the same conformance suite. `qdrant` also takes `url` to point at a Qdrant server. |
| `[lexical]` | `fts5` | `path`, `tokenizer` (SQLite FTS5 BM25-only; `tokenizer` defaults to `unicode61`). A search index that returns ids only — a hit's display text is resolved through `[documents]`. |
| `[documents]` | `sqlite` | `path` (default `:memory:`). The relational chunk-row store **every retrieval path resolves a hit through** — a search index (FTS5 BM25, or the vector ANN) returns ids, and this store turns an id back into its display text + metadata. Give it a `path` to keep the corpus on disk (required for a large corpus and for build-once reuse); with no `path` it is in-memory SQLite, the default for small corpora and tests. |
| `[introspector]` | `sqlite` \| `duckdb` | `path` (reads a schema without importing a store driver). |

Any table also accepts a **dotted path** (`driver = "mypkg:MyStore"`) or an entry-point name for a
third-party driver — resolved through the registry, no framework change.

The two database roles are kept apart: the framework's own writable store, and the external
task data source (`read_only` — a write is refused at the port).

**On-disk streaming ingest, built once.** Ingest streams the reference corpus line-by-line in
batches, so a multi-GB corpus never materialises in RAM — the inverted index, the vectors, and the
chunk rows all live in their stores. When `[lexical]` and `[documents]` are given a `path`, the
on-disk stores persist across runs and are **built once**: assembly re-ingests only when the
document store is empty (`count == 0`), so an already-populated on-disk corpus is read once and
reused. An in-memory store (no `path`) is empty at every startup and so is rebuilt each run.

---

## recipe.toml

What the task produces, its extra validators, and an optional reference corpus.

### `[task]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `output_schema` | string | `"json_field"` | Registered output schema (`json_field`, `form`, or a dotted path). |
| `output_schema_options` | table | `{}` | Passed to the schema (e.g. `field`, or a `fields` array for `form`). |
| `input_label` | string | `"Input to act on:"` | Label shown before the input in the prompt. |
| `stand_in` | string | `"they"` | Readable replacement for a masked placeholder in a neighbour line. |
| `use_memory` | bool | `false` | Keep produced outputs in memory for the `established` block. |

### `[[validator]]`

| Key | Type | Meaning |
|---|---|---|
| `kind` | string (required) | Registered validator name or dotted path. |
| *(others)* | — | Passed as that validator's options (e.g. a `[[validator.field]]` array). |

### `[reference]` (optional retrieval memory)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `file` | string | `""` | JSONL corpus (relative to the config dir) to retrieve from. |
| `retriever` | string | `"lexical"` | `"lexical"` builds over the corpus; a **dotted path / entry-point name** selects a corpus-free custom retriever resolved through the `RETRIEVERS` registry. |
| `index_field` | string | `"source"` | The JSON field matched on. |
| `display_field` | string | `""` | The field a hit shows (falls back to a sensible default). |
| `options` | table | `{}` | Options passed to a custom (dotted-path) retriever's `from_config`. |

**Replacing the retriever without editing our code** — two seams, matching how the components are
stateful:
- A **corpus-free** retriever (its own search backend) is named by dotted path in `retriever` and
  built from `options` through the registry.
- A **corpus-stateful** retriever (one that must hold our reference corpus) is injected:
  `assemble(config_dir, retriever=my_retriever)`, exactly as `client_factory` and
  `extra_validators` are injected.

---

## retrieval.toml

Optional. When present, it assembles the reference retriever — a lexical, dense, or hybrid stack —
from config instead of the default single lexical retriever, exposing the per-arm **floors**, the
candidate pool, the MMR trade-off, and the rerank model. The dense/hybrid kinds need a `[vector]`
store in `storage.toml` (built with the embedding model's output dimension) and an embedding model
in `models.toml`; rerank needs a rerank model. The final relevance floor for a run stays the
retrieved *block*'s `min_score` (`context.toml`); the per-arm floors below are the fusion-input
floors.

### `[retrieval]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `kind` | `"lexical" \| "dense" \| "hybrid"` | `"lexical"` | Which stack to build. |
| `candidate_pool` | int ≥ 1 | `40` | Candidates fetched per arm before fusion (hybrid). |
| `mmr_lambda` | float in `[0,1]` | `0.7` | MMR relevance-vs-diversity trade-off (hybrid). |

| Sub-table | Key | Type | Default | Meaning |
|---|---|---|---|---|
| `[retrieval.lexical]` | `min_score` | float in `[0,1]` | `0.30` | Lexical-arm fusion floor. |
| `[retrieval.dense]` | `model` | string | required for dense/hybrid | Embedding model (`models.toml`, `kind="embedding"`). |
| `[retrieval.dense]` | `min_score` | float in `[0,1]` | `0.55` | Dense-arm fusion floor. |
| `[retrieval.rerank]` | `enabled` | bool | `false` | Add a cross-encoder rerank stage (hybrid). |
| `[retrieval.rerank]` | `model` | string | required if enabled | Rerank model (`models.toml`, `kind="rerank"`). |
