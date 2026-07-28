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
[`recipe.toml`](#recipetoml). A recipe's directory holds the subset it needs.

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
| `backend` | `"auto" \| "json_schema" \| "json_object"` | `"auto"` | How a structured request is shaped for this model. |
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
| `lexicon` | `heading` | Terminology entries matching the input. |
| `neighbours` | `before`, `after`, `heading` | Surrounding source lines as one passage. |
| `retrieved` | `k` (≥1), `min_score` (`[0,1]`), `heading` | Reference examples from the wired retriever (the "memory"). |
| `established` | `heading` | Already-produced outputs for neighbouring inputs (needs memory). |
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
| `[sql]` | `sqlite` | `path`, `read_only` (the external data source is `read_only = true`). |
| `[vector]` | `lancedb` | `path`, `dim`, ... (the default real vector DB). |
| `[lexical]` | `fts5` | `path` (SQLite FTS5 BM25). |
| `[introspector]` | `sqlite` | `path` (reads a schema without importing a store driver). |

The two database roles are kept apart: the framework's own writable store, and the external
task data source (`read_only` — a write is refused at the port).

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
| `retriever` | `"lexical"` | `"lexical"` | How to retrieve (lexical is buildable from config alone). |
| `index_field` | string | `"source"` | The JSON field matched on. |
| `display_field` | string | `""` | The field a hit shows (falls back to a sensible default). |
