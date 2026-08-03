# Tutorial: building pipelines with rag-llm-template

This guide shows how to *actually build solutions* on the framework: the mental model, the whole
pipeline end to end, a new recipe from scratch (config only, then with a plugin, then with a
memory, then with a database), and how to extend every seam — always **without editing framework
code**.

Prerequisites: `tools/setup_python_env.sh` (builds `.venv/`), then `tools/check_environment.sh` to
confirm the machine is ready. Every example is runnable with no server using the pattern in
[§11](#11-testing-your-pipeline); to run against real models, serve them
([§9](#9-running-against-real-models)).

**One thing to get straight before anything else.** A JSONL file is *not* this framework's memory
or retrieval engine. It is one possible **one-time import source**. What answers queries at run
time is a real on-disk database — the `[pairings]` store. [§6](#6-the-two-stage-shape-import-once-query-forever)
is about exactly that distinction, and it is the single most common misreading of the design.

**Nothing below is the only way to do it.** Every seam has alternatives, and this tutorial names
them as it goes with a one-line "when you'd pick this one". The exhaustive key-by-key reference is
[`docs/config.md`](config.md); the shape of the whole system is
[`docs/architecture.md`](architecture.md).

- [1. The mental model](#1-the-mental-model)
- [2. The pipeline, end to end](#2-the-pipeline-end-to-end)
- [3. Anatomy of a recipe](#3-anatomy-of-a-recipe)
- [4. A pipeline from scratch — config only](#4-a-pipeline-from-scratch--config-only)
- [5. Add a mechanical validator (a plugin)](#5-add-a-mechanical-validator-a-plugin)
- [6. The two-stage shape: import once, query forever](#6-the-two-stage-shape-import-once-query-forever)
- [7. Retrieval modes: lexical, dense, hybrid](#7-retrieval-modes-lexical-dense-hybrid)
- [8. Every storage option, and when to pick it](#8-every-storage-option-and-when-to-pick-it)
- [9. Running against real models](#9-running-against-real-models)
- [10. Evaluation](#10-evaluation)
- [11. Testing your pipeline](#11-testing-your-pipeline)
- [12. The extension points, one by one](#12-the-extension-points-one-by-one)
- [13. The tool scripts](#13-the-tool-scripts)
- [14. Swapping components by config](#14-swapping-components-by-config)

---

## 1. The mental model

Every task, however different, runs the same loop:

```
produce  ──▶  mechanical check (code)  ──▶  review panel (LLMs)  ──▶  revise
```

- **produce** — one *producer* persona generates a structured output for one input.
- **mechanical check** — deterministic, code-decided validators run *before any GPU time*: empty
  output, forbidden patterns, a bad SQL statement, a field of the wrong type. Cheap and certain.
- **review panel** — an ordered list of *reviewer* personas judge the output; the loop stops at the
  first objection, so the most decisive reviewer runs on everything.
- **revise** — an objection sends the output back to the producer with the reasons, bounded by
  `max_revisions`; a malformed reply is repaired up to `max_repairs`.

The unit of work is a **`Record`**: `{record_id, source, meta}` in, a produced `output` and a
`status` out (`VERIFIED` / `PRODUCED` / `REJECTED` / `SKIPPED`). A run's durable state lives in a
`RunStore` (a database, not a file): `ragkit import` loads a JSON-Lines *catalogue* into it once,
`ragkit run` reads and writes it directly (one ACID commit per result — a crash mid-run loses
nothing already committed), and `ragkit export` writes the results back out as JSON Lines for an
`eval.py` to score. Everything the loop needs — which models, the panel and its prompts, the
rules, what context to build, which stores to read — is **config**, resolved through a registry so
your own components plug in by name.

Two statuses are worth understanding now, because they are how the loop stays honest:

- `PRODUCED` means *mechanically sound, but the panel still objected when the budget ran out*, or a
  reviewer could not be evaluated. It is injectable, and it is findable — the records a human might
  want to revisit.
- A reviewer that returns garbage never becomes an acceptance. A degenerate reply degrades the
  record to `PRODUCED`; it can never produce a false `VERIFIED`.

## 2. The pipeline, end to end

Building a recipe means filling in some subset of this. Most recipes use only part of it, and each
stage below has its own section later.

```
  ┌─ INGEST (once, offline) ────────────────────────────────────────────────┐
  │  fetch.sh ─▶ extract ─▶ chunk ─▶ dedup ─▶ normalise ─▶ import_reference │
  │                                                              │          │
  └──────────────────────────────────────────────────────────────┼──────────┘
                                                                 ▼
  ┌─ MEMORY (on disk, queried every run) ───────────────────────────────────┐
  │  [pairings]  rows + BM25 index, one database   (+ [vector] ANN, opt.)   │
  │  [lexicon]   established terminology            [sql] external, read-only│
  └─────────────────────────────┬───────────────────────────────────────────┘
                                ▼
  ┌─ PER RECORD ────────────────────────────────────────────────────────────┐
  │  context assembly ─▶ produce ─▶ mechanical check ─▶ review ─▶ revise    │
  │  (blocks, budget)                        │                              │
  └──────────────────────────────────────────┼──────────────────────────────┘
                                             ▼
  ┌─ RUN STATE + OUT ───────────────────────────────────────────────────────┐
  │  [run] store  ─▶  ragkit export ─▶ journal.jsonl ─▶ eval.py             │
  │                   ragkit writeback ─▶ back into [pairings]              │
  └─────────────────────────────────────────────────────────────────────────┘
```

**Ingest** (`src/ragkit/ingest/`) turns a raw corpus into reference memory. Each stage is a seam:

| Stage | Port / function | Built-ins | Notes |
|---|---|---|---|
| fetch | `fetch.sh` | per recipe | A hardened downloader. Produces whatever it likes; JSONL is merely the convention every shipped recipe settled on. |
| extract | `Extractor` | `text`, `jsonl`, `html`, `markdown` | Raw bytes → `Document`s. |
| chunk | `Chunker` | `structure` (default), `sentence`, `fixed` | `Document` → `Chunk`s with provenance (`document_id`, `ordinal`, true `char_start`/`char_end`). |
| dedup | `dedup_documents` / `dedup_chunks` | — | Exact-match only, on the text **verbatim**. Near-duplicate detection is a plugin's job. |
| normalise | `normalise` | — | NFC, whitespace collapse, optional de-hyphenation. |
| import | `import_reference` | — | Streams into `[pairings]`, resumable, batched. |

> **Ordering constraint.** `normalise(collapse_whitespace=True)` folds the `\n\n` that the default
> `structure` chunker splits on, so normalising *first* silently degrades it to whole-document
> sentence packing. Chunk before you collapse whitespace, or pass `collapse_whitespace=False`.

## 3. Anatomy of a recipe

A recipe is a directory. Nothing here is special-cased by the framework; it is discovered by the
files present.

```
recipes/<task>/
  config/
    models.toml       # logical models -> endpoints (which server, which model id, backend)
    personas.toml     # the producer + reviewer panel (as prompts), budgets, per-persona sampling
    rules.toml        # policy: numeric limits, forbidden regexes, prose advisories, style
    context.toml      # the ordered context blocks that build the prompt, + a char budget
    recipe.toml       # the output schema, the task validators, the optional reference corpus
    storage.toml      # (optional) [pairings] [vector] [lexicon] [run] [sql] [introspector]
    retrieval.toml    # (optional) config-driven lexical/dense/hybrid retrieval
  plugins/            # (optional) your task-specific Validator / OutputSchema / ContextBlock / ...
  fetch.sh            # a hardened downloader for the real dataset (git-ignored data/)
  eval.py             # scores the exported journal against held-out gold
  tests/              # the recipe's own suite, against tiny committed fixtures
```

Only `models.toml`, `personas.toml`, `rules.toml`, `context.toml` and `recipe.toml` are required.
The full per-key reference is [`docs/config.md`](config.md). Below we build one.

## 4. A pipeline from scratch — config only

**Goal:** triage a customer-support message into a `category` and an `urgency`. No Python yet.
We extend this same *triage* recipe through the rest of the tutorial.

Create `recipes/triage/config/`. First, the models — two logical models on one local endpoint (the
producer a bit larger, the reviewers small; personas sharing a model share it on the server):

```toml
# recipes/triage/config/models.toml
[endpoint.local]
provider = "llamacpp-router"
base_url = "http://127.0.0.1:8080/v1"
resident_max = 4

[model.author]
endpoint = "local"
model_id = "Qwen3-4B-Q4_K_M"
backend  = "auto"           # auto | generic | bielik | eurollm | gemma
context_window = 8192

[model.reviewer]
endpoint = "local"
model_id = "Qwen3-1.7B-Q8_0"
```

Those two ids are what `tools/fetch_models.sh` downloads by default. Any OpenAI-compatible endpoint
works; see [§9](#9-running-against-real-models).

The panel — a producer and two reviewers. The producer runs a little warm; reviewers are
deterministic by default. The last reviewer builds its instructions *from the rules*, so policy has
one home:

```toml
# recipes/triage/config/personas.toml
[revision]
max_revisions = 2
max_repairs = 2

[[persona]]
id = "author"
kind = "producer"
model = "author"
instructions = """
You triage a customer-support message. Choose the single best category and an urgency.
Base both only on the message; do not invent facts. Return only the requested fields.
"""
[persona.sampling]
temperature = 0.2          # per-request decode settings live on the persona

[[persona]]
id = "consistent"
kind = "reviewer"
model = "reviewer"
instructions = """
You check the category and urgency actually fit the message. Object only when a competent
support lead would clearly have chosen differently; a defensible alternative is not an error.
"""

[[persona]]
id = "compliance"
kind = "reviewer"
model = "reviewer"
from_rules = true          # judges the [[advisory]] criteria in rules.toml, in one pass
```

The rules — mechanical limits (code-checked) and prose advisories (reviewer-judged):

```toml
# recipes/triage/config/rules.toml
[limits]
require_nonempty = true

[[advisory]]
id = "grounded"
description = "The category and urgency follow from the message text, not from assumptions about the customer."

[[advisory]]
id = "calibrated_urgency"
description = "'high' is reserved for outages, data loss, or a blocked customer; routine questions are 'low'."
```

The context — what goes in the prompt, in order. Two rules govern assembly: a block that has
nothing to say contributes **nothing at all** (no dangling headings, ever), and when `max_chars` is
set, whole blocks are dropped by `trim_order` priority rather than truncated mid-section. Here just
a grounding instruction:

```toml
# recipes/triage/config/context.toml
[[context.block]]
kind = "literal"
text = "Read the support message below and triage it. Choose exactly one category and one urgency."
```

The built-in block kinds are `literal`, `retrieved`, `sql_rows`, `schema`, `lexicon`,
`established`, `neighbours`, `previous_attempt` and `readings` — each covered where it becomes
relevant below, and all listed in [`config.md`](config.md#contexttoml). Your own block plugs in by
dotted path ([§12](#12-the-extension-points-one-by-one)).

The recipe — the **output shape** and the input label. Two output schemas ship: `json_field` (one
string field — pick it when the task produces one piece of text, like a translation) and `form`
(several typed fields — pick it when the task fills a structure). We want the latter:

```toml
# recipes/triage/config/recipe.toml
[task]
output_schema = "form"
input_label = "Support message:"

[[task.output_schema_options.fields]]
name = "category"
description = "one of: billing, bug, feature_request, how_to, other"

[[task.output_schema_options.fields]]
name = "urgency"
description = "one of: low, medium, high"
```

That is a complete pipeline. The catalogue is JSON-Lines, one `Record` per line:

```
# work/tickets.jsonl
{"record_id": "t1", "source": "I was charged twice this month, please refund the duplicate."}
{"record_id": "t2", "source": "How do I export my data to CSV?"}
```

Run it (once models are served — [§9](#9-running-against-real-models)):

```bash
PYTHONPATH=src python -m ragkit.cli import --catalog work/tickets.jsonl --run-db work/triage.db
PYTHONPATH=src python -m ragkit.cli run --config recipes/triage/config --run-db work/triage.db
PYTHONPATH=src python -m ragkit.cli export --run-db work/triage.db -j work/out.jsonl
```

Or the whole sequence, server included, with `tools/run_recipe.sh --recipe triage`
([§13](#13-the-tool-scripts)).

Each record's `output` is the filled form as canonical JSON, e.g. `{"category": "billing",
"urgency": "high"}`, with `status = VERIFIED` once the panel accepts it.

## 5. Add a mechanical validator (a plugin)

The config above *asks* for a category from a fixed set, but nothing yet *enforces* it in code. A
model could return `"payments"`. Add a deterministic check — a `Validator` selected by dotted path,
no framework change. A validator returns zero or more `Violation`s; an `ERROR` blocks acceptance.

```python
# recipes/triage/plugins/validators.py
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ragkit.core.records import Record
from ragkit.core.rules import Severity, Violation


class EnumFieldsValidator:
    """Refuse a form whose field values are not in their allowed set — defence-in-depth over the
    output schema's shape check. Deterministic, and abstains when it cannot evaluate."""

    CONFIG_KEYS = frozenset({"field"})   # the keys this component accepts in recipe.toml

    def __init__(self, allowed: Mapping[str, tuple[str, ...]]) -> None:
        self._allowed = allowed

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> "EnumFieldsValidator":
        allowed = {entry["name"]: tuple(entry["values"]) for entry in options.get("field", [])}
        if not allowed:
            raise ValueError("EnumFieldsValidator needs at least one [[validator.field]]")
        return cls(allowed)

    def validate(self, record: Record, output: str,   # noqa: ARG002 -- record unused; port shape
                 context: Mapping[str, Any]) -> list[Violation]:
        try:
            form = json.loads(output)
        except json.JSONDecodeError:
            return []   # not our concern; the schema/other validators own JSON validity
        violations: list[Violation] = []
        for field, values in self._allowed.items():
            value = form.get(field)
            if value is not None and str(value).lower() not in {v.lower() for v in values}:
                violations.append(Violation(
                    "enum_field", Severity.ERROR,
                    f"field {field!r} is {value!r}, not one of {list(values)}"))
        return violations
```

Wire it into `recipe.toml` by its dotted path, with its own options (unknown keys are refused, just
like a built-in's):

```toml
# recipes/triage/config/recipe.toml  (append)
[[validator]]
kind = "recipes.triage.plugins.validators:EnumFieldsValidator"

[[validator.field]]
name = "category"
values = ["billing", "bug", "feature_request", "how_to", "other"]

[[validator.field]]
name = "urgency"
values = ["low", "medium", "high"]
```

Now a category outside the set is a blocking violation: the producer is asked to fix it, and after
the revision budget the record is `REJECTED` rather than shipped wrong.

> **Why a plugin and not framework code?** The category set is *your* policy, not the framework's.
> The `Validator` port (`validate(record, output, context) -> list[Violation]`) is the seam; the
> registry resolves your class by dotted path. The framework never learns about "triage".

**Validator discipline**, which the framework relies on: *abstain unless there is positive
evidence*. A check that cannot be evaluated returns no violation — it is skipped, never reported as
passed. Note the `json.JSONDecodeError` branch above returning `[]` rather than an error: JSON
validity is someone else's contract.

A defect in your validator can no longer take down a run: an unexpected exception out of
`validate` is contained to that one record (journalled `REJECTED`, with the failing frame in the
diagnostic and a full traceback in the log) rather than aborting the batch.

## 6. The two-stage shape: import once, query forever

This is the part most worth reading slowly.

Reference memory has **two stages, and they are not the same thing**:

```
   IMPORT SOURCE                          THE STORE
   (read exactly once)                    (queried on every run, forever)

   data/examples.jsonl    ──stream──▶     data/examples.pairings.db
   whatever fetch.sh                      a real SQLite/DuckDB database:
   produced                               rows + a BM25 index, one file
                                                  │
                                          every later run queries THIS
```

- The **import source** is any file your fetch step produces. JSONL is the convention every shipped
  recipe happens to use because that is what their `fetch.sh` scripts emit — nothing more. The
  framework reads it *once*.
- The **store** is a real on-disk database. It is what `retrieve()` queries on every record of
  every run. Once imported, the JSONL is never read again — you could delete it.

`import_reference` is **resumable and idempotent**: the resume floor is the store's own
`count()`, so a second run over an already-populated store fast-forwards past every imported line
and adds nothing. That is why a recipe can point at a multi-GB corpus and still start in seconds
after the first time.

The evidence that this is real rather than decorative, from the two large recipes in this repo:

| Recipe | Import source | `[pairings]` store | Imported |
|---|---|---|---|
| `legal_procurement` | `passages.jsonl` (2.8 GB) | `passages.pairings.db` — **7,097,288 rows** | once |
| `med_evidence` | `abstracts.jsonl` (510 MB) | `abstracts.pairings.db` — **597,055 rows** | once |

Neither JSONL has been read since its import. Every retrieval those recipes perform — and their
full-scale evaluations run 956 and 1,000 questions respectively — is served by the database.

### Wiring it, on the triage recipe

Give the classifier **worked examples** retrieved per input — a labelled corpus of past tickets.

```
# recipes/triage/data/examples.jsonl   (git-ignored; a fetch.sh would build it)
{"source": "Refund the duplicate charge on my invoice", "target": "billing / high"}
{"source": "Where do I find the CSV export button", "target": "how_to / low"}
```

```toml
# recipes/triage/config/recipe.toml  (append)
[reference]
file = "../data/examples.jsonl"     # the IMPORT SOURCE, read once
retriever = "lexical"               # BM25 over the store, built from config alone
index_field = "source"              # which JSON field matching happens on
target_field = "target"             # a hit displays as "source -> target"
```

```toml
# recipes/triage/config/storage.toml
[pairings]
driver = "sqlite"                   # THE STORE, queried every run
path = "../data/examples.pairings.db"
```

```toml
# recipes/triage/config/context.toml  (insert before or after the literal block)
[[context.block]]
kind = "retrieved"
k = 3
min_score = 0.3
heading = "Similar past tickets and how they were triaged (examples, not the required answer):"
```

A `[reference].file` **requires** a `[pairings]` store — there is no in-memory fallback, precisely
so that "where does this live?" always has an answer. The `path` is required too: give a file path
to keep the corpus on disk (built once, reused every run), or write `path = ":memory:"` *explicitly*
for an ephemeral store rebuilt from the source on every startup (right for tests and tiny corpora,
wrong for anything you care about restarting quickly). A *forgotten* `path` is refused rather than
silently becoming an ephemeral store that quietly loses the corpus between runs.

A hit displays as `"source -> target"`. There is no "show this arbitrary field alone" option — the
display convention is always a pairing's `source`/`target`, so shape the fields you want shown as
one or the other.

### Feeding the memory from your own runs

`ragkit writeback` is a deliberate, separate post-run step: it reads a finished run's `VERIFIED`
results and folds each `(source, context, target)` into the `PairingStore` as a new pairing, so a
later run retrieves what an earlier one produced. It is never automatic.

```bash
PYTHONPATH=src python -m ragkit.cli writeback --run-db work/triage.db \
    --pairings-db recipes/triage/data/examples.pairings.db
```

## 7. Retrieval modes: lexical, dense, hybrid

Retrieval is configured by an optional `retrieval.toml`. With no such file you get the single
lexical retriever [§6](#6-the-two-stage-shape-import-once-query-forever) set up. With one, you pick
the whole stack:

| `kind` | What runs | When you'd pick it |
|---|---|---|
| `lexical` | BM25 over `[pairings]`'s search index | Names, ids, codes, recurring terminology. No embedding model, no GPU, no extra store. Start here. |
| `dense` | Embed the query, ANN search a `[vector]` store | Paraphrase and cross-script matches BM25 is blind to. Costs a one-time embedding pass over the corpus. |
| `hybrid` | Both arms → RRF fusion → optional cross-encoder rerank → MMR | Best quality, most moving parts. Worth measuring rather than assuming — see the caution below. |

Switching between them is a config edit. **Lexical** (the default; the file is optional):

```toml
# recipes/triage/config/retrieval.toml
[retrieval]
kind = "lexical"
```

**Dense** — needs an embedding model in `models.toml` and a `[vector]` store built with that
model's output dimension:

```toml
# recipes/triage/config/models.toml  (append)
[endpoint.embed]
provider = "llamacpp-router"
base_url = "http://127.0.0.1:8081/v1"
resident_max = 1
server_args = ["--models-preset", "tools/embed_presets.ini"]

[model.embedder]
endpoint = "embed"
model_id = "embed"
kind = "embedding"
context_window = 32768
```

```toml
# recipes/triage/config/storage.toml  (append)
[vector]
driver = "lancedb"
path = "../data/examples.lance"
dim = 1024                          # must equal the embedding model's output dimension
```

```toml
# recipes/triage/config/retrieval.toml
[retrieval]
kind = "dense"
[retrieval.dense]
model = "embedder"                  # a models.toml model of kind = "embedding"
```

**Hybrid** — both arms, fused, with an optional reranker:

```toml
# recipes/triage/config/retrieval.toml
[retrieval]
kind = "hybrid"
candidate_pool = 40                 # candidates fetched per arm before fusion
mmr_lambda = 0.7                    # relevance-vs-diversity trade-off

[retrieval.lexical]
min_score = 0.30                    # fusion-input floor for the BM25 arm

[retrieval.dense]
model = "embedder"
min_score = 0.55                    # fusion-input floor for the dense arm

[retrieval.rerank]
enabled = true
model = "reranker"                  # a models.toml model of kind = "rerank"
score_scale = "logit"               # "logit" for llama.cpp; "unit" for Jina/Cohere
```

> **A key outside its kind's column is refused, not ignored.** `candidate_pool`, `mmr_lambda`, the
> per-arm floors and the whole `[retrieval.rerank]` table only mean something to the hybrid stack,
> so setting `[retrieval.rerank].enabled = true` under `kind = "dense"` is a `ConfigError` rather
> than a dense stack that quietly never reranks. To floor a *single-arm* stack, use the retrieved
> block's `min_score` in `context.toml` — that is the run's final relevance floor either way.

**Enabling dense on an existing corpus.** The `[pairings]` store already holds the rows; the vector
store needs filling once, and then indexing:

```bash
tools/serve_models.sh --config recipes/triage/config/models.toml --endpoint embed &
tools/embed_reference.sh --config recipes/triage/config \
    --embedding-url http://127.0.0.1:8081/v1                 # embed + upsert, resumable
tools/build_vector_index.sh --config recipes/triage/config   # build the ANN index
```

Skipping `build_vector_index.sh` is not a small omission: without an index, LanceDB's `search()`
falls back to scanning **every** row, which turned a 956-question evaluation on the 7.1M-row
`legal_procurement` corpus into a multi-hour, CPU-pegged run at 0% GPU. With the index, `nprobes`
controls how much of it a query actually probes — see [§8](#8-every-storage-option-and-when-to-pick-it).

> **Measured caution, worth repeating.** Equal-weight RRF fusion *blends* its arms rather than
> taking the better one, and on two real corpora scored **below** the better single arm unless a
> reranker was also attached. On `legal_procurement`'s full 956-question held-out set, dense alone
> reached Recall@20 0.342 while hybrid + rerank reached 0.511 — hybrid was worth it there, with the
> reranker. Measure on your own corpus before adopting it as a default; each recipe's README carries
> its own numbers.

## 8. Every storage option, and when to pick it

Every store is bound in `storage.toml`, resolved through its own registry, and swappable by editing
one `driver` line. A `path` puts a store on disk; the persistence-critical stores (`[pairings]`,
`[run]`, `[lexicon]`) **require** one — write `path = ":memory:"` explicitly for an ephemeral store,
since a *forgotten* path is refused rather than silently losing your data between runs. All the keys
are in [`config.md`](config.md#storagetoml); the trade-offs are here.

### `[pairings]` — the reference memory

One row per `(source, target, context)` pairing **plus its search index, in one database**. This
co-location is the design point: the row and its BM25 entry are written in the same transaction, so
there is no window where one exists without the other.

| Driver | When you'd pick it |
|---|---|
| `sqlite` (default) | Almost always. WAL, so a reader is never blocked by a writer; FTS5 triggers keep the index incremental, so an `add` costs O(batch). Proven here to 7.1M rows. |
| `duckdb` | You already have a DuckDB-shaped workflow, or you want its columnar analytics over the same file. Retrieval behaviour is identical (same BM25 transform, same ranking). Its FTS extension rebuilds the whole index rather than updating it, so the rebuild is deferred to the next `search` — a bulk load pays one rebuild, not one per batch. |

Both pass the same conformance suite, including all-or-nothing batch writes and score bounds. One
honest caveat: the two engines compute different raw BM25, so absolute scores differ (a term
present in every document scores 0.0 on SQLite, ~0.19 on DuckDB) even though the transform and the
ranking are identical. Re-check a `min_score` after a driver swap.

```toml
[pairings]
driver = "sqlite"        # or "duckdb"
path = "../data/examples.pairings.db"
tokenizer = "unicode61"  # sqlite only
```

### `[vector]` — the ANN index for dense retrieval

Only needed for `kind = "dense"` or `"hybrid"`.

| Driver | When you'd pick it |
|---|---|
| `lancedb` (default) | The default embedded choice: on-disk columnar, versioned, no daemon. Pick it unless you need metadata filtering. |
| `qdrant` | You need to **filter on metadata** at query time, or you want to point at a real Qdrant server later (`url = ...`) with the same driver code. |

That filtering difference is real and worth knowing: LanceDB stores metadata as one opaque JSON
column, so a predicate on a metadata key is **refused at the boundary** with an error naming
qdrant, rather than failing mid-query. Filtering on `id` works on both.

```toml
[vector]
driver = "lancedb"       # or "qdrant"
path = "../data/examples.lance"
dim = 1024
metric = "cosine"        # lancedb
nprobes = 64             # lancedb: IVF partitions probed per query (see below)
```

`nprobes` is the recall/latency dial for a LanceDB ANN index. Left unset it scales with the table
(5% of the `~sqrt(rows)` partitions the index build creates, never below 20), which is the right
default; set it explicitly to move along the trade, and you **must** set it if you built the index
with an explicit `--num-partitions`. Without an index the setting is inert — search is exact.

### `[sql]` and `[introspector]` — an external, read-only database

The database your *task* reads: a business database for form-autofill, a schema for NL→SQL. The
framework never writes it — a write through a `read_only` binding is refused at the port, before
the database.

| Driver | When you'd pick it |
|---|---|
| `sqlite` | The file you already have. |
| `duckdb` | Analytical queries, Parquet/CSV attached, or a DuckDB-native warehouse. |

```toml
[sql]
driver = "sqlite"                 # or "duckdb"
path = "../data/business.sqlite"
read_only = true

[introspector]
driver = "sqlite"                 # reads a schema without importing a store driver
path = "../data/business.sqlite"
```

A typo'd `[introspector].path` raises rather than silently returning an empty schema — both drivers
open read-only, so a missing file is an error.

The read-only binding is one of two guardrails around an external database. The other is
task-level: the `nl_to_sql` recipe additionally parses the *generated* statement down to a single
schema-bounded `SELECT` before anything runs it, as a plugin validator — the port stops the
framework writing, and the validator stops the model asking for something it should not.

### `[run]` — the run state

The record catalogue and the append-only result history a run reads and writes *while it executes*.
One driver, `sqlite` (WAL, foreign-key enforced, one ACID commit per result). Give it a `path` for
anything you might need to resume:

```toml
[run]
driver = "sqlite"
path = "../data/triage.run.db"   # relative to the config directory, like every other path
synchronous = "FULL"             # fsync every commit; "NORMAL" is faster, slightly less durable
```

In practice the CLI's `--run-db` names this directly, so most recipes never configure it at all.

### `[lexicon]` — established terminology

A term → rendering mapping (a different shape and key from a pairing), for tasks that must use
agreed vocabulary consistently. One driver, `sqlite`; usually co-located in the same file as
`[pairings]` as its own table. Surfaced by the `lexicon` context block.

```toml
[lexicon]
driver = "sqlite"
path = "../data/examples.pairings.db"   # same file, different table
```

### Your own driver

Anywhere a `driver` is accepted, a dotted path selects your class:
`driver = "mypkg:MyVectorIndex"`. Put the third-party dependency lazily inside your driver module
only (a boundary test enforces the confinement), and run it through
`tests/store/test_conformance.py` — that suite is where a new driver earns the word
"interchangeable".

## 9. Running against real models

Everything above is served by any OpenAI-compatible endpoint. The recommended local setup is
llama.cpp in **router mode** (one process serves several models, loading each once):

```bash
tools/fetch_models.sh                       # download the pinned default GGUFs (once)
tools/serve_models.sh --config recipes/triage/config/models.toml --endpoint local --models-dir models
PYTHONPATH=src python -m ragkit.cli import --catalog work/tickets.jsonl --run-db work/triage.db
PYTHONPATH=src python -m ragkit.cli run --config recipes/triage/config --run-db work/triage.db \
    --concurrency 2
PYTHONPATH=src python -m ragkit.cli export --run-db work/triage.db -j work/out.jsonl
```

The pool checks, **before contacting any server**, that you are not asking one endpoint to hold
more distinct models than it keeps resident (the thrash guard), and that every persona names a
model that exists. Sampling is per-persona (`[persona.sampling]`); genuinely launch-time server
flags go in `[endpoint.<name>].server_args` and `serve_models.sh --config` reads them, so
`models.toml` is the single source for routing *and* serving.

A run is resumable: `ragkit run` reads `pending()` from the run store, so re-running after an
interruption picks up exactly where it stopped. `--limit N` runs a trial slice.

## 10. Evaluation

Three separable layers, all built so a **circular configuration is refused rather than reported** —
a metric must be independent of what it ranks.

### Retrieval quality

`ragkit.eval.evaluate_retrieval` scores one or more retrievers against `Qrels` (ground truth
carrying a `source` label):

- `recall_at_k` — fraction of a query's gold ids in the top *k*.
- `hit_rate_at_k` — 1/0 per query for whether *any* gold id landed in the top *k*, averaged. This is
  the "top-k accuracy" that OpenQA papers report; it is **not** the same statistic as recall.
- `mrr` — mean reciprocal rank of the first gold id.
- `map` — mean average precision.
- `ndcg_at_k` — normalised discounted cumulative gain, binary relevance.

Each metric can take its own depth, because the conventional depths differ:

```python
from ragkit.eval import Qrels, evaluate_retrieval

scores = evaluate_retrieval(
    {"lexical": lexical_retriever, "hybrid": hybrid_retriever},
    queries={"q1": "how do I export data?"},
    qrels=Qrels(relevant={"q1": frozenset({"ref-42"})}, source="human-annotators"),
    k=20, hit_rate_k=10, rank_k=10)          # Recall@20, Acc@10, MRR@10/NDCG@10
print(scores["hybrid"].recall_at_k, scores["hybrid"].hit_rate_at_k)
```

It raises `CircularEvaluationError` if the ground truth's `source` names a system under evaluation,
and `IncompleteGroundTruthError` if an evaluated query has no gold judgments (scoring it would
invent a number). Note the circularity guard is *nominal* — it compares labels, so it catches the
mistake, not an adversary.

### Classification

For a recipe that predicts a label per record, `ragkit.eval.gold` and `ragkit.eval.classify` supply
the whole frame, so an `eval.py` writes only what is task-specific:

```python
from ragkit.eval.classify import ClassificationReport, score_labels
from ragkit.eval.gold import join_journal_with_gold, load_label_gold, run_report

gold = load_label_gold(args.gold, field="decision")          # {record_id: label}
pairs = join_journal_with_gold(args.journal, gold)           # (id, produced_or_None, gold)
report = ClassificationReport(score_labels(pairs, field="decision"))
print(f"accuracy {report.accuracy:.3f} over {report.total}")
```

A record the run failed to produce pairs as `None` — a **miss**, counted in the denominator, never
an empty-string answer that would flatter the score. Add your own rates over
`ClassificationReport.rate(predicate)` rather than recounting; `med_evidence` (abstention rate) and
`predictive_maintenance` (an "actionable" rate forgiving a within-group confusion) are worked
examples.

### Blinded A/B judging

`evaluate_ab` compares two systems' outputs with an LLM judge that sees neutral "Output 1/2" in an
injected deterministic order, never the system names. A/B-ing a system against itself is refused.

```python
from ragkit.eval import AbItem, evaluate_ab

summary = evaluate_ab(judge_client, [AbItem("t1", "the input", "output A", "output B")],
                      system_a="baseline", system_b="candidate", criterion="which triage is right")
print(summary.a_win_rate)
```

## 11. Testing your pipeline

The whole framework is tested with **no server, GPU, or network**: model calls go through an
in-memory `httpx.MockTransport`, and databases run embedded. Do the same for your recipe. Inject a
scripted client with `assemble(..., client_factory=...)`:

```python
import json, httpx
from ragkit.cli.app import assemble
from ragkit.core.records import Record, Status
from ragkit.harness import run_batch
from ragkit.store.run.sqlite import SqliteRunStore

def factory(_base_url, _timeout):
    def handler(request):
        body = json.loads(request.content)
        props = body.get("response_format", {}).get("json_schema", {}).get("schema", {}).get("properties", {})
        # reviewers get a REVIEW schema (has "acceptable"); the producer gets the output schema
        content = ('{"acceptable": true, "issues": []}' if "acceptable" in props
                   else json.dumps({"category": "billing", "urgency": "high"}))
        return httpx.Response(200, json={"choices": [{"message": {"content": content},
                                                      "finish_reason": "stop"}], "usage": {}})
    return httpx.Client(transport=httpx.MockTransport(handler))

def test_triage_verifies(tmp_path):
    assembled = assemble("recipes/triage/config", client_factory=factory)
    store = SqliteRunStore(str(tmp_path / "run.db"))
    store.add_records([Record(record_id="1", source="Charged twice, refund please.")])
    run_batch(assembled.harness, store.pending(), store, install_signal_handlers=False)
    [result] = list(store.results())
    assert result.record.status is Status.VERIFIED
    assert json.loads(result.record.output)["category"] == "billing"
```

If your recipe uses an embedding endpoint, have the fake emit the `index` field the real API does
(`{"index": i, "embedding": [...]}`) — the client orders vectors by it, and a fake that omits it is
not exercising the real contract.

Run with `tools/run_tests.sh recipes/triage`. Keep `src/` at 100% statement+branch coverage (the
suite gates it); a recipe's own coverage is your responsibility to keep meaningful. The real
recipes' `tests/` are worked examples of adversarial validator tests and DB-swap parametrization.

## 12. The extension points, one by one

Every seam is a `runtime_checkable` Protocol in `src/ragkit/core/ports.py`. A component declares its
own `CONFIG_KEYS` and (usually) a `from_config(options)`; it is selected by a **dotted path**
(`"mypkg.mod:MyClass"`), a published **entry point**, or a registered **built-in name** — checked in
that order. Anything you write is chosen the same way a built-in is, and `create()` verifies both
that your class has the port's members and that each accepts the port's keyword-only parameters —
so a mismatch is reported where the driver was named, not deep inside a later call.

The complete list of ports, and where each is selected:

| Port | Selected in | Built-ins |
|---|---|---|
| `Extractor` | code / `EXTRACTORS` | `text`, `jsonl`, `html`, `markdown` |
| `Chunker` | code / `CHUNKERS` | `structure`, `sentence`, `fixed` |
| `Embedder` | `[retrieval.dense].model` | `EmbeddingClient` over any OpenAI-compatible endpoint |
| `VectorIndex` | `storage.toml [vector]` | `lancedb`, `qdrant` |
| `SearchIndex` | — (satisfied by `PairingStore`) | — |
| `PairingStore` | `storage.toml [pairings]` | `sqlite`, `duckdb` |
| `LexiconStore` | `storage.toml [lexicon]` | `sqlite` |
| `SqlStore` | `storage.toml [sql]` | `sqlite`, `duckdb` |
| `SchemaIntrospector` | `storage.toml [introspector]` | `sqlite`, `duckdb` |
| `RunStore` | `storage.toml [run]` / `--run-db` | `sqlite` |
| `Retriever` | `retrieval.toml [retrieval].kind`, or `[reference].retriever`, or injected | `lexical` / `dense` / `hybrid` via `retrieval.toml`; `[reference].retriever` takes only `"lexical"` or a dotted path to a corpus-free one |
| `Reranker` | `[retrieval.rerank].model` | `RerankClient` |
| `ContextBlock` | `context.toml [[context.block]] kind` | `literal`, `lexicon`, `neighbours`, `established`, `retrieved`, `previous_attempt`, `sql_rows`, `schema`, `readings` |
| `Validator` | `recipe.toml [[validator]] kind` | plus the built-in mechanical checks |
| `OutputSchema` | `recipe.toml output_schema` | `json_field`, `form` |
| `Backend` | `models.toml backend` | `auto`, `generic`, `bielik`, `eurollm`, `gemma` |
| `Source` / `Sink` | code | `PairingSink` (write-back) |
| `Provider` | `models.toml provider` | **unimplemented** — see the note below |

### Validator — `validate(record, output, context) -> list[Violation]`
Code-decided acceptance checks. Shown in [§5](#5-add-a-mechanical-validator-a-plugin).

### ContextBlock — `render(record, context) -> str | None`
Builds one prompt section. Return `None` (or empty) to contribute nothing — no dangling headers.
The `context` mapping carries the wired services (`retriever`, `sql_store`, `introspector`,
`lexicon`, `memory`, `previous_attempt`).

```python
class SignatureBlock:
    CONFIG_KEYS = frozenset({"heading"})
    def __init__(self, heading: str = "Account:") -> None:
        self._heading = heading
    @classmethod
    def from_config(cls, options): return cls(heading=str(options.get("heading", "Account:")))
    def render(self, record, context):
        tier = record.meta.get("tier")
        return f"{self._heading}\n  tier={tier}" if tier else None   # no section when absent
```

Two built-ins deserve their own mention because they need a database:

- **`sql_rows`** runs a parameterised query against the read-only `[sql]` store, binding `?`
  placeholders from `record.meta`, and shows the rows (`form_autofill` uses it for a customer's
  history). The `limit` is applied by the database, not by slicing in Python.
- **`schema`** shows the introspected tables and columns so the model uses real names
  (`nl_to_sql` uses it).

```toml
# context.toml — the customer's recent tickets, from the external database
[[context.block]]
kind = "sql_rows"
heading = "This customer's recent tickets:"
query = "SELECT subject, status FROM tickets WHERE customer_id = ? ORDER BY created DESC"
param_keys = ["customer_id"]      # bound positionally from record.meta["customer_id"]
limit = 10
```

### OutputSchema — `json_schema() -> dict`, `extract(reply) -> str`, `name`
Describes what the producer returns and pulls the output string out. Built-ins: `json_field` (one
string field) and `form` (many fields, incl. `array`). Selected in `recipe.toml` `output_schema`.

### Retriever — `retrieve(query, *, k, min_score=0.0) -> tuple[Retrieved, ...]`
A **corpus-free** custom retriever (its own backend) is named by dotted path in
`[reference].retriever` and built through the `RETRIEVERS` registry from `[reference].options`. A
**corpus-stateful** one is injected: `assemble(config_dir, retriever=my_retriever)`, the same way
`client_factory` and `extra_validators` are injected.

### A database driver — `SqlStore` / `VectorIndex` / `PairingStore` / `RunStore` / `LexiconStore`
See [§8](#8-every-storage-option-and-when-to-pick-it). Each driver module's docstring is the honest
account of that backend's trade-offs — read it before choosing.

### A model backend — `structured_request(messages, schema) -> StructuredRequest`
How a model family is asked for conforming JSON (a strict grammar vs a prompt-described shape).
Registered in `ragkit.llm.backends`; selected per model in `models.toml` `backend = "..."`.

### OutputMemory — the run's own accumulating context
Set `[task].use_memory = true` and the `established` context block shows the outputs already
produced for neighbouring records, so a long run stays internally consistent. It is seeded from the
run store on resume, so a restarted run's context is reproducible rather than starting empty.

> **`Provider` — the endpoint-kind seam.** `[endpoint.<name>].provider` drives real behaviour, not
> just a label. Every backend speaks the same OpenAI-compatible HTTP, so one transport serves them
> all; the provider declares what that endpoint *can* serve and the request quirks it needs.
> `llamacpp-router` (the default) serves chat, embeddings and rerank and takes llama.cpp's
> `chat_template_kwargs` reasoning toggle; `ollama` serves chat + embeddings but **has no rerank
> endpoint**, so a reranker on it is refused at build time rather than 404-ing mid-run;
> `openai-compatible` serves all three but never receives the llama.cpp-only field (a strict server
> would reject the whole request). A custom endpoint kind is a component like any other: call
> `ragkit.llm.register_provider(LlmProvider(...))` from your own module and name it in config.

## 13. The tool scripts

Every multi-step routine is a hardened script in `tools/` — each validates its own preconditions
and fails with an actionable message rather than half-running. `--help` on any of them prints the
full usage.

| Script | What it does |
|---|---|
| `setup_python_env.sh` | Builds `.venv/` from the pinned `requirements.txt`. |
| `check_environment.sh` | Reports on the whole environment (Python, venv, SQLite FTS5, optional deps, llama-server) as a table of pass/fail rows. Run it first when something is off. |
| `fetch_models.sh` | Downloads the pinned default GGUFs. Every entry is verified against the real Hugging Face API; a failure reports all of them rather than dying on the first. |
| `build_llama_cpp.sh` | Builds and pins the `llama-server` binary. |
| `serve_models.sh` | Starts llama.cpp in router mode, reading host/port/flags from a recipe's `models.toml` so serving and routing cannot drift. |
| `run_recipe.sh` | The whole loop for one recipe: serve → import → run → export → score, with the server's lifetime owned by the script (stopped on every exit path, so the next recipe's models fit in VRAM). |
| `init_storage.sh` | Creates a recipe's configured stores empty, before any import. |
| `embed_reference.sh` | Embeds the `[pairings]` corpus into the `[vector]` store. Resumable, batched, with periodic compaction — designed for multi-hour runs. |
| `build_vector_index.sh` | Builds the ANN index on a LanceDB table. **Not optional at scale**: without it, search scans every row. |
| `compact_vector_store.sh` | Consolidates the fragments many incremental upserts leave behind. |
| `migrate_storage.sh` | Folds a pre-overhaul recipe's on-disk artifacts into the current stores, without re-reading the original corpus. |
| `run_tests.sh` | The suite; `--coverage` adds the 100% statement+branch gate on `src/`. |
| `lint.sh` | ruff + mypy. |

## 14. Swapping components by config

The headline property: replace a component's internals by editing config, `git diff` over `src/`
empty. Each is a one-liner.

**The database** — SQLite ↔ DuckDB, or LanceDB ↔ Qdrant (see
[§8](#8-every-storage-option-and-when-to-pick-it) for which to pick):

```toml
# storage.toml
[sql]
driver = "duckdb"          # was "sqlite"

[vector]
driver = "qdrant"          # was "lancedb"; both pass the same conformance suite
path = "../data/v.qdrant"
dim = 1024

[pairings]
driver = "duckdb"          # was "sqlite"; the reference memory, same ranking either way
path = "../data/reference.pairings.duckdb"
```

**The retrieval stack** — swap the whole lexical/dense/hybrid assembly from `retrieval.toml`
([§7](#7-retrieval-modes-lexical-dense-hybrid)).

**A persona's model** — point `[[persona]] model = ...` at a different `[model.<name>]`; llama.cpp
loads each distinct model once, so two personas on one model share it.

**Your own component** — anywhere a `driver` / `kind` / `retriever` string is accepted, a dotted
path selects your class: `driver = "mypkg:MyStore"`, `kind = "mypkg:MyBlock"`. No framework change,
ever — that is the whole point.

---

**Next:** [`docs/config.md`](config.md) for every key; [`docs/architecture.md`](architecture.md) for
the layer boundaries, the durability model, and the evaluation layer; and the six
[`recipes/`](../recipes) for complete, real, worked examples — including their measured numbers.
