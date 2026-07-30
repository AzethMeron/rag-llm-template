# Tutorial: building pipelines with rag-llm-template

This guide shows how to *actually build solutions* on the framework: the mental model, a new
pipeline from scratch (config only, then with a plugin, then with retrieval and a database), and how
to extend every seam — a validator, a context block, an output schema, a retriever, a database
driver, a model backend — always **without editing framework code**.

Prerequisites: `tools/setup_python_env.sh` (builds `.venv/`). Every example is runnable with no
server using the test harness pattern in [§9](#9-testing-your-pipeline); to run against real models
you serve them with `tools/serve_models.sh` ([§7](#7-running-against-real-models)).

- [1. The mental model](#1-the-mental-model)
- [2. Anatomy of a recipe](#2-anatomy-of-a-recipe)
- [3. A pipeline from scratch — config only](#3-a-pipeline-from-scratch--config-only)
- [4. Add a mechanical validator (a plugin)](#4-add-a-mechanical-validator-a-plugin)
- [5. Add a memory (retrieval)](#5-add-a-memory-retrieval)
- [6. Read from a database](#6-read-from-a-database)
- [7. Running against real models](#7-running-against-real-models)
- [8. The extension points, one by one](#8-the-extension-points-one-by-one)
- [9. Testing your pipeline](#9-testing-your-pipeline)
- [10. Swapping components by config](#10-swapping-components-by-config)

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

## 2. Anatomy of a recipe

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
    storage.toml      # (optional) the external database + introspector; reference memory ([pairings]/[lexicon]) and run state ([run])
    retrieval.toml    # (optional) config-driven lexical/dense/hybrid retrieval
  plugins/            # (optional) your task-specific Validator / OutputSchema / ContextBlock / ...
  fetch.sh            # a hardened downloader for the real dataset (git-ignored data/)
  eval.py             # scores the exported journal against held-out gold
  tests/              # the recipe's own suite, against tiny committed fixtures
```

The full per-key reference for each config file is [`docs/config.md`](config.md). Below we build one.

## 3. A pipeline from scratch — config only

**Goal:** triage a customer-support message into a `category` and an `urgency`. No Python yet.

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
model_id = "Qwen3.5-2B-Instruct-Q4_K_M"
backend  = "auto"           # auto | generic | bielik | eurollm | gemma
context_window = 8192

[model.reviewer]
endpoint = "local"
model_id = "Qwen3.5-0.8B-Instruct-Q4_K_M"
```

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

The context — what goes in the prompt, in order. Each block contributes only when it has content
(no empty sections, ever). Here just a grounding instruction:

```toml
# recipes/triage/config/context.toml
[[context.block]]
kind = "literal"
text = "Read the support message below and triage it. Choose exactly one category and one urgency."
```

The recipe — the **output shape** and the input label. Use the built-in `form` output schema for a
multi-field object; a field of type `string` is asked for verbatim (an `enum` is enforced
mechanically in [§4](#4-add-a-mechanical-validator-a-plugin)):

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

Run it (once models are served — [§7](#7-running-against-real-models)):

```bash
PYTHONPATH=src python -m ragkit.cli import --catalog work/tickets.jsonl --run-db work/triage.db
PYTHONPATH=src python -m ragkit.cli run --config recipes/triage/config --run-db work/triage.db
PYTHONPATH=src python -m ragkit.cli export --run-db work/triage.db -j work/out.jsonl
```

Each record's `output` is the filled form as canonical JSON, e.g. `{"category": "billing",
"urgency": "high"}`, with `status = VERIFIED` once the panel accepts it.

## 4. Add a mechanical validator (a plugin)

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

## 5. Add a memory (retrieval)

Give the classifier **worked examples** retrieved per input — a labelled corpus of past tickets.
Point the recipe at a JSONL reference corpus and add the `retrieved` context block.

```
# recipes/triage/data/examples.jsonl   (git-ignored; a fetch.sh would build it)
{"source": "Refund the duplicate charge on my invoice", "label": "billing / high"}
{"source": "Where do I find the CSV export button", "label": "how_to / low"}
```

```toml
# recipes/triage/config/recipe.toml  (append)
[reference]
file = "../data/examples.jsonl"
retriever = "lexical"      # BM25 over the corpus, built from config alone
index_field = "source"     # what matching happens on
display_field = "label"    # what a hit shows (or omit for a sensible default)
```

```toml
# recipes/triage/config/context.toml  (insert before or after the literal block)
[[context.block]]
kind = "retrieved"
k = 3
min_score = 0.3
heading = "Similar past tickets and how they were triaged (examples, not the required answer):"
```

That is the whole "RAG" wiring for the default lexical retriever. With no `storage.toml`, the
corpus rebuilds in memory every run — fine for a small example. For anything larger, add a
`[pairings]` store so it imports once and persists (see [`docs/config.md`](config.md#storagetoml)):

```toml
# recipes/triage/config/storage.toml
[pairings]
driver = "sqlite"
path = "../data/examples.pairings.db"
```

`display_field` above is read only by the legacy in-memory fallback (no `[pairings]` configured).
`[pairings]` has no equivalent "show this arbitrary field alone" option — its display convention is
always `"source -> target"` (or bare `source` with no target), matching a translation-memory-shaped
pairing. `target_field = "label"` would show `"Refund the duplicate charge on my invoice -> billing
/ high"`, not `"billing / high"` alone; if a bare label display matters more than persistence, stay
on the legacy fallback.

To retrieve **densely** over a real vector database instead, add a `retrieval.toml` and a
`[vector]` store — see [§10](#10-swapping-components-by-config). The retriever is also injectable
in code (`assemble(..., retriever=my_retriever)`) for a corpus-stateful retriever you build
yourself.

## 6. Read from a database

Two blocks read an **external, read-only** SQL database (the framework never writes it):

- `sql_rows` runs a parameterised query, binding `?` placeholders from `record.meta`, and shows the
  rows — historical records for the entity being processed (see `form_autofill`).
- `schema` shows the introspected tables/columns so the model uses real names (see `nl_to_sql`).

```toml
# storage.toml — the external source, opened READ-ONLY
[sql]
driver = "sqlite"                 # or "duckdb" — one-line swap, same behaviour
path = "../data/business.sqlite"
read_only = true

[introspector]
driver = "sqlite"
path = "../data/business.sqlite"
```

```toml
# context.toml  (a block that puts the customer's recent tickets in the prompt)
[[context.block]]
kind = "sql_rows"
heading = "This customer's recent tickets:"
query = "SELECT subject, status FROM tickets WHERE customer_id = ? ORDER BY created DESC"
param_keys = ["customer_id"]      # bound positionally from record.meta["customer_id"]
limit = 10
```

The query, the read-only binding, and (for NL→SQL) a generated-SQL **safety** validator are the
framework's guardrails: a write through a `read_only` binding is refused *at the port*, and the
`nl_to_sql` recipe additionally parses the generated statement to a single schema-bounded `SELECT`.

## 7. Running against real models

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

The pool checks, **before contacting the server**, that you are not asking one endpoint to hold more
distinct models than it keeps resident (the thrash guard). Sampling is per-persona
(`[persona.sampling]`); genuinely launch-time server flags go in `[endpoint.<name>].server_args` and
`serve_models.sh --config` reads them, so `models.toml` is the single source for routing *and*
serving.

## 8. The extension points, one by one

Every seam is a `runtime_checkable` Protocol in `src/ragkit/core/ports.py`. A component declares its
own `CONFIG_KEYS` and (usually) a `from_config(options)`; it is selected by a **dotted path**
(`"mypkg.mod:MyClass"`), a published **entry point**, or a registered **built-in name** — checked in
that order. Anything you write is chosen the same way a built-in is.

### Validator — `validate(record, output, context) -> list[Violation]`
Code-decided acceptance checks. Shown in [§4](#4-add-a-mechanical-validator-a-plugin). Discipline:
**abstain unless there is positive evidence**, and a check that cannot be evaluated is *skipped*,
not reported as passed. Selected in `recipe.toml` `[[validator]] kind = "..."`.

### ContextBlock — `render(record, context) -> str | None`
Builds one prompt section. Return `None` (or empty) to contribute nothing — no dangling headers.
The `context` mapping carries the wired services (`retriever`, `sql_store`, `introspector`,
`lexicon`, `memory`, `previous_attempt`). Selected in `context.toml` `[[context.block]] kind =
"..."`. Built-ins: `literal`, `lexicon`, `neighbours`, `established`, `retrieved`,
`previous_attempt`, `sql_rows`, `schema`, `readings`.

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

### OutputSchema — `json_schema() -> dict`, `extract(reply) -> str`, `name`
Describes what the producer returns and pulls the output string out. Built-ins: `json_field` (one
string field) and `form` (many fields, incl. `array`). Selected in `recipe.toml` `output_schema`.

### Retriever — `retrieve(query, *, k, min_score=0.0) -> tuple[Retrieved, ...]`
A **corpus-free** custom retriever (its own backend) is named by dotted path in
`[reference].retriever` and built through the `RETRIEVERS` registry from `[reference].options`. A
**corpus-stateful** one is injected: `assemble(config_dir, retriever=my_retriever)`, the same way
`client_factory` and `extra_validators` are injected.

### A database driver — `SqlStore` / `VectorIndex` / `PairingStore` / `RunStore` / `LexiconStore`
Implement the port, put the third-party dependency **lazily inside your driver module only**
(the boundary test enforces confinement), and select it in `storage.toml` by dotted path
(`driver = "mypkg:MyVectorIndex"`) or register a built-in name. The conformance suite
(`tests/store/test_conformance.py`) is where a new driver earns "interchangeable".

### A model backend — `structured_request(messages, schema) -> StructuredRequest`
How a model family is asked for conforming JSON (a strict grammar vs a prompt-described shape).
Registered in `ragkit.llm.backends`; selected per model in `models.toml` `backend = "..."`.

## 9. Testing your pipeline

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

Run with `tools/run_tests.sh recipes/triage`. Keep `src/` at 100% statement+branch coverage (the
suite gates it); a recipe's own coverage is your responsibility to keep meaningful. The real
recipes' `tests/` are worked examples of adversarial validator tests and DB-swap parametrization.

## 10. Swapping components by config

The headline property: replace a component's internals by editing config, `git diff` over `src/`
empty. Each is a one-liner.

**The database** — SQLite ↔ DuckDB (both real, embedded, MIT/BSD), or LanceDB ↔ Qdrant:

```toml
# storage.toml
[sql]
driver = "duckdb"          # was "sqlite"

[vector]
driver = "qdrant"          # was "lancedb"; both pass the same conformance suite
path = "../data/v.qdrant"
dim = 1024

[pairings]
driver = "duckdb"          # was "sqlite"; the reference memory, identical retrieval either way
path = "../data/reference.pairings.duckdb"
```

**The retrieval stack** — swap the whole lexical/dense/hybrid assembly from `retrieval.toml`, with
the per-arm floors, candidate pool, MMR, and rerank model all in config:

```toml
# retrieval.toml
[retrieval]
kind = "hybrid"            # lexical | dense | hybrid
[retrieval.dense]
model = "embedder"         # a models.toml model of kind = "embedding"
[retrieval.rerank]
enabled = true
model = "reranker"         # a models.toml model of kind = "rerank"
```

**A persona's model** — point `[[persona]] model = ...` at a different `[model.<name>]`; llama.cpp
loads each distinct model once, so two personas on one model share it.

**Your own component** — anywhere a `driver` / `kind` / `retriever` string is accepted, a dotted
path selects your class: `driver = "mypkg:MyStore"`, `kind = "mypkg:MyBlock"`. No framework change,
ever — that is the whole point.

---

**Next:** [`docs/config.md`](config.md) for every key; [`docs/architecture.md`](architecture.md) for
the layer boundaries, the durability model, and the evaluation layer; and the six
[`recipes/`](../recipes) for complete, real, worked examples.
