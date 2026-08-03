# rag-llm-template

A heavily configurable **RAG + LLM framework** for building many different tasks on top of local
models — translation, form/template autofill from a database, natural-language → SQL, grounded
decision support, and anything else that fits the *produce → mechanical check → review panel →
revise* shape.

Every component — the databases, the retrieval stack, the model layer, the review harness — is a
replaceable part behind a typed interface. The built-ins are selected by configuration; anything
you write yourself is selected the same way, by a dotted path or an entry point, with **no change
to framework code**. Defaults are free and locally hosted, and the whole test suite runs with **no
server, GPU, or network**.

## What you can build

A task is a directory of TOML config plus (optionally) a small plugin. Six are worked end-to-end,
each fetching real data on demand and none of which required a framework change:

| Recipe | Task | Proves |
|---|---|---|
| [`translation`](recipes/translation) | translate with a retrieved translation memory | a **faithful port of `AzethMeron/llm-translator`** — same prompts, 5-reviewer panel, rules, context |
| [`nl_to_sql`](recipes/nl_to_sql) | natural language → a safe SQL `SELECT` | the two-database split + a generated-SQL **safety** validator (adversarially tested) |
| [`form_autofill`](recipes/form_autofill) | fill a record's missing fields from a relational DB | a multi-field `FormSchema` + retrieval from SQL rows |
| [`predictive_maintenance`](recipes/predictive_maintenance) | decide from a manuals **memory** + sensor request | retrieval memory *and* a structured request, with a **grounding** validator |
| [`med_evidence`](recipes/med_evidence) | grounded yes/no/maybe biomedical decision from retrieved abstracts | a **~600k-doc on-disk corpus** (ClinicalTrials.gov + PubMedQA gold) with a hallucination-blocking grounding validator |
| [`legal_procurement`](recipes/legal_procurement) | Polish legal / procurement passage retrieval | the **retrieval-metrics eval** (Recall@20/MRR@10/NDCG@10) over the **full 7.1M-passage polqa corpus**, ingested and queried at ~37 MB RSS |

New to the project? **Start with [`docs/tutorial.md`](docs/tutorial.md)** — it builds a pipeline
from scratch and shows how to extend every seam.

## Quick start

```bash
tools/setup_python_env.sh      # build the pinned environment (.venv/)
tools/check_environment.sh     # confirm it is ready
tools/run_tests.sh --coverage  # the suite — no server, GPU, or network; 100% statement+branch
tools/lint.sh                  # ruff + mypy
```

The scripts find `.venv/` on their own; you do not need to activate it first. To run a recipe
against real models:

```bash
tools/build_llama_cpp.sh                                       # one-time: build llama-server (pinned commit)
tools/fetch_models.sh                                          # one-time: download the pinned default GGUFs
recipes/translation/fetch.sh                                  # download the real dataset (git-ignored)
tools/serve_models.sh --config recipes/translation/config/models.toml --endpoint local --models-dir models
PYTHONPATH=src python -m ragkit.cli import --catalog recipes/translation/data/heldout.jsonl \
    --run-db work/translation.db                               # load the catalogue once
PYTHONPATH=src python -m ragkit.cli run --config recipes/translation/config \
    --run-db work/translation.db \
    --set source_language=English --set target_language=Polish  # execute pending records
PYTHONPATH=src python -m ragkit.cli export --run-db work/translation.db -j work/out.jsonl
```

## Layout

```
src/ragkit/
  core/      the contract: ports, records, registry, errors, config, width, placeholders, lexicon, jsonshape  [stdlib only]
  store/     sql (sqlite | duckdb), vector (lancedb | qdrant), pairings (sqlite | duckdb, co-located rows + search index), run (sqlite), lexicon (sqlite)   [optional deps, lazily imported]
  ingest/    extract, normalise, dedup, chunk, streaming reference-corpus import + embed, write-back into reference memory
  retrieve/  lexical, dense, fusion (RRF/MMR), rerank, hybrid, config-driven assembly (tuning)
  llm/       client, backends, model pool + thrash guard, serve-args
  harness/   personas, panel, validators, context blocks, memory, output schemas, runner
  eval/      retrieval metrics (Recall@k/MRR/MAP/NDCG) + a blinded A/B judge
  cli/       assemble a run from a config directory, and execute it
recipes/     six fully-worked tasks, each with config/, plugins/, a hardened fetch.sh, eval.py, tests/
docs/ tools/ tests/ license/ .audit/
```

Two **real** drivers ship behind each database port (`sqlite`↔`duckdb`, `lancedb`↔`qdrant`), so
swapping a database is a one-line config edit proven by the conformance suite — see the tutorial.

Retrieval never holds the corpus in RAM: a search index (FTS5 BM25, or the vector ANN) returns
only ids, and every hit resolves its text + metadata through the `PairingStore` (the `[pairings]`
store) that owns it — its rows and its FTS5 search index are co-located in one database, kept in
sync by triggers inside the same transaction, so they cannot drift apart. Import streams the
source in batches into the on-disk store and builds it once, so a multi-GB, multi-million-row
corpus ingests and queries at a few tens of MB of RSS — `legal_procurement` does exactly this at
the scale of a multi-million-passage polqa corpus. Run state is equally DB-native: `RunStore`
(`[run]`, WAL SQLite) replaces a JSONL catalogue/journal as the thing a run actually reads and
writes; `ragkit import`/`export` bridge to and from JSONL at the edges, and `ragkit writeback`
folds a finished run's verified outputs back into the reference memory as new pairings — the
accumulating-memory use case, a deliberate, separate post-run step, never automatic.

## Documentation

| Read | For |
|---|---|
| [`docs/tutorial.md`](docs/tutorial.md) | **how to actually build things** — a new pipeline from scratch, and every extension point with runnable examples |
| [`docs/architecture.md`](docs/architecture.md) | the shape of the whole thing: layer boundaries, the one extension mechanism, the `Record`, durability, evaluation |
| [`docs/config.md`](docs/config.md) | the per-file config reference — every key, type, default, and meaning |
| [`docs/api/`](docs/api) | the per-symbol API reference — every public module, class, and function with real signatures, arguments, returns, and errors |
| [`recipes/<task>/README.md`](recipes) | each worked recipe: its data, prompts, validators, and eval |
| [`.audit/`](.audit) | design investigations kept across sessions (e.g. the circular-metric lesson the eval layer encodes) |

## License

Source-available and **dual-licensed**: free for any noncommercial use under the **PolyForm
Noncommercial License 1.0.0**, with **commercial use by a separate paid license**. See
[`license/`](./license/). Copyright © 2026 Jakub Grzana.
