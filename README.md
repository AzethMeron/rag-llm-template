# rag-llm-template

A heavily configurable **RAG + LLM framework** for building many different tasks on top of
local models — translation, form/template autofill from a database, natural-language → SQL,
and anything else that fits the *produce → check → review → revise* shape.

Every component — the databases, the retrieval stack, the model layer, the review harness —
is a replaceable part behind a typed interface. The built-ins are selected by configuration;
anything you write yourself is selected the same way, through a dotted path or an entry point,
with **no change to framework code**. Defaults are free and locally hosted.

> **Status: early construction.** The stdlib-only core contract (records, the component
> registry, the ports, config loading, display width, placeholders, terminology) is in place
> and fully tested. The layers above it — storage, ingestion, retrieval, the model pool, the
> harness, and the task recipes — are being built milestone by milestone. See
> `docs/architecture.md` for the shape of the whole thing.

## Layout

```
src/ragkit/
  core/      the contract: ports, records, registry, errors, config, width, placeholders, lexicon  [stdlib only]
  store/     sql / vector / lexical / blob drivers          [optional deps, lazily imported]
  ingest/    extract, normalise, dedup, chunk, embed
  retrieve/  lexical, dense, fusion (RRF/MMR), rerank, hybrid
  llm/       client, backends, model pool
  harness/   personas, panel, validators, context, memory, runner
  cli/
recipes/     fully-worked tasks (translation, form autofill, nl_to_sql), each with a fetch script for real data
config/ docs/ tools/ tests/ license/ .audit/
```

## Quick start

```bash
tools/setup_python_env.sh     # build the pinned environment (.venv/)
tools/check_environment.sh    # confirm it is ready
tools/run_tests.sh            # the suite — no server, GPU, or network required
tools/run_tests.sh --coverage # statement AND branch coverage
tools/lint.sh                 # ruff + mypy
```

The scripts find `.venv/` on their own; you do not need to activate it first.

## Design

`docs/architecture.md` covers the whole design: the enforced layer boundaries, the one
extension mechanism (a typed component registry per port), the two database roles (the
framework's own store versus an external read-only data source), the local model layer and
its router-mode model pool, and the harness. The package READMEs are closer to the code.

## License

Source-available and **dual-licensed**: free for any noncommercial use under the **PolyForm
Noncommercial License 1.0.0**, with **commercial use by a separate paid license**. See
[`license/`](./license/). Copyright © 2026 Jakub Grzana.
