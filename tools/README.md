# tools/

Hardened scripts for the routines that take more than one command. Each sources
`lib/common.sh`, resolves the repo root from its own location, validates its preconditions,
and fails with an actionable message rather than crashing opaquely. The header comment of each
script *is* its `--help` text.

| Script | What it does |
|---|---|
| `setup_python_env.sh` | create `.venv/` and install the exact pinned `requirements.txt` |
| `check_environment.sh` | report, read-only, whether the environment can run the code and tests (probes SQLite FTS5 and loadable-extension support) |
| `run_tests.sh` | run the suite (needs no server/GPU/network); `--coverage` for statement + branch |
| `lint.sh` | static analysis: ruff (ruleset pinned in `ruff.toml`) + mypy; `--fix` to autofix |
| `fetch_models.sh` | download the pinned default GGUF models into `./models` |
| `serve_models.sh` | launch `llama-server` in **router mode**; `--config models.toml --endpoint <name>` reads the endpoint's launch flags from the same config the pool uses |
| `init_storage.sh` | create/open the stores a `storage.toml` describes (sqlite/duckdb SQL, lancedb/qdrant vector, fts5 lexical) |
| `lib/common.sh` | shared helpers, sourced by the others |

## Running a task

```bash
recipes/translation/fetch.sh                # download the recipe's real data (git-ignored), once
tools/fetch_models.sh                       # download the default GGUFs (once)
tools/serve_models.sh --config recipes/translation/config/models.toml --endpoint local --models-dir models
PYTHONPATH=src python -m ragkit.cli --config recipes/translation/config \
    -c recipes/translation/data/heldout.jsonl -j work/out.jsonl \
    --set source_language=English --set target_language=Polish
```

Evaluation is per-recipe: each recipe ships an `eval.py` scoring its own metric against the
held-out gold (`python -m recipes.<task>.eval ...`), and the generalised retrieval/output metrics
live in `ragkit.eval`. See `docs/tutorial.md`.

## From a fresh checkout

```bash
tools/setup_python_env.sh     # one-time: build the pinned environment
tools/check_environment.sh    # confirm it is ready
tools/run_tests.sh            # run everything
```

The scripts find `.venv/` on their own, so you do not need to activate it first.
