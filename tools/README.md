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
| `serve_models.sh` | launch `llama-server` in **router mode** (one endpoint routes by the `model` field) |
| `init_storage.sh` | create/open the stores a `storage.toml` describes (SQLite schema, LanceDB table) |
| `lib/common.sh` | shared helpers, sourced by the others |

## Running a task

```bash
tools/fetch_models.sh                       # download the default GGUFs (once)
tools/serve_models.sh --models-dir models   # start the router-mode endpoint
tools/init_storage.sh --config recipes/translation/config   # ready the stores
PYTHONPATH=src python -m ragkit.cli --config recipes/translation/config \
    -c work/records.jsonl -j work/journal.jsonl --set source_language=English --set target_language=Polish
```

The evaluation gates (`eval_retrieval.sh`, `eval_output.sh`) land with the recipes.

## From a fresh checkout

```bash
tools/setup_python_env.sh     # one-time: build the pinned environment
tools/check_environment.sh    # confirm it is ready
tools/run_tests.sh            # run everything
```

The scripts find `.venv/` on their own, so you do not need to activate it first.
