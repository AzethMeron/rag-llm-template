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
| `lint.sh` | static analysis: ruff (ruleset pinned in `ruff.toml`) + mypy `--strict` (pinned in `mypy.ini`); `--fix` to autofix ruff's findings |
| `fetch_models.sh` | download the pinned default GGUF models into `./models` |
| `serve_models.sh` | launch `llama-server` in **router mode**; `--config models.toml --endpoint <name>` reads the endpoint's launch flags from the same config the pool uses |
| `init_storage.sh` | create/open the stores a `storage.toml` describes (sqlite/duckdb SQL, lancedb/qdrant vector, sqlite/duckdb pairings, sqlite run/lexicon) |
| `migrate_storage.sh` | fold a pre-storage-overhaul recipe's on-disk artifacts (a legacy row store, a `lexicon.jsonl`, a catalog+journal pair) into the current DB-native stores — the one-time bridge for data that predates `[pairings]`/`[run]`/`[lexicon]` |
| `embed_reference.sh` | embed a `[pairings]` store's rows into its `[vector]` index (dense/hybrid retrieval needs this; import alone only builds the lexical/BM25 side) — resumable, only embeds what `VectorIndex.reconcile` reports missing |
| `lib/common.sh` | shared helpers, sourced by the others |

## Running a task

```bash
recipes/translation/fetch.sh                # download the recipe's real data (git-ignored), once
tools/fetch_models.sh                       # download the default GGUFs (once)
tools/serve_models.sh --config recipes/translation/config/models.toml --endpoint local --models-dir models
PYTHONPATH=src python -m ragkit.cli import --catalog recipes/translation/data/heldout.jsonl \
    --run-db work/translation.db                                # load the catalogue once
PYTHONPATH=src python -m ragkit.cli run --config recipes/translation/config \
    --run-db work/translation.db \
    --set source_language=English --set target_language=Polish  # execute pending records
PYTHONPATH=src python -m ragkit.cli export --run-db work/translation.db -j work/out.jsonl
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
