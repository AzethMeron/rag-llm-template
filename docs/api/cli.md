# `ragkit.cli`

`ragkit.cli` is the command-line entry point layer: four subcommands over a durable run store
(`import` / `run` / `export` / `writeback`), and the `assemble()` function that wires a complete,
runnable recipe (harness, model pool, storage, retriever) from a config directory's `.toml` files.
`ragkit.cli.app` does the config-to-object assembly and all of its consistency checking, strictly
before any server is contacted; `ragkit.cli.main` builds the `argparse` parser, dispatches to the
four command handlers, and turns any `RagkitError` into a clean exit code instead of a traceback.
Invoke it as `python -m ragkit.cli <command> ...`.

## ragkit.cli (package `__init__`)

Re-exports `assemble`, `Assembled`, and `CliError` from `ragkit.cli.app`. Nothing else is public at
the package level; `ragkit.cli.main`'s command functions and `build_parser`/`main` are reached via
`ragkit.cli.main`, not re-exported here.

## ragkit.cli.__main__

`python -m ragkit.cli` entry point. Imports `main` from `.main` and, when run as `__main__`, calls
`raise SystemExit(main())` — `main()` is invoked with no explicit `argv`, so it parses
`sys.argv[1:]`. This module has no other content and defines no public names.

## ragkit.cli.app

Assembles a run from a configuration directory: `models.toml`, `personas.toml`, `rules.toml`,
`context.toml`, an optional `storage.toml`, and a `recipe.toml` that names the task's output
schema, its extra validators, its input label, and (optionally) a reference corpus to retrieve
from. Every component is resolved through its registry (`OUTPUT_SCHEMAS`, `VALIDATORS`,
`RETRIEVERS`), so a user's own driver, validator, block, or schema is selected the same way a
built-in one is.

The assembly does all of its config reading and consistency checks (including the model pool's
capacity/thrash guard) *before* any server is contacted, so a misconfiguration is reported for what
it is rather than masked by a "no server" error downstream. That ordering is the guarantee —
`assemble()` does *not* promise to never touch a server: with `retrieval.kind` of `dense` or
`hybrid`, building the retriever imports and embeds the reference corpus, which does call the
embedding endpoint, but only after every config check has already passed.

### CliError

`ConfigError` subclass (and therefore ultimately a `RagkitError`). Raised when a run cannot be
assembled as configured — the family of `assemble()`-specific validation failures listed under
`assemble` below (as opposed to a lower-level `ConfigError` raised directly by one of the `.toml`
loaders `assemble` calls, e.g. an unknown key in `models.toml`).

### Assembled

Everything a run needs, built and consistency-checked before any server is contacted. Returned by
`assemble()`.

**Attributes:**
- `harness: Harness` — the fully wired produce/review/revise loop (`ragkit.harness.Harness`).
- `pool: ModelPool` — the model pool (`ragkit.llm.pool.ModelPool`); a context manager, used as
  `with assembled.pool: ...` around the run (see `cmd_run`).
- `storage: Storage` — the resolved stores (`ragkit.store.Storage`); any field may be `None` if
  `storage.toml` didn't configure that port.
- `retriever: Retriever | None` — the reference retriever actually wired into the harness, or
  `None` if the recipe has none configured.

Dataclass with `slots=True` (not frozen — fields are mutable).

#### assemble(config_dir: Path, *, substitutions: dict[str, str] | None = None, client_factory: ClientFactory | None = None, extra_validators: list[Any] | None = None, retriever: Retriever | None = None) -> Assembled

Builds an `Assembled` run from the config files in `config_dir`.

**Args:**
- `config_dir` — directory containing `models.toml`, `personas.toml`, `rules.toml`,
  `context.toml`, `recipe.toml`, and optionally `storage.toml` / `retrieval.toml` /
  `lexicon.jsonl`. Coerced through `Path(...)`.
- `substitutions` — fills `{placeholder}` tokens in persona instructions (e.g. the language pair
  for a translation task), passed through to `load_panel`. Substitution is literal string
  replacement, not `str.format`, since instructions are prose that may legitimately contain braces;
  an unfilled placeholder is an error there, never a stray brace reaching the model.
- `client_factory` — injects the HTTP client used for every model/embedding/rerank endpoint (model
  pool, and — when built here — the embedding/rerank clients for `retrieval.toml`-driven
  retrieval). Lets a test drive the whole run against an in-memory transport with no real server.
- `extra_validators` — task-specific validator instances a recipe constructs directly and passes
  in, appended after whatever `[[validator]]` entries `recipe.toml` names.
- `retriever` — injects a pre-built `Retriever`, overriding whatever `[reference]`/`retrieval.toml`
  would otherwise build. This is the seam for a *corpus-stateful* custom retriever (one that must
  hold the framework's own reference corpus) — a recipe or host constructs it and passes it here,
  the same injection pattern as `client_factory` and `extra_validators`. A *corpus-free* custom
  retriever (its own backend) needs no injection: name it by dotted path or entry-point name in
  `[reference].retriever` and it resolves through the `RETRIEVERS` registry.

**Returns:** an `Assembled` with the harness, pool, storage, and resolved retriever.

**What it does, in order:**
1. Loads the model pool from `models.toml` (`load_models`, with `client_factory`).
2. Loads the persona panel from `personas.toml` (`load_panel`, with `substitutions`).
3. Loads the rule set from `rules.toml` (`RuleSet.load`).
4. Loads the context assembler config from `context.toml` (`load_context`).
5. Loads storage from `storage.toml` if present, else an empty `Storage()`.
6. Parses `recipe.toml` (strict: `reject_unknown` at the top level and within `[task]` and
   `[reference]`; every `[[validator]]` entry must have a `kind`).
7. Checks the model pool's capacity against the panel's models-in-use (`pool.check_capacity`) —
   the "must not hold more distinct models than an endpoint keeps resident" guard, run before any
   server contact.
8. Resolves the output schema (`OUTPUT_SCHEMAS.create`) and reads `lexicon.jsonl`
   (`read_lexicon`).
9. Builds the recipe's named validators (`VALIDATORS.create` per `[[validator]]` entry) and
   appends `extra_validators`.
10. Resolves the external read-only SQL store (only exposed if `storage.sql` is set *and*
    `read_only`; the framework's own store, if any, stays write-only-internal).
11. Resolves the retriever: the injected `retriever` argument if given; else, if `retrieval.toml`
    exists in `config_dir`, the full lexical/dense/hybrid stack built from it (see "retrieval.toml
    validation" below); else the legacy `[reference]`-table path (a plain lexical retriever built
    from the recipe's reference corpus, or a corpus-free custom retriever named in
    `[reference].retriever`).
12. Builds the `ValidatorPipeline` (rule set, lexicon, extra validators, and a `shared` dict of
    `sql_store`/`introspector`/`retriever` available to any pluggable validator).
13. Constructs the `Harness` (pool, panel, rule set, output schema, pipeline, context; memory is
    `OutputMemory()` if `recipe.toml`'s `[task].use_memory` is true, else `None`; also given the
    retriever, external SQL store, introspector, `input_label`, and `stand_in` from the recipe).
14. Returns `Assembled(harness, pool, storage, retriever)`.

**retrieval.toml validation** (only reached when `retrieval.toml` exists in `config_dir`; parsed by
`ragkit.retrieve.tuning.load_retrieval` into a `RetrievalSettings`):
- `[retrieval].kind` must be `"lexical"`, `"dense"`, or `"hybrid"`.
- Several keys apply **only** under `kind="hybrid"` — `[retrieval].candidate_pool`,
  `[retrieval].mmr_lambda`, `[retrieval.lexical].min_score`, `[retrieval.dense].min_score`,
  `[retrieval.rerank].enabled`/`.model`/`.score_scale` — because only the hybrid stack fuses,
  builds a candidate pool, diversifies with MMR, or re-scores with a reranker. Setting any of these
  under `kind="lexical"` or `kind="dense"` is refused (`ConfigError`), not silently parsed and
  dropped.
- **Reranking specifically is refused outside `kind="hybrid"`**: if `[retrieval.rerank].enabled =
  true` while `kind` is `"lexical"` or `"dense"`, `RetrievalSettings.__post_init__` raises
  `ValueError` (wrapped by `load_retrieval` into `ConfigError`) — "reranking re-scores the fused
  candidate pool, which only the hybrid stack builds." This is a cross-field check independent of
  (and in addition to) the per-key applicability check above: setting `[retrieval.rerank].model`
  alone (without `enabled = true`) under a non-hybrid `kind` is caught by the applicability check
  instead, since `model`'s presence is checked regardless of `enabled`.
- `[retrieval.dense].model` (an embedding model name from `models.toml`) is required when `kind` is
  `"dense"` or `"hybrid"`, and refused when it is set under `kind="lexical"`.
- Beyond `retrieval.toml`'s own validation, `assemble()`/`_build_retrieval` add: the reference
  corpus file must exist; a `dense`/`hybrid` `kind` needs a `[vector]` store configured in
  `storage.toml`; the named embedding model must actually be of `kind="embedding"` in
  `models.toml` (and the named rerank model of `kind="rerank"`) — each a `CliError` if violated.

**Raises:**
- `ConfigError` (or a subclass) — from any of the `.toml` loaders (`load_models`, `load_panel`,
  `RuleSet.load`, `load_context`, `load_storage`, recipe parsing, `load_retrieval`) for a missing
  file, unknown key, or mistyped value; see "retrieval.toml validation" above for that file's
  specific rules.
- `CliError` — a `ConfigError` subclass, for `assemble`-level failures: reference corpus file not
  found; `retrieval.toml` present but the recipe sets no `[reference].file`; `retrieval.kind` needs
  a `[vector]` store but `storage.toml` has none; the recipe has a `[reference].file` but
  `storage.toml` sets no `[pairings]` store to import into; a `[reference].retriever` /
  `[retrieval.dense].model` / `[retrieval.rerank].model` name that doesn't resolve to the expected
  kind of thing; a `ReferenceImportError` from importing the reference corpus (re-raised as
  `CliError` with the original `reason`).
- `RegistryError` (from `ragkit.core.registry`, not wrapped) — an unregistered `output_schema`,
  validator `kind`, or `[reference].retriever` name.
- `ModelPoolError` (from `ragkit.llm.pool`, not wrapped) — from `pool.check_capacity` if the panel's
  models-in-use would exceed the pool's configured capacity.

**Side effects:** with `retrieval.kind` of `dense` or `hybrid` (or a corpus-stateful custom
retriever import path), building the retriever calls `import_reference`, which both writes into
`storage.pairings` (resumable — a no-op past the first successful import) and calls the embedding
endpoint over HTTP through `client_factory` (or a real client if none was injected). This is the
one place `assemble()` can reach a network, and it happens only after every config check above has
passed.

## ragkit.cli.main

The `argparse`-based CLI: four subcommands over a run store. `import` loads a JSONL catalogue into
it; `run` assembles a config directory and executes pending records against it; `export` writes the
store's results back out as a `journal.jsonl`-compatible file; `writeback` folds a finished run's
verified outputs into the reference memory as new pairings. Splitting execution from the JSONL
catalogue this way is what makes a run durable in a real database rather than a flat file:
`import`/`export` are the bridge to and from the format a fetch script or a recipe's `eval.py` still
speaks, and `run` itself knows nothing about JSONL (or about any particular task — the output
schema, validators, context blocks, and reference corpus all come from the config directory,
resolved through registries via `assemble`).

Every subcommand shares `--log-file PATH` (default `work/ragkit.log`) and `--no-log-file`, handled
by `main()`/`_configure_logging` rather than per-command: the `ragkit` logger is set to `WARNING`,
given a `StreamHandler` to stderr (filtered to drop records marked `leniency_suppress_console` or
`console_shown`, so a message already printed to the console by the CLI itself, or one a reviewer's
leniency setting asked to suppress, isn't duplicated) and, unless `--no-log-file`, an unfiltered
`FileHandler` at the given path (parent directories created as needed).

#### cmd_import(args: argparse.Namespace) -> int

Handler for `ragkit import`.

**Flags:** `-c`/`--catalog PATH` (default `work/records.jsonl`) — the JSONL catalogue to load;
`--run-db PATH` (default `work/run.db`) — the run store.

**Behavior:** opens `SqliteRunStore(args.run_db)`, calls `import_jsonl(store, args.catalog)`
(idempotent on a `record_id` already present — re-running an import is safe), prints `"{added:,}
record(s) added to {run_db} ({total:,} total)"`, returns `0`.

**Reads:** `args.catalog`. **Writes:** `args.run_db` (creates/appends to the SQLite run store).

#### cmd_run(args: argparse.Namespace) -> int

Handler for `ragkit run`.

**Flags:** `-C`/`--config PATH` (required) — the config directory passed to `assemble`; `--run-db
PATH` (default `work/run.db`); `--concurrency N` (default 2, `_positive`-validated: must be `>= 1`)
— records produced at once against the same pool; `--limit N` (`_positive`-validated) — stop after
N records, for trials; `--set KEY=VALUE` (repeatable) — a persona-instruction substitution (e.g.
`--set source_language=English`); collected via `_substitutions`, which raises `CliError` for a
malformed pair (no `=`) or an empty key.

**Behavior:** builds `substitutions` from `--set`, calls `assemble(args.config,
substitutions=substitutions)`, opens the run store, seeds the harness's output memory from
already-completed results (`_seed_memory` — a no-op if the recipe has no memory wired, or the store
is empty), takes `store.pending()` (optionally truncated to `--limit`). If there is nothing
pending, prints `"nothing to do: no pending records in the run store"` and returns `0` without
opening the pool. Otherwise, enters `assembled.pool` as a context manager, prints the queued count
and concurrency, and calls `run_batch(harness, records, store, concurrency=..., already_done=...,
clock=time.monotonic, on_progress=...)`, printing each progress update
(`"  {progress.summary()}"`) as it arrives and the final summary at the end. Returns `0`.

**Reads:** the config directory's `.toml`/`.jsonl` files (via `assemble`), `args.run_db`'s pending
records. **Writes:** results back into `args.run_db` as records complete (via `run_batch`).

**Raises:** propagates whatever `assemble()` raises (see above) and whatever `run_batch` raises;
`CliError` directly from `_substitutions` for a malformed `--set`.

#### cmd_export(args: argparse.Namespace) -> int

Handler for `ragkit export`.

**Flags:** `--run-db PATH` (default `work/run.db`); `-j`/`--journal PATH` (default
`work/journal.jsonl`).

**Behavior:** opens the run store, calls `export_jsonl(store, args.journal)`, prints `"{written:,}
result(s) exported to {journal}"`, returns `0`.

**Reads:** `args.run_db`. **Writes:** `args.journal` (a `journal.jsonl`-compatible file).

#### cmd_writeback(args: argparse.Namespace) -> int

Handler for `ragkit writeback`.

**Flags:** `--run-db PATH` (default `work/run.db`); `--pairings-db PATH` (default
`work/pairings.db`); `--pairings-driver NAME` (default `"sqlite"`, resolved via `PAIRING_STORES`);
`--vector-path PATH` (optional) — reconcile this vector index after write-back, requires
`--embedding-url`; `--vector-driver NAME` (default `"lancedb"`, resolved via `VECTOR_INDEXES`);
`--vector-dim N` (optional) — the vector index's dimension, only meaningful with `--vector-path`;
`--embedding-url URL` (optional) — the embedding endpoint's base URL; `--embedding-model NAME`
(default `"local"`).

**Behavior:** opens the run store and the pairings store, builds an optional
`(VectorIndex, EmbeddingClient)` pair via `_writeback_vector` (only if `--vector-path` is given;
raises `CliError` if `--vector-path` is given without `--embedding-url`), wraps the pairings store
in a `PairingSink`, calls `write_back(run_store, sink, pairing_store, vector=..., embedder=...)`,
prints `"{added:,} pairing(s) written back to {pairings_db}"`, returns `0`.

Notably standalone: writeback does not load a recipe's `models.toml` — the embedding endpoint is
named directly via `--embedding-url`/`--embedding-model` rather than resolved by logical model
name, since writeback is a post-run step independent of any recipe's config directory.

**Reads:** `args.run_db` (finished run results). **Writes:** `args.pairings_db` (new pairings), and
`args.vector_path` if given (index reconciliation).

**Raises:** `CliError` if `--vector-path` is set without `--embedding-url`.

#### build_parser() -> argparse.ArgumentParser

Builds the full `argparse.ArgumentParser` (`prog="ragkit"`) with four required subcommands
(`dest="command", required=True`): `import`, `run`, `export`, `writeback`, each as described above,
each sharing the `--log-file`/`--no-log-file` parent parser. Each subparser's `set_defaults(handler=
cmd_*)` wires it to its handler function. Returns the parser; does not parse anything itself.

#### main(argv: list[str] | None = None) -> int

The process entry point.

**Args:** `argv` — argument list to parse; `None` (the default) means `argparse` reads
`sys.argv[1:]`.

**Behavior:** builds the parser, parses `argv`, configures logging (`None` log file if
`--no-log-file`, else `args.log_file`), then calls `args.handler(args)` (one of the four `cmd_*`
functions) inside a `try`/`except RagkitError`. On success returns `int(args.handler(args))`. On a
caught `RagkitError`, prints `f"ragkit: {exc}"` to stderr, logs it at `error` level (with
`extra={"console_shown": True}`, so the file handler still records it but the console filter
doesn't print it twice), and returns `1`.

**Returns:** the handler's exit code (normally `0`), or `1` on a `RagkitError`.

**Raises:** none — this is the top-level boundary; any exception that is not a `RagkitError`
(a bug, not a configuration or runtime-data problem this framework anticipates) is intentionally
left to propagate as an unhandled traceback rather than being masked as a clean exit code.

**Side effects:** configures the `ragkit` logger (handlers, level, propagation) as a side effect of
every invocation; prints to stdout (via the handlers) and/or stderr (on error).
