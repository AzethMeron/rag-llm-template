# Recipe: natural language → SQL

Turn a question into a single read-only `SELECT` against a real database. The schema is introspected
and shown to the model; the review panel checks that the query answers the question; and a
**generated-SQL safety validator** refuses anything that is not a single, schema-bounded `SELECT`
before it can reach a database.

## Why this recipe is a security surface

A task that emits database commands is exactly where "all failure modes included" has teeth, so the
safety is defence-in-depth and lives in the framework, not as an afterthought:

1. **Read-only by construction.** The external database is wired as a `read_only` `SqlStore`
   (`storage.toml`), opened through SQLite's `mode=ro` URI. A write is refused *at the port*, before
   the database — so even a destructive statement that somehow slipped through could not take effect.
2. **Parse, don't pattern-match the payload** (`plugins/validators.py`). Comments are stripped first
   (a keyword cannot hide behind `--`), then a single `SELECT`/`WITH … SELECT` is required — no
   stacked statements, no DDL/DML keyword anywhere.
3. **Schema-bounded.** Every referenced table must exist in the introspected schema; an unknown
   table is both an injection signal and a hallucination signal.
4. **Executable check.** When the read-only store is wired, the statement is dry-run through
   `EXPLAIN`, catching a syntax error or unknown column the text checks miss.

Anything that fails becomes a blocking `Violation`, and the record ends `REJECTED` — never executed.
The adversarial tests in `tests/test_recipe.py` (DROP/DELETE/UPDATE, stacked-statement injection,
comment smuggling, unknown-table, `PRAGMA`/`ATTACH`) assert exactly that, including that a run whose
model is forced to emit `DROP TABLE` leaves the database's row count unchanged.

## What it demonstrates

- The **two-database split**: the external task data source is a *separate*, read-only `SqlStore`,
  never the framework's own writable store. Conflating them is what would let a generated `DELETE`
  reach the wrong database.
- A `SchemaIntrospector` feeding a `schema` context block — the model sees real tables and columns
  from config alone, no framework change.
- A task-specific safety validator selected by dotted path in `recipe.toml` — the extension API in
  use.

## Data (fetched, never committed)

```bash
recipes/nl_to_sql/fetch.sh --url <spider.zip location> [--db concert_singer]
```

Spider (Yale, CC BY-SA 4.0) ships **actual SQLite databases**, so the external `SqlStore` points
straight at a downloaded `.sqlite` file. The script writes, under `data/`:

- `database.sqlite` — the chosen Spider database (introspected and queried),
- `heldout.jsonl` — the held-out questions to answer,
- `gold.jsonl` — their gold SQL, for execution-accuracy scoring.

The recipe's own unit tests build a tiny SQLite fixture in `tmp_path`, so they need no download, and
the end-to-end run is exercised against **both real `SqlStore` drivers** (SQLite and DuckDB) —
swapped by a one-line `storage.toml` driver edit, no code change — proving the external database is
interchangeable.

## Running

```bash
tools/serve_models.sh --models-dir models
PYTHONPATH=src:. python -m ragkit.cli import --catalog recipes/nl_to_sql/data/heldout.jsonl \
    --run-db work/nl_to_sql.db
PYTHONPATH=src:. python -m ragkit.cli run --config recipes/nl_to_sql/config \
    --run-db work/nl_to_sql.db
PYTHONPATH=src:. python -m ragkit.cli export --run-db work/nl_to_sql.db -j work/nl_to_sql.jsonl
```

`storage.toml` already points at `data/database.sqlite` (relative to the config directory), so no
edit is needed once `fetch.sh` has run — pass a different `--db` to `fetch.sh` to use another Spider
database.

## Evaluation

`recipes/nl_to_sql/eval.py` scores **execution accuracy** — the standard Spider metric: it runs each
produced query and its gold query against the read-only database and compares result sets
(positionally by row; order-sensitive only when the gold query has an `ORDER BY`). A rejected or
unrunnable query counts as a miss, never a crash.

```bash
PYTHONPATH=src:. python -m recipes.nl_to_sql.eval --config recipes/nl_to_sql/config \
    --journal work/nl_to_sql.jsonl --gold recipes/nl_to_sql/data/gold.jsonl
```

It executes only the read-only binding, so evaluation cannot mutate the data either.
