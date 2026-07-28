# Recipe: form autofill

Fill the missing fields of a structured record from **historical records retrieved out of a real
relational database** — the "autofill a template from a database" task. Here the form is a music
track's `genre` and `unit_price`; the recipe fills them from the other tracks on the same album
(which almost always share both), retrieved live from the external database.

## What it demonstrates

- The **multi-field `FormSchema`** output (a reusable framework component): the producer is asked
  for exactly the declared fields, and the filled form is stored as canonical JSON so it can be
  scored field by field.
- **Retrieval from a relational database, not a document corpus** — the `sql_rows` context block
  runs a parameterised query (album id + the track's own id, so the answer is never leaked into its
  own prompt) against the read-only external store and puts the sibling rows in the prompt.
- A **field-value validator** (`plugins/validators.py`, `FieldTypesValidator`) selected by dotted
  path: it layers value-level checks (a field is actually filled, a price is > 0, a value is in an
  enum) on top of the schema's shape check — the extension API in use.
- A **per-persona sampling** override (`[persona.sampling] temperature = 0.1`): the author is kept
  cool because the fill should be grounded in the retrieved rows, not invented.

## Data (fetched, never committed)

```bash
recipes/form_autofill/fetch.sh                 # Chinook sample DB (MIT), ~1 MB
```

This downloads `Chinook_Sqlite.sqlite` and, holding out each eligible track's genre and price,
writes under `data/`:

- `chinook.sqlite` — the real relational database the sibling rows are retrieved from,
- `heldout.jsonl` — the tracks to fill (each carrying its album/track ids in `meta`),
- `gold.jsonl` — the true genre and price that were held out, for scoring.

A tiny in-test SQLite fixture backs the recipe's own tests, so they need no download.

## Running

```bash
tools/serve_models.sh --config recipes/form_autofill/config/models.toml --endpoint local \
    --models-dir models
PYTHONPATH=src:. python -m ragkit.cli --config recipes/form_autofill/config \
    -c recipes/form_autofill/data/heldout.jsonl -j work/form.jsonl
```

## Evaluation

`recipes/form_autofill/eval.py` scores **held-out-field fill accuracy** — genre (case-insensitive)
and unit price (within a small tolerance) against `gold.jsonl`, plus the stricter "both fields
correct" rate. It reads only the journal and gold, so it needs no server or database.

```bash
PYTHONPATH=src:. python -m recipes.form_autofill.eval \
    --journal work/form.jsonl --gold recipes/form_autofill/data/gold.jsonl
```
