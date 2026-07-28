# Recipe: translation

Translation with a retrieved **translation memory**, in the manner the framework was generalised
from. The reference corpus (prior translations) is retrieved per line and shown to the model as
worked examples; the review panel checks fidelity and fluency; the untranslated-echo validator
refuses a line the model handed back in the source language.

## What it demonstrates

- A reference `Retriever` wired from a JSONL corpus, retrieved into the prompt (`context.toml`).
- A task-specific validator plugin (`plugins/validators.py`) selected by dotted path in
  `recipe.toml` — the extension API in use, no framework change.
- The full produce → check → review → revise loop over real config.

## Data (fetched, never committed)

```bash
recipes/translation/fetch.sh                 # Tatoeba English-Polish (CC-BY 2.0 FR)
```

This downloads the pairs and writes, under `data/`:

- `reference.jsonl` — the translation memory retrieved from,
- `heldout.jsonl` — lines to translate (held out of the corpus),
- `gold.jsonl` — their real translations, for scoring.

A tiny committed `sample/reference.jsonl` backs the recipe's own tests, so they need no download.

## Running

```bash
tools/serve_models.sh --models-dir models
PYTHONPATH=src:. python -m ragkit.cli --config recipes/translation/config \
    -c recipes/translation/data/heldout.jsonl -j work/translation.jsonl \
    --set source_language=English --set target_language=Polish
```

For a **different-script** pair (Japanese/Chinese/Russian → English), swap the `[[validator]]` in
`recipe.toml` from `EchoValidator` to `ScriptValidator` with the source script, e.g.
`source_script = "japanese"`.

## Evaluation

`recipes/eval_output.py` (shared across recipes) scores the produced translations against
`gold.jsonl` — trigram consistency and rejection rate — and needs the servers, so it is not part of
`tools/run_tests.sh`.
