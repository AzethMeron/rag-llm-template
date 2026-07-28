# Recipe: translation

A **faithful port of [`AzethMeron/llm-translator`](https://github.com/AzethMeron/llm-translator)**
onto this framework — the same pipeline, prompts, review panel, rules and context, expressed in the
framework's config rather than translation-specific Python. It is the proof that the framework
*generalised* llm-translator without losing it: `config/personas.toml` + `config/rules.toml` +
`config/context.toml` reproduce llm-translator's `agents.toml` + `translation_rules.toml` +
`[context]`, and the harness builds the same `produce → mechanical check → review panel → revise`
loop.

## Faithful to llm-translator

- **The producer prompt** is llm-translator's translate instructions verbatim (meaning-not-words,
  the "who does what to whom" section, natural-target guidance), with `{source_language}` /
  `{target_language}` filled from `--set`.
- **The full five-reviewer panel, in order:** `accuracy → structure → grammar → fluency →
  compliance` (the last built `from_rules`), consulted in file order, stopping at the first
  objection — exactly llm-translator's panel.
- **The rules:** all six forbidden failure-mode patterns (preamble, refusal/commentary, label,
  translator's note, alternative rendering, collapsed placeholders) and all seven advisory criteria
  (register, voice consistency, continuity, proper nouns, pronouns, figurative language,
  localization), plus the line-width limit and the untranslated-echo mechanical check.
- **The context building:** the glossary (lexicon), the retrieved translation-memory examples, the
  surrounding neighbours (3 before / 2 after) and their established translations, and the previous
  attempt on revision — llm-translator's `[context]` knobs mapped onto the framework's blocks.

## What it also demonstrates about the framework

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

`recipes/translation/eval.py` scores the produced translations against `gold.jsonl` two ways: an
**exact match** rate (a strict lower bound — one gold reference rarely equals an equally-valid
alternative) and a **trigram consistency** to the reference (which credits correct but
differently-phrased output). Read them together: high consistency with low exact means valid, varied
wording. It reads only the journal and gold.

```bash
PYTHONPATH=src:. python -m recipes.translation.eval \
    --journal work/out.jsonl --gold recipes/translation/data/gold.jsonl
```
