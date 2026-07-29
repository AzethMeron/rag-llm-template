# Recipe: predictive-maintenance decision support

Decide from **memory + a request**. The memory is a corpus of equipment manuals (retrieved per
record); the request is an operator report plus **sensor readings and fault codes**. The recipe
produces a grounded decision — a diagnosis, a severity, a recommended action, and **verbatim
evidence quotes from the manuals** — and refuses any decision whose evidence is missing or not
actually in the retrieved manuals.

## What it demonstrates

- **Retrieval memory + a structured request together.** A `retrieved` block pulls the relevant
  manual passages (the memory), while a new `readings` block renders this record's sensor readings
  and fault codes straight from `record.meta` (the request) — two different context sources in one
  prompt.
- **A grounded structured decision.** The output is a `FormSchema` with an `array` evidence field
  (a small framework addition: forms can now hold string arrays).
- **The mandatory grounding guard** (`plugins/validators.py`, `GroundedDecisionValidator`), the
  "abstain unless there is positive evidence" rule made mechanical: every decision must cite ≥1
  manual quote, and every quote must appear in the passages actually retrieved for that record
  (re-checked through the same deterministic retriever the prompt used). A hallucinated citation, an
  uncited decision, an unknown severity, or a prompt-injected manual that a citation still cannot be
  found in — each is a blocking violation and a `REJECTED` record, never an accepted decision. If no
  memory is wired at all, the validator blocks rather than passing silently.

## Data (fetched, never committed)

```bash
recipes/predictive_maintenance/fetch.sh --cmapss <train_FD001.txt path or URL>
```

Two real sources:

- **Memory** — a manuals corpus built from Wikipedia articles on turbofan failure modes (EGT,
  compressor stall, bearing wear, vibration, ...), fetched via the MediaWiki API (text is
  CC BY-SA 4.0), written to `data/manuals.jsonl`.
- **Requests** — NASA **C-MAPSS** turbofan degradation data (US-Gov public domain): each held-out
  snapshot becomes a request with an operator report, sensor readings, and fault codes
  (`data/heldout.jsonl`), and a gold severity derived from the engine's remaining useful life
  (`data/gold.jsonl`).

The C-MAPSS download link is unstable, so pass `--cmapss` with a `train_FD001.txt` you have. Tiny
in-test fixtures back the recipe's own tests, so they need no download or network; those tests also
retrieve the manuals memory over **both real vector indexes** (LanceDB and Qdrant), swapped by a
one-line `storage.toml` driver edit, alongside the default `fts5` lexical path.

## Running

```bash
tools/serve_models.sh --config recipes/predictive_maintenance/config/models.toml \
    --endpoint local --models-dir models
PYTHONPATH=src:. python -m ragkit.cli --config recipes/predictive_maintenance/config \
    -c recipes/predictive_maintenance/data/heldout.jsonl -j work/pdm.jsonl
```

## Evaluation

`recipes/predictive_maintenance/eval.py` scores the decision's **severity** against the gold
remaining-useful-life bucket — both exact (3-class) and *actionable* (needs-attention vs normal,
forgiving a watch/urgent mix-up). It reads only the journal and gold.

```bash
PYTHONPATH=src:. python -m recipes.predictive_maintenance.eval \
    --journal work/pdm.jsonl --gold recipes/predictive_maintenance/data/gold.jsonl
```
