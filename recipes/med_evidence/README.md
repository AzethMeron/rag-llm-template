# Recipe: biomedical evidence intelligence

Decide from **memory + a question**. The memory is a corpus of biomedical abstracts (retrieved per
question); the question is a yes/no/maybe research question. The recipe produces a grounded
decision — `decision` ∈ {`yes`, `no`, `maybe`, `unsupported`}, an array of **verbatim evidence
quotes copied from the retrieved abstracts**, and a one-sentence rationale — and refuses any answer
whose evidence is missing (for a non-abstaining decision) or not actually present in the retrieved
abstracts. When the abstracts do not address the question, it must answer `unsupported` (abstain)
rather than guess.

## What it demonstrates

- **Retrieval memory driving a structured decision.** A `retrieved` block pulls the relevant
  abstract excerpts (the memory) into the prompt; the answer's evidence quotes must come from those
  excerpts, and the grounding validator re-checks every quote against them.
- **A grounded structured decision.** The output is a `FormSchema` with an `array` evidence field.
- **Two mechanical grounding guards** (`plugins/validators.py`):
  - `GroundedEvidenceValidator` — the "abstain unless there is positive evidence" rule made
    mechanical: every quote must appear (after whitespace/quote normalisation) in the abstracts
    actually retrieved for that question, and a non-abstaining decision with no evidence at all is
    refused. A hallucinated quote, or an uncited yes/no/maybe, is a blocking violation and a
    `REJECTED` record. If no memory is wired, the validator blocks rather than passing silently.
  - `DecisionEnumValidator` — the `decision` must be one of `yes`/`no`/`maybe`/`unsupported`, so the
    eval and any automation act on a known category, never free text. Malformed JSON blocks too.
- **Safety by construction.** `[[forbidden]]` patterns reject **prescriptive medical advice**
  (imperative dosing/prescribing such as "you should take …", "take 20 mg …", "stop taking …", "we
  recommend that you …") in code; `[[advisory]]` criteria (`grounded_evidence`,
  `calibrated_decision`, `abstain_when_unsupported`, `no_medical_advice`) are judged by the
  `from_rules` compliance reviewer. The system reports what the evidence says; it never instructs.

## Data (fetched, never committed)

```bash
recipes/med_evidence/fetch.sh [--limit N] [--trials N]
```

Two real sources:

- **Gold eval set + base corpus — PubMedQA** (`ori_pqal.json`, **MIT-licensed**). A JSON dict keyed
  by PMID; each entry has a `QUESTION`, an abstract (`CONTEXTS`), and a `final_decision`
  (yes/no/maybe — the **gold**, kept out of every prompt). `fetch.sh` writes `data/abstracts.jsonl`
  (the retrieval corpus, `{id, text}` per PMID), `data/heldout.jsonl` (the questions to answer), and
  `data/gold.jsonl` (`{record_id, decision}` — the gold, written **only** here). `--limit N` caps
  the number of questions (default `0` = all, ~1000).
- **Corpus scaling — ClinicalTrials.gov v2** (US-Gov **public domain**). `--trials N` (default `0` =
  skip) appends `N` real study summaries (`NCT…` id, brief title + brief summary + conditions) to
  `data/abstracts.jsonl`, paginating the API via its `nextPageToken`. This lets the retrieval corpus
  scale to **many GB** of real trial text while the PubMedQA questions and gold stay fixed — a
  realistic "small labelled eval set, huge unlabelled memory" setup.

Tiny in-test fixtures back the recipe's own tests, so they need no download or network; those tests
also retrieve the abstracts memory over **both real vector indexes** (LanceDB and Qdrant), swapped by
a one-line `storage.toml` driver edit, alongside the default `fts5` lexical path.

## Running

```bash
tools/serve_models.sh --config recipes/med_evidence/config/models.toml \
    --endpoint local --models-dir models
PYTHONPATH=src:. python -m ragkit.cli --config recipes/med_evidence/config \
    -c recipes/med_evidence/data/heldout.jsonl -j work/med.jsonl
```

## Evaluation

`recipes/med_evidence/eval.py` scores the produced `decision` against the gold PubMedQA final
decision. It reads only the journal and gold, and reports three numbers: **accuracy** (produced
decision equals gold; a miss or an abstention counts as wrong), **abstain_rate** (fraction answered
`unsupported`), and **answered_accuracy** (accuracy over only the questions it actually decided).

```bash
PYTHONPATH=src:. python -m recipes.med_evidence.eval \
    --journal work/med.jsonl --gold recipes/med_evidence/data/gold.jsonl
```
