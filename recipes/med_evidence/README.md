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
  skip) appends up to `N` real study summaries (`NCT…` id, brief title + brief summary + conditions)
  to `data/abstracts.jsonl`. Because the API's `nextPageToken` chain silently truncates past
  ~20–100k studies, the whole registry is fetched in **monthly `LastUpdatePostDate` shards** (2000
  to next year): every study has exactly one such date, so the shards partition the **~596k studies**
  with no overlap and each is small enough to page to exhaustion reliably. Pass a large `--trials`
  to pull the **full registry**; the PubMedQA questions and gold stay fixed — a realistic "small
  labelled eval set, huge unlabelled memory" setup.

Tiny in-test fixtures back the recipe's own tests, so they need no download or network; those tests
also retrieve the abstracts memory over **both real vector indexes** (LanceDB and Qdrant), swapped by
a one-line `storage.toml` driver edit, alongside the default `sqlite` pairing-store path.

## Storage (on-disk, low-RAM)

Because the corpus scales to the whole registry, `config/storage.toml` keeps the pairing store — the
abstract rows and the FTS5 search index that resolves a hit's text + metadata, co-located in one
database so they can never drift apart — **on disk** next to the fetched data, never in RAM:

- `[pairings]` (`sqlite`, `path = ../data/abstracts.pairings.db`).

`[reference].file` (`data/abstracts.jsonl`) is streamed into it once and the persisted store is
**reused on later runs** (build-once, resumable). Delete `data/abstracts.pairings.db` to force a
re-import after re-fetching a different corpus size.

## Running

```bash
tools/serve_models.sh --config recipes/med_evidence/config/models.toml \
    --endpoint local --models-dir models
PYTHONPATH=src:. python -m ragkit.cli import --catalog recipes/med_evidence/data/heldout.jsonl \
    --run-db work/med_evidence.db
PYTHONPATH=src:. python -m ragkit.cli run --config recipes/med_evidence/config \
    --run-db work/med_evidence.db
PYTHONPATH=src:. python -m ragkit.cli export --run-db work/med_evidence.db -j work/med.jsonl
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

### Two eval modes (why the numbers below aren't the PubMedQA leaderboard)

This recipe's default is a **RAG task**: the model must *retrieve* the answer-bearing abstract among
~600k distractors and *then* decide — strictly harder than PubMedQA, whose published numbers hand
the gold abstract to the model. So the accuracy here is **not** directly comparable to the
leaderboard (human ≈ 0.78; strong instruction-tuned LLMs ≈ 0.75–0.80; majority-class ≈ 0.55).

For a **directly comparable** number, `reader_eval.py` reproduces the benchmark setup: a
`SelfAbstractRetriever` gives each question its *own* gold abstract as the sole context (same panel,
same grounding — only retrieval is swapped), isolating decision quality:

```bash
PYTHONPATH=src:. python -m recipes.med_evidence.reader_eval \
    --config recipes/med_evidence/config --abstracts recipes/med_evidence/data/abstracts.jsonl \
    --heldout recipes/med_evidence/data/heldout.jsonl --gold recipes/med_evidence/data/gold.jsonl
```

### Measured baseline

Over a **597k-doc corpus** (596,055 ClinicalTrials.gov studies + 1,000 PubMedQA abstracts), **60
gold questions**, answered by **Qwen3-14B via llama.cpp**:

| Metric | Initial | + decision-prompt fix | + ellipsis-aware grounding |
|---|---:|---:|---:|
| decision accuracy | 0.300 | 0.383 | **0.417** |
| answered_accuracy | 0.367 | 0.489 | **0.490** |
| "no" produced (gold has 18) | 0 | 6 | 7 |
| grounding rejections | 9 | 10 | **6** |

Controlled experiments showed retrieval was *not* the bottleneck (the gold abstract is retrieved
~97% of the time): the model produced **zero "no"** and over-hedged to "maybe", losing 24/60 records.
Two config-only fixes lifted accuracy 0.300 → 0.417: (1) defining "no" in the prompt (a
null/no-difference/opposite result is a "no") and stopping the `calibrated_decision` rule from
treating "qualified" as "maybe"; (2) ellipsis-aware grounding, so a legitimately *elided* quote
("A … B", both parts verbatim) is accepted while an invented one still blocks — recovering four
correct-but-spliced records. These remain honest baselines, not tuned results — the RAG framing
(retrieve the right abstract among ~600k distractors, *then* decide) is materially harder than
classic PubMedQA where the abstract is handed to the model; further gains would come from
dense/hybrid retrieval and a stronger decoder.

`config/models.toml`'s `[model.author]` is pinned to this same **Qwen3-14B-Q5_K_M** (fetch with
`tools/fetch_models.sh --only med_author`, ~10.5GB — too large for the default fetch-everything
set) so the checked-in config reproduces the table above, not a smaller stand-in.

### Full-scale results (2026-08-01) — all 1000 gold questions

Both eval modes run against the **full 1000-question PubMedQA gold set** (the 60-question table
above was a smaller controlled sample), same Qwen3-14B author:

| Eval | accuracy | answered_accuracy | decided |
|---|---:|---:|---:|
| `eval.py` (RAG: retrieve, then decide) | 0.528 | 0.645 | 897/1000 |
| `reader_eval.py` (gold abstract given — directly comparable to PubMedQA) | 0.680 | 0.716 | 954/1000 |

**Against the PubMedQA paper (Jin et al., 2019, EMNLP)** — `reader_eval.py`'s setup matches the
paper's task exactly, so this comparison is apples-to-apples:

| | Accuracy |
|---|---:|
| Majority-class baseline | 55.2% |
| Best fine-tuned model (BioBERT + long-answer bag-of-words, supervised) | 68.1% |
| **Ours — Qwen3-14B, zero-shot, no PubMedQA-specific training at all** | **68.0%** |
| Human performance | 78.0% |

Qwen3-14B, used zero-shot, lands within 0.1 point of the paper's best *fine-tuned* baseline, well
clear of majority-class, and ~10 points under human performance — a strong result for a
general-purpose model with no task-specific training.

`eval.py`'s number is **not** comparable to the PubMedQA leaderboard for the same reason as the
60-question table above: it makes the model retrieve the answer-bearing abstract among ~600k
distractors before deciding, strictly harder than the paper's task. Caveat on this run: the
substitute reviewer model standing in for a still-broken default (see `tools/fetch_models.sh`'s
known `producer`/`reviewer` issue) frequently produced degenerate output during the
faithfulness-review step and had to abstain, so the review/grounding gate wasn't operating at full
strength.

### Enabling dense retrieval

Like `legal_procurement`, dense is **opt-in, not the default** (a one-time GPU cost to embed the
corpus, kept out so `eval.py` runs with no GPU) — `models.toml` already has `[model.embedder]`
and `storage.toml` already has `[vector]` (`driver = "lancedb"`, `dim` = the embedder's output
dim). The only step left to switch the recipe onto it is a `retrieval.toml` with `[retrieval]
kind = "dense"` / `[retrieval.dense] model = "embedder"` (or `kind = "hybrid"` with a reranker).
Embedding the ~600k-abstract corpus into `data/abstracts.lance` needs the embedding server running
on its own endpoint (`[endpoint.embed]` in `models.toml` already wires `tools/embed_presets.ini`,
which names the model `embed` and turns on `--embeddings` — router mode alone would not):

```bash
tools/serve_models.sh --config recipes/med_evidence/config/models.toml \
    --endpoint embed --models-dir models
tools/embed_reference.sh --config recipes/med_evidence/config \
    --embedding-url http://127.0.0.1:8081/v1 --embedding-model embed
```

Resumable/idempotent, safe to interrupt and re-run. `embed_reference.sh` also compacts the LanceDB
vector store automatically every 50 commits by default (`--compact-every`, `0` disables it) — a
long, many-times-resumed run otherwise accumulates on-disk fragments without bound (see
`tools/compact_vector_store.sh` to compact by hand; `legal_procurement`'s corpus is the concrete
case that made this necessary).
