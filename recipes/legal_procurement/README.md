# Recipe: Polish legal / public-procurement passage retrieval

Answer a Polish legal or **public-procurement** question from a corpus of legal passages (the
memory). The recipe retrieves the most relevant passages, produces a short grounded answer that
**cites the passages it used by their id**, or **abstains** (`brak podstaw`) when nothing relevant is
retrieved — and refuses any answer that cites a passage it never actually retrieved.

This is the project's demonstration of the **retrieval-metrics eval layer**
(`src/ragkit/eval/retrieval.py`): its headline number is **retrieval quality against real gold
relevance judgments** — Recall@20, MRR@10, NDCG@10 — not a model-output score.

## The use case

Public-procurement and legal-compliance work is retrieval-first: an answer is only worth as much as
the provisions it rests on. Here a question ("in what mode does the contracting authority award a
contract?", "what is the deadline for an appeal to the KIO?") is answered strictly from the retrieved
passages, each claim tied to a cited passage id, so a reviewer can check the source rather than trust
the model. An unsupported question is answered `brak podstaw` rather than guessed.

## What it demonstrates

- **Retrieval quality as the headline metric.** `eval.py` builds the recipe's own retriever
  (`assemble(config).retriever`) and scores its ranking of the gold-relevant passages per question
  with the framework's per-query `recall_at_k` / `reciprocal_rank` / `ndcg_at_k`. Gold comes from
  polqa (an independent source), so no metric is true by construction.
- **A grounded, citing answer.** The output is a `FormSchema` with a string `answer` and an `array`
  `citations` field.
- **The citation-grounding guard** (`plugins/validators.py`, `CitationGroundingValidator`): unless
  the answer abstains (`brak podstaw`), it must cite ≥1 passage id, and **every** cited id must be
  among the passages actually retrieved for that question (re-checked through the same deterministic
  retriever the prompt used). A fabricated citation, an uncited substantive answer, or malformed
  output is a blocking violation and a `REJECTED` record. If no memory is wired at all, the validator
  blocks rather than passing silently.

### One id space

The framework's corpus loader gives every passage a stable chunk id `ref-<line>`. `fetch.sh` writes
`data/passages.jsonl` with exactly that id, and the gold in `data/gold.jsonl` is expressed in the
same ids — so a citation, a gold judgment, and a retrieval hit all refer to the same thing, and
Recall is directly computable.

## Data (fetched, never committed)

```bash
recipes/legal_procurement/fetch.sh [--max-passages N] [--limit N]
```

- **Memory** — the **polqa** passage corpus (IPIPAN, **CC BY-SA**): Polish Wikipedia passages used as
  a stand-in legal/encyclopaedic corpus. The full `passages.jsonl` is **~3.3 GB / ~7.1M passages**.
  `--max-passages` (default **0 = load the whole corpus**) bounds how many of its first lines are
  loaded; it streams the file line by line, so the corpus is never held whole in memory. Loading all
  passages guarantees every question's gold passage is present, so all questions are scorable. Each
  kept passage is written to `data/passages.jsonl` as `{id, text}`.
- **Questions + gold** — polqa's `test.csv` (columns:
  `question_id, passage_title, passage_text, passage_wiki, passage_id, duplicate, question, relevant,
  answers, ...`). Rows are grouped per question into `data/heldout.jsonl` (`{record_id, source,
  meta}`) and `data/gold.jsonl` (`{record_id, relevant:[passage ids]}`), keeping only questions whose
  gold passage is inside the loaded subset (so Recall is computable). `--limit` caps how many
  questions are kept; the count kept is reported.

### A real legal corpus instead

polqa is an encyclopaedic stand-in. For genuine legal text, swap the memory for **EUR-Lex**
(EU law, https://eur-lex.europa.eu) or **ELI-Sejm** (Polish legislation via the ELI API,
`https://api.sejm.gov.pl/eli`) documents in the same `{id, text}` JSONL shape — the recipe is
corpus-agnostic; only `data/passages.jsonl` and the gold change.

Tiny in-test fixtures back the recipe's own tests, so they need no download or network; those tests
also retrieve the memory over **both real vector indexes** (LanceDB and Qdrant), swapped by a
one-line `storage.toml` driver edit, alongside the default `sqlite` pairing-store path.

## Storage (on-disk, low-RAM)

The corpus is millions of passages / gigabytes of text, so `config/storage.toml` keeps the pairing
store — the passage rows and the FTS5 search index that resolves a hit's text + metadata, co-located
in one database so they can never drift apart — **on disk** next to the fetched data, never in RAM:

- `[pairings]` (`sqlite`, `path = ../data/passages.pairings.db`).

`[reference].file` (`data/passages.jsonl`) is streamed into it once and the persisted store is
**reused on later runs** (build-once, resumable). Delete `data/passages.pairings.db` to force a
re-import after re-fetching a different corpus size.

## Running

```bash
tools/serve_models.sh --config recipes/legal_procurement/config/models.toml \
    --endpoint local --models-dir models
PYTHONPATH=src:. python -m ragkit.cli import --catalog recipes/legal_procurement/data/heldout.jsonl \
    --run-db work/legal_procurement.db
PYTHONPATH=src:. python -m ragkit.cli run --config recipes/legal_procurement/config \
    --run-db work/legal_procurement.db
PYTHONPATH=src:. python -m ragkit.cli export --run-db work/legal_procurement.db -j work/legal.jsonl
```

## Evaluation

`recipes/legal_procurement/eval.py` scores **retrieval quality** — Recall@20, MRR@10, NDCG@10 — by
retrieving for each held-out question and comparing the ranked passage ids to the gold. Given a run
`--journal`, it additionally reports the **citation-grounding rate** (the fraction of produced
answers whose citations all fall within the question's gold or retrieved passages).

```bash
PYTHONPATH=src:. python -m recipes.legal_procurement.eval \
    --config recipes/legal_procurement/config \
    --heldout recipes/legal_procurement/data/heldout.jsonl \
    --gold recipes/legal_procurement/data/gold.jsonl \
    --k 20 [--journal work/legal.jsonl]
```

### Measured baseline

Over the **full 7.1M real passages**, **956 gold questions**, lexical **BM25** (no GPU):

| Metric | Value |
|---|---|
| Recall@20 | 0.332 |
| MRR@10 | 0.393 |
| NDCG@10 | 0.250 |

This is a **zero-shot lexical baseline** — no dense or hybrid retrieval, no reranking — so it is a
floor, not a ceiling. Controlled experiments confirm BM25 is near its ceiling here: the gold *is*
retrievable (Recall@20 ≈ 0.87 when the gold passages are guaranteed present) but is buried as the
corpus scales; changing the FTS5 tokenizer (`trigram`/`ascii`/`porter`) or stripping query stopwords
does **not** help (Polish is highly inflected and FTS5 has no Polish stemmer). **Dense (semantic)
retrieval is the real lever.** On an identical controlled 25k-passage set (100 questions):

| Retriever | Recall@20 | MRR@10 | NDCG@10 |
|---|---|---:|---:|
| lexical BM25 | 0.174 | 0.451 | 0.197 |
| **dense** (Qwen3-Embedding-0.6B) | **0.230** | 0.545 | 0.240 |
| hybrid (RRF) | 0.215 | **0.568** | **0.242** |

Dense lifts Recall +32% relative and ranks better (MRR/NDCG); hybrid ranks best. **To enable dense**
(a one-time GPU cost to embed the corpus — kept out of the default so `eval.py` runs with no GPU),
add an embedding model to `models.toml` (`[model.embedder]`, `kind = "embedding"`), a `[vector]`
store to `storage.toml` (`driver = "lancedb"`, `dim` = the embedder's output dim), and a
`retrieval.toml` with `[retrieval] kind = "dense"` / `[retrieval.dense] model = "embedder"`; for the
best ranking use `kind = "hybrid"` with a reranker (the framework warns against equal-weight hybrid
without one).

**Low-RAM at scale (the concrete evidence).** The full **7.1M-passage** polqa corpus was **ingested
and queried at ~37 MB process RSS**, because the search index and the chunk rows live on disk and a
hit resolves through the `[pairings]` store rather than a RAM map — the streaming, on-disk import
holds no corpus-sized structure in memory.
