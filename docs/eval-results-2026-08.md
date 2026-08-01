# Full-scale evaluation results (2026-08-01)

Results from running both recipes' evaluations at full scale (not the smaller controlled subsets
the READMEs' "Measured baseline" sections describe), plus a comparison against each recipe's
external reference dataset paper where the comparison is valid.

## `med_evidence` vs. PubMedQA (Jin et al., 2019, EMNLP)

`reader_eval.py`'s setup — the gold abstract handed to the model, yes/no/maybe decision — matches
the paper's task exactly, so this comparison is apples-to-apples.

| | Accuracy |
|---|---:|
| Majority-class baseline (paper) | 55.2% |
| Best fine-tuned model — BioBERT + long-answer bag-of-words (paper, supervised) | 68.1% |
| **Ours — Qwen3-14B, zero-shot, `reader_eval.py`, 1000 questions** | **68.0%** (answered_accuracy 71.6%) |
| Human performance (paper) | 78.0% |

Qwen3-14B, used zero-shot with no PubMedQA-specific training, lands within 0.1 point of the
paper's best *fine-tuned* baseline, and well clear of majority-class, while still ~10 points under
human performance.

`eval.py` (the harder retrieve-then-decide RAG task, not directly comparable to the paper — see
below) at full scale (1000 questions, 597k-doc corpus): accuracy 0.528, abstain rate 0.078,
answered_accuracy 0.645 (897/1000 decided). This is a materially larger, more reliable run than the
README's existing 60-question baseline (0.417 / 0.490), though not a controlled re-run of the same
sample size.

Caveat: `eval.py`'s number is **not** comparable to the PubMedQA leaderboard — it makes the model
retrieve the answer-bearing abstract among ~600k distractors *before* deciding, strictly harder than
the paper's task, which hands the gold abstract to the model (that's what `reader_eval.py` isolates
instead).

## `legal_procurement` vs. PolQA (Rybak et al., 2022/2024, LREC-COLING)

Corpus size matches: the paper reports 7,097,322 candidate passages; our table has 7,097,288 (34
fewer — minor ingest-side filtering, not concerning).

**Metric compatibility matters here.** This framework's own `Recall@20`/`MRR@10`/`NDCG@10` use a
different statistic and depth than the paper's reported "top-10 accuracy" (a binary per-query
hit/miss: did *any* gold passage land in the top 10). The two are not directly comparable. To get a
literature-comparable number, this session added `ragkit.eval.retrieval.hit_rate_at_k`, wired into
`legal_procurement/eval.py` as **Acc@10** — same statistic, same depth as the paper.

Full scale: 956 heldout questions, full 7.1M-passage corpus.

| Retriever | Training | Recall@20 | MRR@10 | NDCG@10 | Acc@10 (paper-comparable) |
|---|---|---:|---:|---:|---:|
| Paper: HerBERT, *Standard* strategy | supervised, fine-tuned on PolQA's own train set | — | — | — | 51.47% |
| Paper: HerBERT, *Efficient* strategy (best) | supervised, fine-tuned on PolQA's own train set | — | — | — | **62.02%** |
| lexical BM25 (README's own baseline) | zero-shot | 0.332 | 0.393 | 0.250 | *(not rerun with Acc@10)* |
| **dense** (Qwen3-Embedding-0.6B) | zero-shot, no PolQA training at all | 0.342 | 0.480 | 0.288 | **66.5%** |
| **hybrid + reranker** (bge-reranker-v2-m3) | zero-shot, no PolQA training at all | 0.511 | 0.750 | 0.503 | **86.0%** |

On the paper's own metric, our **zero-shot** dense retriever already beats the paper's best
**fine-tuned** retriever (66.5% vs. 62.0%), and hybrid+reranker clears it by 24 points (86.0%).
Notable, but scoped to one dataset/domain (Polish Wikipedia trivia questions) — not a universal
claim that zero-shot beats fine-tuning in general.

## Bugs found and fixed along the way

- **Missing LanceDB ANN index.** `legal_procurement`'s 7.1M-row vector table had no index, so
  `search()` silently brute-force scanned every row on every query — a 956-query dense eval hung
  for 2+ hours at 0% GPU / 453% CPU with zero progress. Fixed: `LanceVectorIndex.create_index()`
  (IVF_FLAT) and `tools/build_vector_index.sh`. Building it on the real table took ~3 minutes and
  left the row count exactly unchanged; the same eval then finished in ~3 minutes. Generic to the
  framework, not specific to this recipe — any dense/hybrid retrieval at scale would hit this.
- **Confirmed data corruption from an earlier concurrent-writer incident** (5,000 duplicate rows in
  the same table), found via an exact `pairings.count() == vector.count()` audit, fixed by deleting
  and re-embedding the affected ids. See `.audit/` and prior session memory for the full incident.
- **`tools/fetch_models.sh`'s shared `producer`/`reviewer` model entries point at HuggingFace repos
  that don't exist.** Only the smaller `ci` entry was fixed so far; `producer`/`reviewer` themselves
  remain broken and affect every recipe using the shared "local" chat defaults — a follow-up, not
  done as part of this evaluation.
