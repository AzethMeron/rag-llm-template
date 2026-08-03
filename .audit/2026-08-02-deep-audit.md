# 2026-08-02 — Deep audit (whole codebase: bugs, OOM, contracts, docs, swappability)

**What was checked.** A full-codebase audit across seven subsystems — `store/`, `llm/`,
`harness/` orchestration, context-assembly + memory + `ingest/`, `retrieve/` + `eval/`, `core/` +
`cli/`, and docs + tool-scripts + recipes — hunting for semantic/logical bugs, OOM/performance
risks, concurrency/durability gaps, contract violations, docs/staleness drift, SSOT violations, and
whether the core interfaces (`VectorIndex`, `SqlStore`, `PairingStore`, `RunStore`, `Provider`,
`Backend`, …) are genuinely swappable.

**Method.** Seven parallel auditors each read their subsystem in full against `core/ports.py` and
`CLAUDE.md`. Every High-and-above finding below was then **independently re-verified** by reading the
cited code directly; the verification status is noted per finding. Severities are calibrated here
(reachability × blast-radius × silent-vs-loud), and may differ from a sub-agent's raw label.

**Headline.** The core is well-built. The logic most likely to be subtly wrong came back **clean**:
retrieval-metric arithmetic (incl. `hit_rate_at_k`/Acc@10), fusion/MMR/RRF sign conventions, the
produce→review→revise loop (degenerate reviewer replies degrade to `PRODUCED`, never
"abstain-as-pass"), `RunStore` durability, capture fidelity, config type-readers, and record
atomic-writes. Tool scripts are well-hardened; docs mostly match code (the retired legacy-storage
path is correctly documented as retired). The real problems cluster in **(a)** setup being broken
for 5/6 recipes, **(b)** a few silent-misconfiguration / silent-degradation paths, and **(c)**
swappability enforced only on the happy path.

No files were edited as part of this audit.

---

## Resolution (2026-08-02, same day)

Every finding below has been worked and carries its own **Status** line, in place. Nothing was
deleted: where the fix differs from what was suggested, or where a suggestion turned out to be
wrong, the original text stands and the status says why (this directory's convention — a wrong
conclusion kept on the page, struck through, is more useful than a deleted one).

**41 of 41 findings addressed** (1 critical, 3 high, 16 medium, 20 low). Of those, 33 were fixed
as suggested and 8 were fixed differently, each with the reasoning recorded:

| Finding | Departed from the suggestion how |
|---|---|
| C1 | The audit's own replacement filename does not exist either; the real repos were verified against the live HF API before use. |
| H1 | Per-arm floors **refused** rather than plumbed — plumbing would have silently activated the 0.30/0.55 defaults on stacks that currently floor at 0.0. |
| M2 | `nprobes` defaults to a *fraction* of the partitions, not a fixed count; `refine_factor` deliberately not added (IVF_FLAT stores exact vectors, so refinement buys nothing). |
| M3 | Refused at the boundary rather than stored as columns — LanceDB needs a fixed schema and offers no JSON-path filter (both checked directly). |
| M5 | Rebuild *deferred to the next search* rather than an explicit `finalize()`, so `search` can never return a stale index. |
| M13 | Documented rather than changing the chunker — paragraph structure is what it chunks on. |
| M14 | The flagged `sorted()` copy is pointers, not strings; the real 2× peak was one level down, inside `reconcile`, and that is what was fixed. |
| L5 | Constant **deleted** rather than wired into a knob — a rerank floor would duplicate the block's own `min_score`. |

**Findings the fixes turned up that this audit missed** — all in the corners the swappability
verdict predicted, and all found by the widened conformance suite:

- Qdrant could not filter on `id` at all: it keeps the chunk id in the payload under `_cid`, so
  `Predicate("id", EQ, ...)` matched nothing and every row it should have excluded came back.
- The in-memory conformance double ignored `where` outright, returning everything for a filtered
  query — the one thing the port forbids.
- The same double returned raw overlap counts (`2.0`) as relevance scores, because `SearchIndex`
  never actually stated the `[0, 1)` bound. That omission is the deeper reason M16's `min_score`
  was not portable.
- Every embedding fake in the test suite omitted the `index` field, which is why a suite that
  covers `embedding.py` thoroughly still could not have caught M8.

**Found by the full-scale validation runs, not by reading — recorded, not fixed:**

- **`SqlitePairings` serialises reads, so a retrieval-bound run ignores `--concurrency`.** One
  connection shared by every thread under one lock. WAL means a reader is never blocked by a
  *writer*, but readers block each other, so the concurrency WAL would allow never reaches the
  caller. Measured on `legal_procurement` (7.1M rows): one BM25 query ~4.6 s, throughput pinned at
  ~2.3 records/min with `--concurrency 4` on a 24-core machine, GPU at 0% — every worker queued
  behind the lock, not the model. Scale-dependent: `med_evidence` (597k rows) was model-bound in
  the same run shape and scaled with concurrency (2 → 5.7 records/min going from 2 to 4 workers).
  Written up in the driver's own docstring with the fix sketched (thread-local *read* connections,
  lock retained for writes, `:memory:` excluded since each connection would get its own empty
  database). **Deliberately not attempted here**: it is a performance limitation rather than a
  correctness defect, and rewriting connection management in the most load-bearing store at the
  end of a long session — without room to test it to the standard everything else here was held
  to — is the wrong trade. It wants its own change with its own concurrency tests.

**Left undone, deliberately:** `license/THIRD-PARTY.md` still lists the two retired Qwen3.5
models. `CLAUDE.md` forbids adding, removing, or modifying anything under `license/` without
explicit permission, so this is flagged for the owner rather than edited.

Verification after the work: **1,373 tests, 100% statement and branch coverage on `src/`,
`tools/lint.sh` clean.** Every fix carries a regression test; several tests were rewritten because
they had been asserting the buggy behaviour (`tests/cli/test_app.py`'s lexical-stack test set a
`min_score` that went nowhere; two runner tests stood on the scripted queue's `AssertionError` —
exactly the exception H2 now contains — to simulate a dead server).

---

## CRITICAL

### C1 — `fetch_models.sh` `producer`/`reviewer` point at nonexistent HF repos
- **Status:** **FIXED.** `producer`/`reviewer` now name `Qwen/Qwen3-4B-GGUF` (`Qwen3-4B-Q4_K_M.gguf`) and `Qwen/Qwen3-1.7B-GGUF` (`Qwen3-1.7B-Q8_0.gguf`), each verified against the live HF API (repo reachable, exact filename in its file list, apache-2.0, un-gated) and then actually downloaded — the audit's *own* suggested replacement, `Qwen3-1.7B-Q4_K_M.gguf`, does not exist either; that repo ships only Q8_0. The caveat above is now resolved: network was available, and both original repos do return 401. Five recipes' `models.toml`, `docs/tutorial.md` and translation's `approx_vram_mb` updated; med_evidence keeps its smaller 0.6B reviewer, now documented as a deliberate VRAM/reproducibility choice rather than a stopgap. New `tests/test_model_pins.py` relates the two files that drifted. **Not fixed:** `license/THIRD-PARTY.md` still lists the two retired models — CLAUDE.md forbids touching `license/` without explicit permission.
- **Where:** `tools/fetch_models.sh:33-34` (`Qwen/Qwen3.5-2B-Instruct-GGUF`, `Qwen/Qwen3.5-0.8B-Instruct-GGUF`).
- **Defect:** there is no Qwen3.5 GGUF family at those sizes; the repos 404/401. *Verified directly:* those two files are the only ones **absent** from `models/` (ci/embed/rerank/med_author are all present), and `grep` shows every recipe **except** `med_evidence` uses those ids as live `model_id`s in `config/models.toml` (`legal_procurement`, `nl_to_sql`, `form_autofill`, `predictive_maintenance`, `translation`).
- **Failure scenario:** a bare `tools/fetch_models.sh` `die`s on the first failed download, so the documented one-command setup (`README.md:45`, `docs/tutorial.md:361`, `tools/README.md`, every recipe's "Running" section) cannot complete — the self-contained-setup guarantee is broken. Retrieval-only evals still run (no chat model built); every chat pipeline for those 5 recipes is unrunnable.
- **Suggested fix:** repoint `producer`/`reviewer` at real repos+files (e.g. `Qwen/Qwen3-1.7B-GGUF` → `Qwen3-1.7B-Q4_K_M.gguf`, `Qwen/Qwen3-0.6B-GGUF` → `Qwen3-0.6B-Q8_0.gguf`), then update the 5 recipes' `models.toml` `model_id`s and `docs/tutorial.md:89,95` to match — exactly the correction already applied to `med_evidence`. The script's hardening is fine; only its data is wrong.
- **Caveat:** HF non-existence is inferred from `models/` contents + the recipe's own comment (`med_evidence/config/models.toml`) + prior sessions; no network was available to hit the HF API here.

---

## HIGH

### H1 — Rerank and per-arm `min_score` silently dropped for `kind = "dense"` / `"lexical"`
- **Status:** **FIXED, by refusing rather than plumbing.** `RetrievalSettings.__post_init__` rejects `rerank_enabled` outside `kind="hybrid"` (and an embedding model named under a kind that never embeds); `load_retrieval` additionally refuses any *present* key the chosen kind would ignore (`_APPLICABILITY`), which covers `candidate_pool` and `mmr_lambda` too. On the per-arm floors the audit offered plumbing or refusing: plumbing would have silently activated the 0.30/0.55 defaults for every single-arm stack that currently floors at 0.0, trading a silently-dropped knob for a silent recall change, so this refuses and points at `context.toml`'s block `min_score` — which already is the single-arm floor.
- **Where:** `src/ragkit/cli/app.py:239-242`; missing cross-check in `src/ragkit/retrieve/tuning.py:47-62`.
- **Defect (verified):** the `lexical`/`dense` branches `return` before the reranker is built (app.py:243 is hybrid-only), and `RetrievalSettings.__post_init__` validates `rerank_model` is set when `rerank_enabled` but **never** that `rerank_enabled` requires `kind=="hybrid"`. The single-arm builders are also called with no arguments, so `lexical_min_score`/`dense_min_score` are dropped too.
- **Failure scenario:** a `retrieval.toml` with `kind="dense"` + `[retrieval.rerank] enabled=true, model="reranker"` validates, assembles, and runs **with no reranking and no warning** — a silent fallback hiding a misconfiguration (forbidden by CLAUDE.md). Adjacent to the hybrid `retrieval.toml` shipped for legal_procurement: a user copying it and switching to dense loses reranking silently.
- **Suggested fix:** in `RetrievalSettings.__post_init__`, raise `ValueError` (→ `ConfigError`) when `rerank_enabled and kind != "hybrid"`; and pass `dense_min_score`/`lexical_min_score` into the single-arm builders (or reject the fields as hybrid-only). Rejecting is the smaller, clearer change.

### H2 — A plugin/validator/schema exception is a permanent batch-killing poison pill
- **Status:** **FIXED as suggested.** `process()` converts any non-`LlmError` exception into a REJECTED `Outcome` carrying the exception type, message and innermost frame; a bare `LlmError` still propagates (server down still stops the run, record left PENDING), and `BaseException` is untouched. Each contained defect is logged with a full traceback, so containment is loud. Trade recorded in the docstring: a plugin bug hitting every record now journals every record REJECTED rather than stopping at the first.
- **Where:** `src/ragkit/harness/agents.py:385-390` (`process` catches only `LlmRefusalError`/`LlmContentError`); `src/ragkit/harness/runner.py:182-190,199-200`.
- **Defect (verified):** any other exception — a `re.error`/`KeyError`/`ValueError` in a pluggable `Validator.validate`, a bug in a custom `OutputSchema.extract`, or in an injected `sanitize` — propagates out of `process()`. In `run_batch` the worker future raises, the handler does `failure = failure or exc; continue`, which **skips `append_result` and `progress.record`**, then re-raises after the loop.
- **Failure scenario:** the record stays `PENDING` (**not** recorded as errored — violates CLAUDE.md "a record that errors must be recorded as errored, not dropped"), the whole run aborts, and on resume `store.pending()` yields the same record → same deterministic exception → aborts again. One buggy record makes a 10k-record batch permanently un-completable.
- **Suggested fix:** in `process()`, wrap the body so an unexpected (non-`LlmError`) exception becomes a structured `REJECTED` `Outcome` carrying the diagnostic, while genuine infra `LlmError` still propagates (so "server down" doesn't burn a record on a run that exits 0).

### H3 — Documented one-command setup cannot complete (consequence of C1)
- **Status:** **FIXED.** Resolved by C1, and the optional half was taken too: `fetch()` no longer dies on the first failure — every entry is attempted, failures are collected and reported together, and the script still exits non-zero. A failed `curl` also no longer leaves a truncated destination file that the next run would report as "already have".
- **Where:** `tools/fetch_models.sh:61,64` (`die` on first failure); instructed as working in `README.md:45`, `docs/tutorial.md:361`, `tools/README.md:15,54`, every recipe README.
- **Defect:** `fetch()` aborts the whole run at `producer`/`reviewer`, so even the valid models after them aren't guaranteed fetched in one pass. CLAUDE.md requires setup be a hardened script that actually succeeds, "never a manual, undocumented step."
- **Suggested fix:** resolved by C1's repo correction. Optionally make `fetch()` continue past a single failure and report a summary of what succeeded/failed, so one bad entry can't mask the rest.

---

## MEDIUM

### M1 — Pool client creation races under concurrency (found independently by 2 auditors)
- **Status:** **FIXED as suggested** (the lock option). One reentrant lock guards both `_clients` and `_http`; `close()` takes it too. Verified meaningful by removing it: 8 racing threads built 8 clients, and two distinct models on one endpoint opened it twice.
- **Where:** `src/ragkit/llm/pool.py:204-214` (`client_for`), `:216-223` (`_http_for`); guarantee overclaimed at `harness/agents.py:139-146`; reachability at `cli/main.py:166` (`--concurrency` **default 2**).
- **Defect (verified):** both methods do unguarded check-then-set on `self._clients` / `self._http`; `UsageStats` is locked but the lazy client construction is not, and nothing pre-warms the pool. Cold-start, N workers all call `client_for(producer.model)` at once.
- **Failure scenario:** several threads see the key absent, all build an `LlmClient` + `httpx.Client`; the overwritten ones are never closed (`close()` only closes what survives the dict) → leaked connection pools (RAII violation), plus split `UsageStats` (undercounted throughput). No output corruption; self-limits to the startup window per model — hence Medium, not High.
- **Suggested fix:** guard both dicts with a double-checked `threading.Lock`, or eagerly build every model in `models_in_use()` once (single-threaded) after `check_capacity`, before `run_batch` submits.

### M2 — LanceDB IVF search never sets `nprobes` → recall drop from the index added 2026-08-01
- **Status:** **FIXED, partly differently.** `search()` now sets `nprobes`, defaulting to a *fraction* of the partition count (5%, floored at 20) rather than a fixed number, so the recall side of the trade holds as a table grows; deliberately uncapped, since an upper bound would reintroduce the same cliff at the top end. Exposed as `[vector].nprobes`. The `sqrt(n)` heuristic now has one home (`_ivf_partitions`) shared by `create_index` and the default. **`refine_factor` deliberately not added:** this index is IVF_FLAT, which stores exact vectors, so refinement re-reads vectors that were never lossy — it belongs with IVF_PQ, which this driver deliberately does not use.
- **Where:** `src/ragkit/store/vector/lancedb.py:88` (search builder, no `.nprobes()`/`.refine_factor()`); `:154-159` (`num_partitions = sqrt(row_count)`).
- **Defect (verified):** before `create_index`, search is exact brute force; after, it's approximate IVF with ~sqrt(n) partitions (~2,650 for 7.1M rows) but the query still probes only LanceDB's small default fraction. Recall drops silently on exactly the large tables the index targets.
- **Evidence:** the 3-question sample Recall@20 dipped 0.370→0.333 after indexing — consistent with a modest approximation loss. An honest caveat on the dense/hybrid eval numbers reported 2026-08-01.
- **Suggested fix:** set `nprobes` (≈1–5% of `num_partitions`, e.g. 20–130) and optionally `refine_factor` on the search builder; expose as config; document the recall/latency trade-off in `create_index`.

### M3 — LanceDB `search(where=…)` cannot filter on metadata; Qdrant can (masked by a mis-named test)
- **Status:** **FIXED by refusing at the boundary** (the audit's second option). Storing metadata as real columns is not possible — which keys exist varies per record and LanceDB needs a fixed schema — and there is no JSON-path filter to reach into the `meta` string (checked directly: `json_extract`/`get_json_object` are unavailable). A metadata predicate is now refused with a structured error naming the qdrant driver, which does honour it; a mixed filter is refused whole rather than half-applied. The mis-named test is renamed, and a cross-driver conformance test was added — which immediately found **two more** divergences of the same family: Qdrant could not filter on `id` at all (it keeps the chunk id under `_cid`, so the predicate matched nothing), and the in-memory conformance double ignored `where` outright. Both fixed.
- **Where:** `src/ragkit/store/vector/lancedb.py:77` (meta stored as one opaque `json.dumps` string), `:88-91` (compiles `to_sql(where)` to a column predicate); contrast `store/vector/qdrant.py:84-99,144-155`; masking test `tests/store/test_lancedb.py:42-46`.
- **Defect (verified structurally):** a predicate on a metadata key compiles to a nonexistent column → LanceDB fails at query time; Qdrant stores each key as a payload field and honors it. The port even says a driver that can't honor a predicate should refuse it *at the boundary*, not at query time. `test_metadata_filter` only filters on the real `id` column, so it passes while metadata filtering is broken.
- **Failure scenario:** swap `driver = qdrant → lancedb` (or a future filtered retriever) and `search(where=(Predicate("document_id", EQ, "d1"),))` silently breaks. Latent today (the only caller passes no `where`).
- **Suggested fix:** store metadata as real LanceDB columns (or use a JSON-path filter it supports) so `to_sql` predicates resolve; OR reject any predicate whose field isn't `id` at the boundary. Add a conformance test that filters on a genuine metadata key across all vector drivers.

### M4 — LanceDB `upsert` is delete-then-add (torn-write window, no unique `id` constraint)
- **Status:** **FIXED as suggested.** Native `merge_insert`, so the torn-write window and the per-upsert fragment doubling are both gone. It also surfaced a divergence the happy-path suite could not see: `merge_insert` refuses a batch naming the same id twice, where Qdrant's deterministic point id let the last occurrence silently win. Both drivers now refuse, through a new shared `store/vector/common.py` (which also homes the duplicated `VectorIndexError` and length/dimension checks).
- **Where:** `src/ragkit/store/vector/lancedb.py:76-79`.
- **Defect (verified):** `upsert` = `self.delete(ids)` then `self._table.add(rows)` as two independent transactions; a crash between them removes old vectors without adding new. No structural uniqueness on `id` — precisely why the documented concurrent-writer incident could produce 5,000 duplicate rows undetected. `reconcile()` self-heals on a later run, so recoverable, but a real torn-write/durability gap and the source of the fragment growth `compact()` exists to clean.
- **Suggested fix:** use LanceDB's native `merge_insert` (atomic upsert) instead of delete+add; removes both the torn-write window and the per-upsert fragment doubling.

### M5 — DuckDB pairing store rebuilds its whole FTS index on every `add()` → O(n²) ingest
- **Status:** **FIXED, differently.** Rather than an explicit `finalize()` the ingest loop must remember to call — which would leave `search` able to return a stale index — the rebuild is *deferred*: a write marks the index stale and the next `search` rebuilds first. A bulk load then queries pays one rebuild instead of ~1,400, `search`'s contract is unchanged, and no caller has to know. Opening a store no longer rebuilds either, so a read-only user pays nothing.
- **Where:** `src/ragkit/store/pairings/duckdb.py:95-101` (`_rebuild_fts` = `PRAGMA create_fts_index(..., overwrite=1)`), called at `:124` inside every `add`; contrast SQLite's incremental triggers `store/pairings/sqlite.py:37-50`.
- **Defect (agent-traced):** each `add` re-indexes the whole table, so a batched load (≈1,400 batches at 5k for 7M rows) does Θ(N²/batch) work — a silent, config-triggered performance cliff versus SQLite's O(batch).
- **Suggested fix:** build the DuckDB FTS index once after the bulk load (an explicit `finalize()`/deferred-index step the ingest loop calls at the end), not per `add`.

### M6 — DuckDB pairing `add()` is not atomic — a partial batch can persist
- **Status:** **FIXED as suggested.** Explicit `BEGIN`/`COMMIT`/`ROLLBACK`; verified directly that DuckDB does roll back a failed `executemany` under one. Conformance now asserts all-or-nothing on both shipped drivers.
- **Where:** `src/ragkit/store/pairings/duckdb.py:103-127`; contract `core/ports.py:276-280`; atomic SQLite counterpart `store/pairings/sqlite.py:88-99`.
- **Defect:** DuckDB doesn't roll back a whole `executemany` (the driver's own docstring admits it); a mid-batch failure leaves a partial write, and the returned added-count reflects the partial result — violating `PairingStore.add`'s "in one transaction" promise that SQLite honors.
- **Suggested fix:** wrap the DuckDB `add` in explicit `BEGIN`/`COMMIT`/`ROLLBACK` so the batch is all-or-nothing.

### M7 — `_load_recipe` bypasses the config type-checkers (bool inversion; unstructured crash)
- **Status:** **FIXED as suggested.** Every recipe field reads through `read_string`/`read_bool`/`as_table`. L12 was folded in: `_Recipe` carries its own field defaults, read from one `_RECIPE_DEFAULTS` instance.
- **Where:** `src/ragkit/cli/app.py:156-167`.
- **Defect (verified):** uses raw `bool(...)`/`str(...)`/`dict(...)` instead of `read_string`/`read_bool`/`as_table`. `use_memory = "false"` → `bool("false")` → **True** (the exact silent inversion `read_bool` was built to refuse); a non-mapping `output_schema_options = 5` raises a bare `TypeError` from inside `dict()` that escapes `main()`'s `except RagkitError` as a full traceback.
- **Suggested fix:** read every recipe field through the existing `config.py` helpers (already used correctly by `tuning.py`).

### M8 — Embedding client trusts response array order, ignoring the `index` field
- **Status:** **FIXED, strictly.** Vectors are placed by their own `index`, and the indices must be exactly `range(n)` — a missing, repeated or out-of-range one is refused rather than papered over. Strictness is safe because it was checked rather than assumed: llama.cpp's `/v1/embeddings` emits `index` on every item (verified directly against Qwen3-Embedding-0.6B), as the OpenAI schema requires. Every embedding fake in the suite omitted the field, which is why thorough coverage of this code could not have caught the bug; they now emit it.
- **Where:** `src/ragkit/retrieve/embedding.py:130-140` (`[item["embedding"] for item in data]`); contract `core/ports.py:206` ("rows returned in input order").
- **Defect (verified):** only the count is checked; the OpenAI `/v1/embeddings` `index` field (present precisely because array order isn't guaranteed) is ignored. A reordering server/proxy silently pairs every vector with the wrong text → all downstream search subtly wrong, no error. Latent (llama.cpp/ollama/vLLM preserve order today).
- **Suggested fix:** sort `data` by `int(item["index"])` before extracting (or assert the indices equal `range(len(chunk))`).

### M9 — Hybrid double-squashes an already-normalized rerank score
- **Status:** **FIXED, in the driver.** Normalisation moved into `RerankClient` (matching how the vector drivers work), which returns `[0, 1]` relevance; `score_scale` (`"logit"` | `"unit"`) says which convention the endpoint speaks, since nothing in the response distinguishes them. Exposed as `[retrieval.rerank].score_scale`. L5's dead constant was deleted rather than wired up — see L5.
- **Where:** `src/ragkit/retrieve/hybrid.py:113-114` applies `sigmoid` to `rerank.py:68`'s verbatim return; docstring `rerank.py:1-8` claims Jina/Cohere compatibility.
- **Defect:** correct for an unbounded llama.cpp logit, but a Jina/Cohere `relevance_score` already in `[0,1]` gets mapped to `[0.5, 0.731]` — ranking preserved (sigmoid monotone) but every `[0,1]` floor becomes meaningless (`min_score=0.30` admits everything).
- **Suggested fix:** normalize inside `RerankClient` per the port's score convention, or add a config flag stating whether the endpoint returns a logit or an already-`[0,1]` score and skip `sigmoid` in the latter case.

### M10 — Cross-recipe SSOT: the `eval.py` scoring skeleton is copy-pasted across all 6 recipes
- **Status:** **FIXED as suggested, both halves.** Two shared modules: `ragkit.eval.gold` (a `RagkitError`-based `EvalError`, one path:line-aware row validator, three gold loaders, the journal join, the JSON field extractor, and the argparse + report tail) and `ragkit.eval.classify` (`Labelled`/`ClassificationReport` with a generic `rate()`). Recipe `eval.py` drops 976 → 676 lines, replaced by 212 lines of directly-tested shared code; six duplicated copies of the gold-loading *tests* collapse into one. `evaluate_retrieval` gained `hit_rate_k`/`rank_k`, which is what let legal_procurement stop forking the metric loop — and the fork's real cost was not duplication but that it bypassed the circularity and missing-gold guards entirely, in the one recipe that runs retrieval eval. translation and form_autofill keep their own `Report`: one scores text against a reference, the other several typed fields, so neither is a label classification.
- **Where:** `recipes/*/eval.py` — `EvalError`, `load_gold`, `_pairs`, the single-field extractor, and the `main()` argparse skeleton are near-verbatim (med_evidence vs predictive_maintenance differ only by the gold key name). `legal_procurement/eval.py:96-114` additionally re-implements the metric loop, **bypassing** the load-bearing circularity guard in `eval/retrieval.py:104-127`.
- **Defect:** the divergent required-key check is a latent drift risk; the guarded evaluator is routed around by the one recipe that runs retrieval eval. *(The 2026-08-01 Acc@10 change added `hit_rate_at_10` to the recipe's duplicated `Report` as well as the shared `evaluate_retrieval`, extending the duplication rather than fixing it.)*
- **Suggested fix:** add classification-scoring helpers to `ragkit.eval` (`load_label_gold(path, *, field)`, `join_journal_with_gold(journal, gold)`, a generic `ClassificationReport`); have each recipe supply only its field name + metric predicates. Route `legal_procurement` through a `Qrels`-based guarded evaluator (parameterise `evaluate_retrieval` for per-metric depths so recipes needn't fork it).

### M11 — Registry `isinstance` port-check is signature-blind while the docstring guarantees conformance
- **Status:** **FIXED, both halves.** The docstring now describes what is actually verified, and `create()` additionally checks that each port member accepts the port's **keyword-only** parameters — which is where the risk is, since the ports put every load-bearing option there and callers pass them by name. Positional names are deliberately not compared (renaming `query` to `q` breaks nothing), and `**kwargs` or a non-introspectable C callable is exempt.
- **Where:** `src/ragkit/core/registry.py:175-179`; overclaim `:166-167` ("guaranteed to satisfy the port").
- **Defect (verified concept):** `runtime_checkable` `isinstance` checks only that named members exist, not their signatures. A dotted-path driver whose `search(self, vector, k)` lacks `where=` passes both the structural check and the `create()` gate, then fails later with a `TypeError` deep in `search()`, far from registration.
- **Suggested fix:** soften the docstring to "has the port's members," and/or add an opt-in `inspect.signature` conformance check for the critical storage/model ports.

### M12 — Default StructureChunker records reconstructed, not source-exact, offsets
- **Status:** **FIXED as suggested** (locate each piece in `document.text`). A piece that is not a substring is refused with a structured error rather than recorded as a wrong span. Note the chunk's text and its span are now deliberately different lengths — the text joins pieces with single spaces, the span covers the original including its real separators — so `_chunk` takes an explicit end. No test asserted these fields at all before; several do now.
- **Where:** `src/ragkit/ingest/chunk.py:121-140` (`_pack`), recorded via `:32-36`; docstring `:1-9` promises provenance "traceable to an exact span."
- **Defect:** offsets are computed from a running cursor that adds one synthetic space per stripped piece; the real text has multi-char separators (`\n\n`, arbitrary whitespace), so recorded `char_start`/`char_end` are positions in a reconstructed single-space-joined string, not in `document.text`. Only `FixedChunker` records true offsets. No test asserts these fields.
- **Failure scenario:** any citation/highlight feature slicing `document.text[char_start:char_end]` gets the wrong span. Latent (no such slicer wired yet).
- **Suggested fix:** locate each piece in `document.text` (`document.text.index(piece, cursor)`) to record true offsets, or document that `_pack`-based chunkers emit approximate offsets.

### M13 — Documented pipeline order (`normalise` before `chunk`) silently defeats the default chunker
- **Status:** **FIXED by documenting**, as suggested. The `ingest` package docstring stated the pipeline order as `extract -> normalise -> ... -> chunk`, which is precisely the order that breaks it; it and `normalise`'s docstring now state the constraint and the two ways to satisfy it. Making the chunker not depend on `\n\n` was rejected: paragraph structure *is* what it chunks on.
- **Where:** `src/ragkit/ingest/normalise.py:28-29` (`collapse_whitespace=True` default) vs `chunk.py:107-118` (StructureChunker splits on `\n\s*\n`); order stated in `ingest/__init__.py:1-4`.
- **Defect:** collapsing whitespace folds `\n\n` to a single space, so the structure chunker sees one paragraph and falls back to whole-document sentence mode — silently discarding the paragraph/heading-preserving behavior that is its point. Latent (no `src/` code currently composes them; only recipe validators call `normalise` on outputs).
- **Suggested fix:** document that structure-aware chunking must run before whitespace collapse (or that `collapse_whitespace` must be off with the structure chunker), or make the chunker not depend on `\n\n` surviving.

### M14 — `reconcile_vector` materializes the full missing-id set + sorted copy in RAM
- **Status:** **FIXED, but not where the audit pointed.** Measuring first changed the answer: `sorted()` copies pointers, so at 7.1M ids it costs ~57 MB of array on top of the set, not another copy of the strings — and it buys deterministic, resumable batch order. Left as is, with the real profile written down. The dominant allocation was one level below, inside `reconcile` itself, which built the full authoritative set *and* the full indexed set before diffing: ~2× the necessary peak. Both drivers now stream the indexed side through one shared `reconcile_against`, holding one set instead of two. `indexed_ids()` is dropped as a result — never part of the port, no caller left in `src/`, and materialising every id is exactly the hazard being removed.
- **Where:** `src/ragkit/ingest/reference.py:153` (`missing = sorted(vector.reconcile(pairing_store.all_ids()))`).
- **Defect:** the embedding loop is chunked and `all_ids()` streams into `reconcile`, but the returned `set[str]` is fully realized and then `sorted()` into a second list. On a from-scratch reconcile (wipe/migration/large-tail crash) that's the entire id corpus held twice (~hundreds of MB of short strings at 7.1M) — the one non-streamed allocation on an otherwise streaming path. Bounded and ~100× smaller than the old whole-vector-table bug, but partly inherent to `VectorIndex.reconcile` returning a `set`.
- **Suggested fix:** if it matters at target scale, add a reconcile variant that yields missing ids in batches so the caller never holds them all.

### M15 — SqliteIntrospector silently creates a missing DB (empty schema); DuckDB fails loud
- **Status:** **FIXED as suggested.** The introspector opens read-only via URI, which makes a missing file raise *and* proves introspection cannot write to what it reads. The URI form now has one home, `_connect_read_only`.
- **Where:** `src/ragkit/store/sql/sqlite.py:116` (`sqlite3.connect` — read-write, creates on miss) vs `store/sql/duckdb.py:117-118` (`read_only=True`, raises on miss).
- **Defect:** a typo'd `[introspector].path` makes SQLite create an empty database and return `{}` with no error; NL→SQL then generates against an empty schema. A silent failure; the two drivers diverge for the same misconfiguration.
- **Suggested fix:** open the SQLite introspector connection read-only via URI (`mode=ro`, as `migrate.py:37` already does) so a missing file raises; align both introspectors on loud failure.

### M16 — BM25 relevance scales differ between the two pairing drivers → `min_score` not portable
- **Status:** **FIXED, with an honest limit.** Both drivers share one transform (`bm25_to_relevance`, taking a positive higher-is-better magnitude; SQLite passes `-bm25()` since FTS5's is negated). The rational form is kept over the exponential one because `1 - 2**-raw` is 0.999 by raw=10, collapsing every strong match onto one score. **It does not make the numbers equal** — the engines compute different raw BM25 (a term in every document scores 0.0 on FTS5, ~0.19 on DuckDB), so a floor transfers approximately; converging further would mean reimplementing BM25 instead of using each engine's native index. Conformance pins what actually holds (identical ranking, `[0, 1)`) rather than pinning a fiction. This also exposed that `SearchIndex` never stated the `[0, 1)` bound at all — the deeper reason `min_score` was not portable — and that the in-memory conformance double was returning raw overlap counts like `2.0`. Both fixed.
- **Where:** SQLite `1 - 2**bm25` (`store/lexical/bm25.py:13-20`) vs DuckDB `score/(1+score)` (`store/pairings/duckdb.py:67-70`); floor applied at `retrieve/retrievers.py:33`.
- **Defect:** both monotone into `[0,1)` (so ranking and RRF are unaffected), but the absolute values differ, so a `min_score` threshold admits/drops different hits after a driver swap. The conformance suite only asserts `score >= 0`, never score equivalence.
- **Suggested fix:** converge the two transforms on one shared mapping, or document `min_score` as driver-relative; add a conformance assertion pinning a known match's score.

---

## LOW

- **L1 — Width validator splits on the literal `"\\n"`, not `"\n"`.** `harness/validators.py:118` — a JSON-extracted output's line breaks are real `\n`, so multi-line answers are measured as one concatenated line (spurious over-limit note; genuinely long inner lines never flagged). WARNING-severity only. *Fix:* split on `"\n"`.
  - *Status:* FIXED as suggested.
- **L2 — Duplicated `blocking()` (SSOT).** `harness/validators.py:166` and `core/rules.py:37` are byte-equivalent; the `core.rules` copy is exported but unused. *Fix:* delete one and import the other.
  - *Status:* FIXED; the `core.rules` copy is the survivor (it sits beside `Violation`), and the harness imports it.
- **L3 — `records.from_json` coercion gaps.** `records.py:186-188` — `notes=tuple(raw.get("notes", ()))` splits a JSON *string* into per-char entries; `span_start`/`span_end`/`line_no` get no int check (a string span makes `__post_init__` raise a bare `TypeError` the `except ValueError` misses → unstructured crash on `ragkit import`). *Fix:* validate field types before constructing; catch `TypeError` alongside `ValueError`.
  - *Status:* FIXED as suggested: `notes` must be a JSON array, the three integer fields are validated before construction, and `TypeError` joins `ValueError` in the handler as a backstop.
- **L4 — `jsonshape.json_type_matches` accepts unknown type names.** `core/jsonshape.py:30-31` returns `True` for every value when a declared type isn't in `JSON_TYPE_CHECKS` — a schema-author typo silently disables that field's check (reachable via a custom dotted-path `OutputSchema`). *Fix:* treat an unknown type name as an error for the framework-controlled flat case.
  - *Status:* FIXED as suggested; an unknown type name now raises `JsonShapeError`, including inside a union.
- **L5 — Dead `DEFAULT_RERANK_MIN_SCORE` + misleading docstring.** `retrieve/rerank.py:20-22` — no consumer; unlike the lexical/dense floors there is no rerank-floor knob at all, yet the docstring implies symmetry. *Fix:* wire a real `rerank_min_score` through `RetrievalSettings`, or delete the constant and correct the docstring.
  - *Status:* FIXED by **deleting** the constant, not wiring a knob. A rerank floor would sit at the same point as the block's own `min_score` — the reranked score *is* the final relevance — so it would be a second knob with nothing to do. The docstring's implied symmetry with the fusion-input floors was the error.
- **L6 — Circularity guard is nominal-string-only.** `eval/retrieval.py:117-121` compares `qrels.source` to `systems` keys; ground truth genuinely from a system-under-eval but labelled with any other string passes. The docstring frames it as "mechanical" independence. *Fix:* document the guard's true (nominal) scope, and/or draw `source` from a closed vocabulary of known-external sources.
  - *Status:* FIXED by documenting the guard's true (nominal) scope on `CircularEvaluationError` itself. A closed vocabulary of sources was not added: it would refuse legitimate third-party gold sets without actually verifying provenance.
- **L7 — Missing-ground-truth raised as `CircularEvaluationError`.** `eval/retrieval.py:122-127` — a "missing gold" condition uses the circularity error type, so a caller handling circularity mishandles a data-completeness problem. *Fix:* a distinct error type.
  - *Status:* FIXED as suggested: `IncompleteGroundTruthError`, with a test pinning that it is not a `CircularEvaluationError` subclass.
- **L8 — False `# pragma: no cover` in judge.** `eval/judge.py:128-129` marks a branch unreachable "because the enum," but `_check_schema_shape` (`llm/client.py:330-348`) never validates `enum`, so a `json_object` backend can reach it. The branch correctly fails loud; the comment is wrong and hides a test-worthy path. *Fix:* correct the comment; add a regression test feeding an out-of-enum reply.
  - *Status:* FIXED as suggested: comment corrected, pragma removed, and the branch is now tested both directly and end to end through a scripted out-of-enum reply.
- **L9 — `Registry._from_entry_points` replaces *all* dots.** `core/registry.py:225` — `value.replace(".", ":", -1)` turns `pkg.mod.Attr` into `pkg:mod:Attr`. Latent (setuptools object-refs always contain a colon, taking the other branch). *Fix:* `value.rsplit(".", 1)` joined with `":"`.
  - *Status:* FIXED as suggested (`rpartition`, one home in `_as_dotted_path`).
- **L10 — CLI boundary inputs unvalidated.** `cli/main.py` — `--limit -1` does `records[:-1]` (silently drops the last record); `--concurrency` accepts `0`/negatives; `--set =value` accepts an empty key; arg errors raise bare `SystemExit` bypassing the structured error/log path. *Fix:* reject non-positive `--limit`/`--concurrency` and empty keys with clear messages.
  - *Status:* FIXED as suggested, plus the structured-error half: `--limit`/`--concurrency` are argparse-level positive-integer checks (so `--limit -1` is refused before any work rather than silently dropping the last record), an empty `--set` key is refused, and the three bare `SystemExit`s that bypassed `main()`'s print-and-log path now raise `CliError`.
- **L11 — `RagkitError.context` shape inconsistent.** `ConfigError` puts `path`/`label` in `context`; `CatalogError`/`LexiconError` fold them into a single `context["location"]` string. A handler reading `exc.context["path"]` (as the errors docstring advertises) works for config but not catalog/lexicon. *Fix:* settle on one convention (carry `path`/`line_no` as structured keys everywhere).
  - *Status:* FIXED as suggested: a shared `LocatedError` base carries `path`/`line_no` as structured keys everywhere, which also collapses three identical `__init__` bodies to none.
- **L12 — Recipe defaults inlined, not single-homed.** `cli/app.py:156-167` retypes every default inline instead of a `_DEFAULTS = _Recipe(...)` home (contrast `tuning.py`). Nothing drifts *today*, but the `"json_field"` default silently couples to `harness/schemas.py:54`. *Fix:* give `_Recipe` field defaults.
  - *Status:* FIXED with M7.
- **L13 — `app.py` docstring "before the server is contacted" is inaccurate for dense/hybrid.** `cli/app.py:9-12` — for `kind=dense/hybrid`, `assemble()` embeds the corpus, contacting the embedding endpoint. The config checks *do* precede that, so fail-fast holds; the blanket claim is false. *Fix:* qualify the docstring.
  - *Status:* FIXED with M7: the docstring now describes the fail-fast property that actually holds (every config check precedes any server contact) instead of the stronger claim.
- **L14 — `embed_and_upsert` duplicates full pairing `meta` into the vector index as dead weight.** `ingest/reference.py:109-110` — the entire JSON record is copied into the ANN meta store, but the read path resolves display meta via `pairing_store.document()`, so it's never read (contradicts `RetrievedRef`'s SSOT intent). Substantial disk/IO bloat at scale, no correctness impact. *Fix:* pass minimal/empty metas to `vector.upsert`.
  - *Status:* FIXED as suggested (empty metas). The consequence is written down: a filtering-capable driver has nothing to filter on through this path, and a caller wanting payload filtering populates the index itself.
- **L15 — `dedup` seen-set stores 64-char hex digests (2× memory).** `ingest/dedup.py` — `hexdigest()` instead of `.digest()`. Latent (not on the active 7.1M import path). *Fix:* store the 32-byte `.digest()`.
  - *Status:* FIXED as suggested; `content_hash()` still returns hex, which is the public form.
- **L16 — `JsonlExtractor` raises a bare `KeyError` on a missing `id_field`.** `ingest/extract.py:77` — inconsistent with the module's structured-error contract. *Fix:* check membership and raise `ExtractError(f"line {line_no}: no id field {self._id_field!r}")`.
  - *Status:* FIXED as suggested.
- **L17 — `EstablishedBlock` renders a dangling `line -> ` for an empty neighbour output.** `harness/context/blocks.py:172-173` filters on `is not None`; an empty-string recorded output renders a blank RHS. *Fix:* filter on truthiness (`if (output := memory.get(line))`).
  - *Status:* FIXED as suggested.
- **L18 — `HtmlExtractor` over-segments inline text and drops inter-node spaces.** `ingest/extract.py:104-112,135` — every text node is stripped and joined with `\n\n`, so `<p>The <b>quick</b> brown fox</p>` becomes three "paragraphs." *Fix:* break only on block-level close tags; join inline runs with a single space.
  - *Status:* FIXED as suggested: text accumulates into the current paragraph and flushes only at a block-level tag, so inline runs keep their spaces. This also stopped it defeating the structure chunker downstream, which splits on exactly those blank lines.
- **L19 — `embed_presets.ini` overclaims scope.** `tools/embed_presets.ini:2-3` says "used by every recipe's `[endpoint.embed]`"; only 2 of 6 define one. *Fix:* "used by the recipes that enable dense retrieval (`med_evidence`, `legal_procurement`)."
  - *Status:* FIXED as suggested.
- **L20 — Minor doc/behaviour mismatches.** `MarkdownExtractor` docstring says "top-level ATX (`#`/`##`)" but treats any `#`-leading line (any depth, even inside fenced code) as a boundary; `dedup.py:5` docstring says "over normalised text" but hashes verbatim text; `SqlRowsBlock` (`blocks.py:282`) fetches the full result then slices `[:limit]` in Python instead of a SQL `LIMIT`; `check_environment.sh:30` can `die` (via `project_python`) before printing its report row. *Fix:* align each docstring/behaviour.
  - *Status:* ALL FOUR FIXED. MarkdownExtractor: description corrected, behaviour kept (splitting at every depth is the useful behaviour for retrieval). dedup: docstring corrected to "verbatim", with the consequence stated. SqlRowsBlock: the limit is pushed into the SQL via a subquery wrap, verified against both shipped drivers, a WITH clause, and a query that already carries its own LIMIT. check_environment.sh: reports a missing interpreter as a failed row instead of dying before printing anything.

---

## Verified CLEAN (recorded so the audit is balanced)
- **Retrieval metrics** (`eval/retrieval.py`) — `recall_at_k`, `hit_rate_at_k` (Acc@10), `reciprocal_rank`, `average_precision`, `ndcg_at_k` hand-checked against definitions; the Acc@10 depth handling (`retrieve at max(k,10)`, slice per metric) is correct.
- **Fusion/hybrid/rerank/retrievers** — no sign-convention/ranking-inversion bug; RRF, MMR, the hybrid sort, per-arm floors, and the overflow-safe `sigmoid` are all correctly higher-is-better; determinism and concurrency-safety hold. Embedding NaN/Inf rejection is correct and complete.
- **Produce→review→revise loop** — abstention vs rejection handled correctly; a degenerate/garbled reviewer reply becomes `PRODUCED` (never `VERIFIED`, never abstain-as-pass); all loops bounded; a whole-panel outage stops the run rather than marking everything accepted. Validator abstain-vs-block, RunStore write path, runner RAII, and capture fidelity all sound.
- **Context assembly** — budget/ordering/no-empty-section/`min_score` correct. **Write-back idempotency** and **import resume** are safe/idempotent; **dedup** is correct exact-match.
- **`core/`** — `config.py` type-readers correctly refuse coercions (the SSOT `_load_recipe` should use); `records.py` durability (atomic temp+fsync+rename, torn-final-line tolerance, orphaned-journal refusal); `width.py` unicode column math; `lexicon.py`, `rules.py`, `placeholders.py`.
- **Tool scripts** — all 12 `tools/*.sh` + `lib/common.sh` are well-hardened (arg validation, precondition checks, `|| die` on heredocs, install hints). Only nit: `check_environment.sh` can `die` before printing.
- **Docs** — major reference docs (`config.md`, `architecture.md`, `tutorial.md` signatures) match code; the retired legacy `[lexical]`+`[documents]` split store is correctly documented as removed; version pinning is internally consistent (Python 3.11+ floor, GGUF filenames match presets + `models/`).

## Interface / swappability verdict
- **Ports are well-designed** — backend-neutral `Filter` AST (not raw dicts), score-normalization pushed inside each driver, `read_only` enforced at the port, streaming iterators on `RunStore`, registry `isinstance`-gating; callers depend on ports, not concrete drivers.
- **`Backend` (model family): genuinely swappable** — the constrain-decoding vs describe-in-prompt split is implemented cleanly.
- **`Provider` (LLM transport): the weakest seam** — the port is **unimplemented**, `LlmClient` doesn't satisfy it, `EndpointSpec.provider` is validated then **ignored** (so `ollama`/`openai-compatible` behave identically to llama.cpp), and llama.cpp specifics leak (`chat_template_kwargs` on by default, 429/408 not retried, `--jinja`/router flags in `serveargs`). Adding a genuinely different endpoint kind today means editing the pool and matching `LlmClient`'s concrete surface — the opposite of what the port advertises.
  - *Related latent items:* `client.py:216-219` raises immediately on any 4xx (transient `429`/`408` should back off + retry); `client.py:189-194` sends `chat_template_kwargs` by default (a strict-OpenAI server 400s). Both only bite against a non-llama.cpp endpoint; fixing them belongs with making `Provider` real.
  - **RESOLVED 2026-08-03** (`.audit/2026-08-03-provider-implementation.md`). `Provider` is now a real port with a concrete registry (`llm/providers.py`), mirroring `Backend`: each endpoint kind declares its capabilities and request quirks, `EndpointSpec.provider` drives behaviour (no longer validated-then-ignored), the `chat_template_kwargs` leak is gone (only a provider that honours it receives it), and an unsupported capability — rerank on Ollama — is refused at build time, as is a non-`llamacpp-router` endpoint in `serveargs`. The `429`/`408` half of the related items was already fixed in PR #9 (`e32736e`, transient-retry unified across the chat/embedding/rerank clients).
- **Storage: swappable on the happy path only** — the real divergences (M3 LanceDB filtering, M5/M6 DuckDB O(n²)+non-atomic, M15 introspector-missing-file, M16 BM25 scale) all live in corners the conformance suite doesn't test (filtering, atomicity, score-equivalence, concurrency). Widening `tests/store/test_conformance.py` to cover them is what turns "the driver pairs are equivalent" from an assumption into a tested fact.

## Suggested fix order
1. **H1** (silent rerank drop) and **H2** (poison pill) — small, high-value correctness fixes.
2. **M2** (`nprobes`) — directly improves the retrieval numbers reported 2026-08-01.
3. **C1/H3** (`fetch_models.sh` repos) — unblocks setup for 5 recipes; needs a real-model decision first.
4. **M4** (`merge_insert`), **M1** (pool lock) — durability + RAII.
5. The rest as a batched cleanup, with **M10** (eval SSOT) and a widened conformance suite as the larger refactors.

## Coverage caveats
- HF repo non-existence (C1) is inferred from local evidence; no network to confirm via the HF API.
- The docs sweep's two delegated sub-audits (a full cross-recipe SSOT pass and a 6-recipe README-vs-metric reconciliation) did not consolidate; the SSOT finding (M10) reflects firsthand reads of 2 evals + `grep` across all 6, so a per-README metric reconciliation for the other 5 recipes remains open.
