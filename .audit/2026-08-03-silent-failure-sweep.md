# 2026-08-03 — Whole-codebase silent-failure sweep (+ external-edge review)

**What was checked.** An exhaustive hunt for **silent failures** across all of `ragkit` — any place
a failure is swallowed, papered over with a fallback/retry-and-hope, degraded quietly, or returned
as a sentinel/default such that a *caller cannot tell it failed* (CLAUDE.md: "Never allow failures
to be silent"; "No silent fallbacks or workarounds"). The one legitimate carve-out honoured
throughout: a *documented, explicit* fallback for a genuinely **external** failure (flaky service,
transient network, a filesystem capability that varies) is allowed; an internal bug hidden by a
fallback is not.

Plus a focused **external-boundary review** of the edges the codebase talks to the outside world
across: the LLM HTTP wire protocol, and the database drivers.

**Method.**
1. A deterministic AST sweep classified **every one of the 101 exception handlers** (and every
   `contextlib.suppress`) in `src/ragkit` by what it does. 22 were potentially-swallowing; all 22
   verified legitimate (retry-then-raise, convert-to-`Outcome`, CLI-boundary print+exit,
   best-effort-with-authoritative-check-later, or `_safe_rollback` suppressing a *secondary* error
   so it cannot mask the real one). The exception-handler layer is clean.
2. Four subsystem agents read the non-exception silent paths (sentinel returns, ignored return
   values, `.get(key, default)` masking a required key, silent-degradation branches), each
   cross-referencing real callers.
3. A personal read of the LLM wire path (`client.py`/`embedding.py`/`rerank.py`) and the DB drivers
   (WAL/`busy_timeout`, transaction atomicity, connection lifecycle).

**Verdict.** The codebase's structured-error discipline is genuinely strong — most of it came back
clean. Eight real findings surfaced, all now **fixed** (each its own commit, each with a regression
test, 100% statement+branch coverage held throughout).

---

## Findings (all fixed)

### A — Component `from_config` bypassed the strict config readers (silent coercion)
- **Where:** `harness/context/blocks.py` (7 blocks), `harness/schemas.py` (`FormSchema`),
  `ingest/chunk.py` (3 chunkers), `ingest/extract.py` (4 extractors).
- **Defect:** they read options with raw `int()/float()/bool()/str()/tuple()`, so a mistyped option
  was silently coerced exactly as the recipe loader did before the M7 fix: `min_score = true` →
  `1.0` (retrieval floor becomes exact-match-only), `limit = 12.5` → `12`, `required = "false"` →
  `True` (constraint inverted), `keys = "temperature"` → a char-tuple (block renders nothing). This
  violated `core/config.py`'s own stated invariant that *every* component option block reads through
  its helpers, and `Registry.create` key-checks options but never type-checks them.
- **Fix:** all now route through `read_int`/`read_float`/`read_bool`/`read_string`/
  `read_string_list`, so a mistyped option is refused with a located `ConfigError`. No shipped
  recipe relied on the coercion.

### B — `null`/non-string ingest fields coerced to the string `"None"` (silent data corruption)
- **Where:** `ingest/reference.py` (`reference_pairings`, source/target), `ingest/extract.py`
  (`JsonlExtractor`, text/id).
- **Defect:** field values were `str()`-coerced, so a JSON `null` became the 4-char string `"None"`
  and was stored/indexed as real content — teaching a later run that the correct output is literally
  `"None"`. The tell-tale asymmetry proved it unintended: an *absent* target yielded `""`, a *null*
  target yielded `"None"`. Common at 7.1M-row corpus scale.
- **Fix (per the chosen policy):** a `null`/absent source is a blank source (skipped); a
  `null`/absent target is an empty lexical-only entry; a `null` id/text is refused; a present
  non-string value (number, list, object) raises a structured `ReferenceImportError`/`ExtractError`
  naming the line. A numeric JSON id stays legitimately stringified.

### C — `_collect` silently dropped unresolvable retrieval hits
- **Where:** `retrieve/retrievers.py:_collect`.
- **Defect:** a hit whose text `resolve()`→`None` was dropped with a bare `continue` — no log, no
  count. Two cases: the **lexical** arm's search index and text store are the *same* co-located
  `PairingStore`, whose port contract says a hit and its text cannot drift apart, so an unresolvable
  hit there is an internal invariant violation (a corrupt store) being swallowed; the **dense** arm
  can legitimately drift between reconciles, but the drop was unsignalled, so a badly-stale vector
  index would silently halve every result set.
- **Fix:** lexical raises a new `RetrieverError` (fail loud on corruption); dense still drops but
  emits a `logger.warning` naming the drop count.

### D — The three LLM HTTP clients disagreed on transient-failure retries
- **Where:** `llm/client.py` (chat), `retrieve/embedding.py`, `retrieve/rerank.py`.
- **Defect:** embedding retried `429`+`5xx`+transport (correct), chat raised immediately on a
  `429`/`408` it should have backed off on, and rerank had *no retry loop at all* — a single
  transient `503` (llama.cpp answers 503 when every `--parallel` slot is busy, exactly what happens
  on a concurrent hybrid eval) aborted the whole run. Not silent, but a transient/retryable failure
  treated as fatal, inconsistently.
- **Fix:** new `llm/http.py` is the one home for the transient classification
  (`TRANSIENT_HTTP_STATUS = {408,429,500,502,503,504}` + timeouts/transport errors), imported by all
  three. Chat now retries any transient status (still raising a deterministic 4xx at once); rerank
  gained the embedding client's retry+backoff, wired from the endpoint's
  `max_retries`/`retry_backoff_seconds`.

### E — A forgotten `path` silently became an ephemeral `:memory:` store
- **Where:** `store/pairings/{sqlite,duckdb}.py`, `store/run/sqlite.py`, `store/lexicon/sqlite.py`
  (`from_config`).
- **Defect:** `str(options.get("path", ":memory:"))` — a `[pairings]`/`[run]`/`[lexicon]` config
  that named a driver but forgot the path silently produced an in-memory store: writes and reads
  worked in-process, nothing persisted across runs, no error. A run store that vanishes cannot
  resume; a pairings store that vanishes silently re-imports and re-embeds the whole corpus every
  run. `LanceVectorIndex` already refused this ("needs a 'path'"); the SQLite-family stores were
  inconsistent.
- **Fix:** new `core.config.read_required_path` refuses an absent/blank path with a located
  `ConfigError`, while an explicit `path = ":memory:"` stays a valid deliberate choice. The `[sql]`
  external store is left as-is, where `:memory:` + `schema_sql` is a legitimate ephemeral pattern.
  `docs/config.md` updated.

### F — `SqlRowsBlock` bound SQL `NULL` for a missing declared `param_key`
- **Where:** `harness/context/blocks.py` (`SqlRowsBlock.render`).
- **Defect:** `record.meta.get(key)` for an absent declared `param_key` → SQL `NULL` → `= NULL`
  matches nothing → the block rendered no rows, indistinguishable from a genuine no-match, so the
  producer silently ran without the historical context it was configured to receive.
- **Fix:** a declared `param_key` absent from `record.meta` is refused with `ContextBlockError` (a
  missing bind changes the query's meaning, so it is not a legitimate empty section).

### G — `EstablishedBlock` did not validate `before`/`after >= 0`
- **Where:** `harness/context/blocks.py` (`EstablishedBlock.__init__`).
- **Defect:** its sibling `NeighboursBlock` refuses a negative window; `EstablishedBlock` accepted
  one, and the `limit <= 0` path then silently dropped the established context.
- **Fix:** the same `before/after >= 0` guard added.

### H — `_build_reference` silently returned `None` for an unresolvable retriever name
- **Where:** `cli/app.py` (`_build_reference`).
- **Defect:** the gate `":" in spec or spec in RETRIEVERS.available()` misses entry-point retriever
  names (they resolve lazily in `create()`, not in `available()`), so a bare entry-point name (or a
  typo) with no `[reference].file` fell through to `return None` — the harness silently wired with
  no retriever, contradicting the method's own docstring.
- **Fix:** any non-`lexical` spec that is a dotted path, a registered name, or has no corpus file is
  resolved through the registry, which raises a clear error on a genuinely unknown name.

---

## Verified clean (recorded so the sweep is balanced)
- **All 101 exception handlers** + every `contextlib.suppress` account for their failure (re-raise/
  convert to a structured error, retry-then-raise, or a documented secondary-error suppression).
- **DB edge:** WAL + `busy_timeout=5000` on all three writable SQLite stores (a locked DB retries,
  doesn't fail); FK enforcement + a documented `synchronous` option on the run store; read-only URI
  opens that won't create a missing file; every driver error converted to a structured
  `SqlStoreError`/`PairingStoreError`. Dedup is safe because every id is content-addressed. The
  DuckDB `all_ids` shared-result-set hazard is already handled with a dedicated cursor.
- **`store` package** categories 1/2/4/5 (sentinels, ignored returns, degradation, dedup): clean.
- **`llm`:** clean (the empty-string-completion gap is latent/unreachable — only `complete_json`
  calls `complete`, and it rejects `""`).
- **`harness`:** the review/settle logic (abstention never counted as a pass), memory accumulation,
  runner accounting, and capture fidelity are clean.
- **`core`/`cli`/`eval`/`ingest`:** config readers refuse coercions; records raise structured
  `CatalogError` on any bad shape (only a torn *final* line is tolerated, documented); registry
  raises on every unknown/non-conforming spec; `jsonshape` now *raises* on an unknown type name (a
  prior doc-pass candidate, confirmed already fixed); eval metrics raise on empty gold and model
  garbage scores as a documented *miss*.

## Coverage caveat
The four subsystem agents' non-exception sweep is thorough but not a proof of absence; it is a
best-effort read of the reachable sentinel/ignored-return/degradation paths cross-referenced against
real callers. The exception-handler layer, by contrast, was enumerated exhaustively by AST.
