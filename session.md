# Session state — paused 2026-07-28 ~20:57

Branch: **claude-branch** (never push to main; PR only when asked).
Health at pause: **843 tests pass, 100% statement+branch coverage (enforced by `.coveragerc`
`fail_under=100`), lint clean** — but the current translation work is **uncommitted** (see below).

Run checks with: `bash tools/run_tests.sh --coverage` and `bash tools/lint.sh`.

## The three tasks the user gave (this session)

1. **SSOT scan + fixes — DONE & COMMITTED** (`6d4e97d`). Two grounded Explore sweeps found genuine
   duplications; fixed the clear ones: 5 dataclass-default-vs-loader-default duplications
   (EndpointSpec/ModelSpec in `llm/pool.py`, Limits+Panel-revision in `harness/roles.py`,
   RetrievalSettings in `retrieve/tuning.py`) now read from a module-level default instance /
   named constant; the read-only refusal messages are shared `SqlStoreError.write_on_read_only()`
   / `.schema_on_read_only()`; `_SAMPLING_KEYS` derived from `fields(SamplingParams)`; the
   JSON-type predicate promoted to `core/jsonshape.py:json_type_matches` (reused by the client's
   schema check and the form_autofill validator). **Deliberately left (mild, noted in commit):**
   the `_require_X` lazy-import shape, the `_owns_client` close-what-I-opened pattern, the
   `max(0,min(1,x))` score clamp — messages/classes differ per site.

2. **Faithful llm-translator replication — IN PROGRESS, UNCOMMITTED.** The reference repo is cloned
   at **`/tmp/llm-translator`** (study `config/agents.toml`, `config/translation_rules.toml`,
   `src/translator/agents.py`). Rewrote `recipes/translation/config/` to faithfully reproduce it:
   - `personas.toml`: the verbatim translate prompt + the **full 5-reviewer panel in order**
     (accuracy → structure → grammar → fluency → compliance-from-rules), with limits/leniency/revision.
   - `rules.toml`: all **6 forbidden patterns** + **7 advisories** + 5 style directives + the
     "keep each line ≤110 columns" directive + `max_line_columns`/`max_columns_tolerance`/
     `keep_flagged_rules=["line_width"]`.
   - `context.toml`: lexicon(12) + retrieved(TM) + neighbours(3/2) + **established** (their
     translations, = llm-translator `include_translations`) + previous_attempt + budget.
   - `recipe.toml`: added `stand_in = "they"` (llm-translator `anonymous_subject`); `use_memory=true`.
   - `tests/test_recipe.py`: added `TestFaithfulToLlmTranslator` (panel order, 6 forbidden + 7
     advisories, system-prompt content incl. "Who does what to whom"/"Style policy:"/"110 display
     columns"/ends "Respond only with the requested JSON object.", user-prompt context sections,
     forbidden-preamble → REJECTED). All 19 translation tests pass.
   - `README.md`: rewritten to state it's a faithful llm-translator port.
   Note: the framework's `_system_prompt`/`_user_prompt` (harness/agents.py) already match
   llm-translator's structure verbatim (it was generalised from it) — the only framework difference
   was llm-translator's standalone "keep line ≤N columns" sentence, reproduced here as a style
   directive rather than a framework change (which would wrongly affect the SQL/form/maintenance
   recipes).
   **NEXT: commit this** (all 6 modified files under `recipes/translation/`), full suite is green.

3. **Test sample projects with all DBs — NOT STARTED.** Two real drivers now exist per DB port:
   SqlStore = sqlite|**duckdb**; VectorIndex = lancedb|**qdrant**; LexicalIndex = fts5 (+in-memory
   conformance impl). The recipe end-to-end tests currently exercise only the defaults. Need to
   parametrize / add DB-swap coverage so each recipe runs against BOTH real drivers of every port it
   uses:
   - `nl_to_sql`, `form_autofill` use an external `SqlStore` (read-only) + `SchemaIntrospector` →
     run their end-to-end tests against **sqlite AND duckdb** (build the fixture DB with each; note
     DuckDB SQL dialect differs slightly, e.g. `VARCHAR`, `information_schema`).
   - `translation`, `predictive_maintenance` use retrieval memory → run against the vector path with
     **lancedb AND qdrant** (via a `retrieval.toml` dense/hybrid config + the `retrieval_factory`
     MockTransport pattern in `tests/cli/conftest.py`, or a storage.toml `[vector]` swap), and the
     lexical `fts5` path. The `tests/store/test_conformance.py` already proves the ports are
     interchangeable; task 3 is proving each RECIPE works on each real driver end-to-end.
   Suggested approach: a shared pytest parametrization over `storage.toml` driver strings
   (`sqlite`/`duckdb`, `lancedb`/`qdrant`) in each recipe's `test_recipe.py`, reusing the existing
   `_staged_config` + MockTransport factories. Keep it hermetic (no server): DuckDB/Qdrant local
   modes need no network.

## Conventions to keep (hard requirements)
- 100% statement+branch coverage on `src/` is GATED — a new src line needs a test or a justified
  `# pragma: no cover` (server/network-only paths).
- Every model/embedding/rerank call in tests goes through `httpx.MockTransport` — no server, GPU, or
  network. DBs run embedded/local (sqlite/duckdb/lancedb/qdrant all in-process).
- `reject_unknown` on every loader; `read_int`/`read_float` reject bool; no empty prompt sections;
  RAII (close only the client you opened); docstrings explain *why*; tools/ scripts hardened.
- Datasets are `.gitignore`d (never commit `recipes/*/data/`); fetchers are hardened `fetch.sh`.

## Useful pointers
- Config reference: `docs/config.md` (per-file option tables incl. retrieval.toml, duckdb, qdrant).
- Audit trail: `.audit/2026-07-28-cross-cutting-sweep.md` (records the cross-check + approved
  decisions: Provider kept OpenAI-compatible-only, retrieval.toml built, 2nd drivers shipped).
- llm-translator source for reference: `/tmp/llm-translator` (may be cleared on reboot; re-clone
  `https://github.com/AzethMeron/llm-translator` if gone).
