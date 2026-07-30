> **Status:** Approved 2026-07-30, implementation pending (to be executed in a later session).
> Scope, decisions, and phasing below are the agreed design; nothing here is built yet.

# Full storage overhaul — DB-native reference memory, run store, write-back

## Context

`ragkit` currently uses **flat files as the source of truth for stateful data** and only derives
databases from them — the "misuse of databases" the user flagged. Concretely:

- The **reference retrieval memory** is a JSONL file (`[reference].file`) re-ingested every
  `assemble()` into a *derived, rebuildable* cache; the **row store (`SqliteDocuments`) and the
  lexical index (`Fts5Index`) are two separate SQLite databases** with no cross-store transaction,
  so they drift and are held together by write-ordering + `_ingest`/`_discard_tail` resume +
  `VectorIndex.reconcile` (defined but with **no runtime caller**). There is **no write path** — you
  cannot add a pairing at runtime, so write-back is impossible without replacing the retriever
  wholesale via `assemble(retriever=...)`.
- The **run catalogue + journal are JSONL** files; the **in-run `OutputMemory` is RAM-only** (never
  persisted; `from_records` defined but unused → the `established` block is non-reproducible); the
  **`Sink` port exists but is never implemented**; and the **context that produced an output is
  discarded** (retrieved chunks, assembled passage, structured reviews are not persisted — only
  `source`/`output`/`status`/flattened `notes`/`meta` survive).

**Goal:** make the reference memory an authoritative, **writable** DB store; move all stateful data
(reference memory, run catalogue/journal, lexicon, in-run memory) onto databases; capture the
context that produced each output; and add deterministic **write-back** of produced
`(source, context, target)` triples. The external read-only `SqlStore` (nl_to_sql / form_autofill)
is a *different, correct* role and is left unchanged.

### Decisions (from the user)
1. **Scope = full storage overhaul** (reference memory + run journal/catalogue + lexicon + in-run memory).
2. **Pairing store = rows + FTS5 co-located** in one SQLite (DuckDB as 2nd real driver / in-memory for conformance); the vector index stays separate and synced.
3. **Write-back = separate post-run step** (a CLI reads a finished run, writes verified triples into the pairing store).
4. **Context capture = capture + persist** the retrieved chunk ids/text and the assembled passage, so `(source, context, target)` is complete and reproducible.

## Target architecture — three DB roles, all behind ports resolved by the registry

| Role | Store (new port) | Writer | Notes |
|---|---|---|---|
| External task data | `SqlStore` (read-only) | never | **unchanged** — do not touch nl_to_sql/form_autofill |
| Reference memory | **`PairingStore`** (rows+FTS5 in one SQLite; vector separate) | import + write-back | authoritative + writable |
| Run state | **`RunStore`** (records + results, ACID) | runner thread | replaces JSONL catalogue+journal |
| Terminology | **`LexiconStore`** | import | co-located in the pairings DB |

### New ports (`src/ragkit/core/ports.py` — stdlib-only Protocols + value types; keeps `core` clean, `test_boundaries` green)
- `Pairing(chunk_id, source, target="", context="", meta={}, verified=False, created_at=0.0)`.
- `PairingStore`: `add(Iterable[Pairing])->int` (bulk, **one txn**, syncs FTS), `search(query,*,k)->[(id,score)]` (**same signature as `LexicalIndex.search`**), `document(id)->(display,meta)|None` (**same as `DocumentStore.document`**), `get(id)->Pairing|None`, `all_ids()->Iterator[str]` (feeds vector reconcile), `count()->int`.
- `RunStore`: `add_records(Iterable[Record])->int`, `append_result(RunResult)->None` (one commit), `completed_ids()->set[str]`, `pending(*,key=catalog_order)->Iterator[Record]` (**streamed cursor**), `results()->Iterator[RunResult]` (latest-per-record), `count_records()->int`.
- `RunResult(record, context_passage="", retrieved=(), reviews=(), violations=(), rounds=0, error=None)` — the `Record` projection **plus** the capture JSONL cannot hold.
- `LexiconStore`: `entries()->list[Entry]`, `add(Iterable[Entry])->int`.

**Key reuse (reuse-over-new):** because `PairingStore` re-exposes `search` and `document`, the one
SQLite driver **structurally satisfies `LexicalIndex` + `DocumentStore` + `PairingStore`** and is
registered in all three registries — so `LexicalRetriever` / `DenseRetriever` / `HybridRetriever` /
`retrieve/{fusion,rerank,hybrid}.py` are reused **verbatim**. Also reuse: `records.py`
read/write/merge as the JSONL **import/export** SSOT; `_Interruptible`/`group_duplicates`/`Progress`/
`learn_memory` in the new runner; `OutputMemory.from_records` (finally wired); `reconcile` (finally
wired); registry/`Storage`/`load_storage` (extended, not replaced). **Extract** `_bm25_to_relevance`
+ `_as_match` from `fts5.py` into `store/lexical/bm25.py` (SSOT) and the batch-embed-dedup helper
from `Corpus._embed_and_upsert`.

### Key schemas
**Pairings DB — co-location via FTS5 external-content + triggers (rows + index, one file, one transaction → drift structurally impossible):**
```sql
CREATE TABLE pairings(id INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE NOT NULL,
  source TEXT NOT NULL, context TEXT NOT NULL DEFAULT '', target TEXT NOT NULL DEFAULT '',
  meta TEXT NOT NULL DEFAULT '{}', verified INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL DEFAULT 0);
CREATE VIRTUAL TABLE pairings_fts USING fts5(source, context, target,
  content='pairings', content_rowid='id', tokenize='unicode61');
-- AFTER INSERT/DELETE/UPDATE triggers keep pairings_fts in sync inside the same txn.
```
Display text (replacing `_default_display`) is computed from the row: `"source -> target"` if
`target` non-empty, else `source`. `chunk_id` = `ref-<line>` on import; deterministic content-hash
(`INSERT OR IGNORE` → idempotent) on write-back. Lexicon lives in the same DB (reference data).

**Run DB (WAL; `synchronous=FULL` default = fsync-per-commit, matching today's per-group fsync):**
```sql
CREATE TABLE records(record_id TEXT PRIMARY KEY, source TEXT NOT NULL, status TEXT NOT NULL,
  rel_path TEXT DEFAULT '', line_no INT DEFAULT 0, span_start INT DEFAULT 0, span_end INT DEFAULT 0, meta TEXT DEFAULT '{}');
CREATE TABLE results(seq INTEGER PRIMARY KEY AUTOINCREMENT,           -- monotone => later-wins
  record_id TEXT NOT NULL REFERENCES records(record_id),             -- FK => orphan guard
  status TEXT NOT NULL, output TEXT, notes TEXT DEFAULT '[]', violations TEXT DEFAULT '[]',
  reviews TEXT DEFAULT '[]', rounds INT DEFAULT 0, error TEXT,
  context_passage TEXT DEFAULT '', retrieved TEXT DEFAULT '[]', created_at REAL DEFAULT 0);
```
`completed_ids`=`SELECT DISTINCT record_id FROM results`; `pending`=records `status='pending'` and
`NOT EXISTS` in results, `ORDER BY rel_path,line_no` (indexed, streamed); `results()`=row at
`max(seq)` per record. A result write is one transaction — **ACID replaces** temp+rename atomicity,
per-line fsync, and torn-final-line handling (a partial row is impossible).

## Phases (each keeps the full suite green)

**P0 — Groundwork (additive, no behaviour change).** Extract `store/lexical/bm25.py` + the
embed-dedup helper. Add optional `context_passage`/`retrieved` to `Outcome` (`harness/agents.py`),
populated via a **capture sink** threaded through `_shared_context`: `RetrievedBlock` + the grounding
validators append the hits they *already* retrieve to `context["capture"]` (no double-retrieval, no
semantic change); `_user_prompt` records the assembled passage. Wire `OutputMemory.from_records` at
run start. Tests for capture + seed; `applied_to`/JSONL unchanged.

**P1 — `PairingStore` port + co-located driver + conformance (unused yet).** `core/ports.py`:
`Pairing`, `PairingStore`. `store/pairings/sqlite.py` `SqlitePairings` (schema above; `add` one txn;
`search` reuses `bm25.py`; `document` computes display). Second impl for the ≥2-driver rule:
`InMemoryPairings` in the conformance suite; DuckDB (`store/pairings/duckdb.py`, DuckDB `fts`) as the
real config-swap driver (its FTS is index-rebuild, weaker atomicity — documented). `store/__init__.py`:
`PAIRING_STORES` registry, `Storage.pairings`, `[pairings]` binding. Tests: `test_pairings.py`
(lifecycle, **co-location invariant** — an aborted `add` leaves neither row nor FTS entry), extend
`test_conformance.py` + `test_load_storage.py`.

**P2 — Retrieval + import over the PairingStore.** `ingest/reference.py` importer: stream JSONL →
`PairingStore.add(...)` keeping `ref-<line>` ids (reuse `_corpus_items` numbering so citations, gold,
and `eval/retrieval.py` still line up), batched/resumable (floor = `count()`; lexical no longer needs
tail-trim); if a `[vector]` store is configured, `upsert` per batch then `reconcile(all_ids())` →
re-embed missing (**wires `reconcile`**). `cli/app.py`: build the retrievers from `storage.pairings`;
**keep the legacy `[lexical]`+`[documents]` path** so un-migrated recipes stay green. Tests mirror the
existing `test_on_disk_reference_index_is_built_once_and_reused`, plus lexical/dense/hybrid over pairings.

**P3 — `RunStore` (catalogue+journal+memory → DB), crash-safety preserved.** `core/ports.py`:
`RunStore`, `RunResult`, `RetrievedRef`. `store/run/sqlite.py` `SqliteRunStore` (WAL,
`synchronous` configurable default FULL, FK on). Rewrite `harness/runner.py` `run_batch` to take an
**injected** `RunStore` (runner imports the *port*; the driver is resolved in `cli`), keeping the
single writer thread + `_Interruptible` + duplicate grouping + `Progress` + `learn_memory`; replace
`write/flush/fsync` with `append_result`. Persist the P0 capture into `results.context_passage`/
`retrieved`. `records.py` gains `import_jsonl`/`export_jsonl` (reusing `write_catalog` atomicity);
CLI: `ragkit import` / `run` / `export` so **`recipes/*/eval.py`'s `read_journal` keeps working**.
Tests: kill-and-reopen ⇒ correct `completed_ids` (crash-safety), WAL recovery, later-wins via `seq`,
FK orphan guard, capture round-trip, `synchronous` branches; rewrite `test_runner.py` on a `RunStore`
fixture (resume, no-repeat, SIGINT, failure-after-append); conformance `RUN_FACTORIES` (+ InMemory).

**P4 — Write-back CLI + `Sink` + reconcile + `LexiconStore`.** `ragkit writeback`: read
`RunStore.results()`, filter `VERIFIED`/`is_injectable`, build `Pairing(source, context=context_passage,
target=output, verified=True, created_at=clock())` with a deterministic hash id (idempotent), `add`,
then `vector.reconcile(all_ids())`. **Implement the never-used `Sink` port** (`PairingSink.write`) as
the reinjection seam; write-back is its first caller. `LexiconStore` (`store/lexicon/sqlite.py`) +
`read_lexicon` stays as JSONL import; `relevant_entries` unchanged. Injected `clock`/hash keep tests
hermetic. Tests: verified-filter, idempotency, reconcile sync; `Sink` conformance; lexicon compat.

**P5 — Migrate the 6 recipes + tooling + docs.** Rewrite each `storage.toml`: the four retrieval
recipes → `[pairings] driver="sqlite" path="../data/<name>.pairings.db"` (replacing
`[lexical]`+`[documents]`); form_autofill/nl_to_sql keep their external read-only `[sql]`. Hardened
`tools/migrate_storage.sh` (+ src helper): fold `*.fts5`+`*.docs.db` → one `*.pairings.db`,
`lexicon.jsonl` → table, `records/journal.jsonl` → run DB. Update `docs/architecture.md` (§Storage:
two roles → three), `docs/config.md`, `recipes/*/README.md`, recipe tests. Optionally retire the
legacy split once all recipes are migrated.

**Ordering rationale:** P0 additive → P1 unused driver → P2 reference-to-pairings (legacy path kept)
→ P3 run-store (JSONL export bridges evals) → P4 write-back (needs P2 target + P3 capture) → P5 flip.
Legacy configs + JSONL export keep the suite green until the final phase.

## Migration / compatibility
- **JSONL becomes import/export only** (fetch scripts still emit JSONL; `ragkit import` loads once, streaming/resumable; `ragkit export` reproduces `journal.jsonl` so the six `eval.py` work untouched).
- **Existing on-disk indexes** folded by `tools/migrate_storage.sh`; **`ref-<line>` ids preserved** so citations/gold/`eval/retrieval.py` are unaffected.
- **`load_storage`** accepts `[pairings]`/`[run]` (new) and still `[lexical]`+`[documents]` (legacy) through P4 — no flag day.

## Risks + mitigations
1. **Journal-to-DB durability (highest).** Default `synchronous=FULL` (fsync per commit = today's guarantee); `NORMAL` a documented throughput option; keep the single writer thread + `_Interruptible`. SQLite atomic commit is **stronger** than torn-final-line tolerance — **prove it with a kill-and-reopen test**, don't assume.
2. **Vector sync** (can't co-locate). Pairings row is the floor; `reconcile(all_ids())` re-embeds any missing after import/write-back (now wired). RRF/MMR/rerank untouched.
3. **Capture semantics.** The capture-sink records exactly what `RetrievedBlock`/validators already retrieve (producer-time k/min_score) — no centralised re-retrieval that would change behaviour.
4. **Eval coupling to `read_journal`** — mitigated by `ragkit export`; migrating evals to `RunStore.results()` is deferred recipe churn.
5. **100% branch coverage** — every new branch (reconcile empty/non-empty, `synchronous` FULL/NORMAL, WAL recovery, verified/injectable filter, idempotent insert) needs a test.

## Verification
- `bash tools/run_tests.sh --coverage` green at **100% statement+branch** on `src/`; `bash tools/lint.sh` clean; `tests/test_boundaries.py` green (new drivers in `store/`; `core` stays stdlib-only; runner imports the *port*, not a driver).
- **Conformance**: `PairingStore` and `RunStore` each run ≥2 impls through identical operations; the co-location invariant and the RunStore crash-safety (kill-and-reopen) test are explicit.
- **Swap test**: a recipe retrieves identically after `[pairings] driver` sqlite↔duckdb, `git diff` over `src/` empty.
- **End-to-end**: each retrieval recipe still runs (`ragkit import` → `run` → `eval.py`); `ragkit writeback` inserts verified `(source, context, target)` and a subsequent run retrieves them (the accumulating-memory use case); the six `eval.py` pass via `ragkit export`.

### Critical files
`src/ragkit/core/ports.py` · `src/ragkit/store/__init__.py` · `src/ragkit/cli/app.py` ·
`src/ragkit/harness/runner.py` · `src/ragkit/core/records.py` ·
reuse anchors `src/ragkit/ingest/corpus.py`, `src/ragkit/store/lexical/fts5.py`,
`src/ragkit/harness/agents.py` · new drivers `src/ragkit/store/pairings/sqlite.py`,
`src/ragkit/store/run/sqlite.py`.
