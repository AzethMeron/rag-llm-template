# `ragkit.store` API Reference

`ragkit.store` holds every persistence port's concrete drivers behind `ragkit.core.ports`: the
relational store (`SqlStore`), the dense-vector index (`VectorIndex`), the co-located
reference-memory / pairing store (`PairingStore`), the append-only run-state store (`RunStore`),
the established-terminology lexicon (`LexiconStore`), and a read-only schema introspector
(`SchemaIntrospector`) — plus the shared filter-compilation (`filters.py`) and one-time-migration
(`migrate.py`) utilities every driver, or an upgrade off a legacy artifact, needs. Three database
roles are kept apart (see `docs/architecture.md`): the framework's own writable reference memory
(`pairings`/`lexicon`), its own writable run state (`run`), and an *external* task data source read
through `sql`/`introspector` and never written by the framework. Four ports currently ship two
interchangeable drivers each, selected purely by a `driver` key in `storage.toml`: `VectorIndex`
(LanceDB, the default, and Qdrant), and `SqlStore`, `PairingStore`, and `SchemaIntrospector`
(SQLite, the default, and DuckDB). `RunStore` and `LexiconStore` currently ship one driver each
(SQLite). A `ragkit.store.blob`
package exists in source (`src/ragkit/store/blob/__init__.py`) but is presently an empty stub — zero
bytes, no docstring, no code — a reserved namespace, not an implemented port; nothing in this
reference documents it beyond this note. Every third-party dependency (`lancedb`, `pyarrow`,
`qdrant_client`, `duckdb`) is imported lazily inside its own driver module, so importing
`ragkit.store` itself never pulls in a driver a given recipe does not use.

This package went through an audit + fix pass specifically around **driver-equivalence gaps**: two
drivers of the same port are not interchangeable in every respect, and each divergence below is
called out explicitly rather than assumed away. The two structural kinds of divergence that recur:
a capability one driver has and the other genuinely lacks (LanceDB refuses a metadata filter Qdrant
honours), and a scheduling/timing difference that is invisible to a correct caller but affects when
work happens (DuckDB's FTS index rebuild is deferred to the next `search`; SQLite's is incremental,
inside the same transaction as the write, via triggers).

---

## ragkit.store

The package root. Builds a `Storage` bundle from a `storage.toml` file, and hosts one `Registry`
(`ragkit.core.registry.Registry`) per port so a driver is selected by a `driver` string in config
rather than an import. Registers the built-in drivers under these registries and re-exports every
driver class and error type as this module's public API:

| Registry | `"sqlite"` | `"duckdb"` | `"lancedb"` | `"qdrant"` |
|---|---|---|---|---|
| `SQL_STORES` (`SqlStore`) | `SqliteStore` | `DuckDBStore` | — | — |
| `VECTOR_INDEXES` (`VectorIndex`) | — | — | `LanceVectorIndex` | `QdrantVectorIndex` |
| `PAIRING_STORES` (`PairingStore`) | `SqlitePairings` | `DuckDBPairings` | — | — |
| `RUN_STORES` (`RunStore`) | `SqliteRunStore` | — | — | — |
| `LEXICON_STORES` (`LexiconStore`) | `SqliteLexicon` | — | — | — |
| `SCHEMA_INTROSPECTORS` (`SchemaIntrospector`) | `SqliteIntrospector` | `DuckDBIntrospector` | — | — |

### Storage

`@dataclass(frozen=True, slots=True)`. The stores a run assembles from its `storage.toml`. Not a
driver of any port itself — it is the plain container `load_storage` returns. Any field may be
absent: a run with no reference memory has no `pairings`; a run with no external data source has no
`introspector`. `pairings` is called out because it is the DB-native reference memory (rows and
search index co-located in one store, per `PairingStore`'s contract), not because it differs
structurally from the other optional fields.

**Attributes:**
- `sql` (`SqlStore | None`, default `None`) — the relational store binding: the framework's own DB,
  or a read-only external task data source.
- `vector` (`VectorIndex | None`, default `None`) — the dense-vector index; absent for a
  lexical-only retrieval config.
- `pairings` (`PairingStore | None`, default `None`) — the co-located reference-memory store.
- `run` (`RunStore | None`, default `None`) — the record catalogue + result-history store.
- `lexicon` (`LexiconStore | None`, default `None`) — established-terminology store.
- `introspector` (`SchemaIntrospector | None`, default `None`) — schema reader for the NL→SQL
  feature.

#### load_storage(path: Path, *, base_dir: Path | None = None) -> Storage

Build the configured stores from a `storage.toml`. Each present `[<port>]` table names a `driver`
and passes its remaining keys as that driver's options; unknown keys are refused by the driver's
own `CONFIG_KEYS` schema (surfaced as `ConfigError`, not silently ignored).

**Args:**
- `path` (`Path`): path to the storage TOML file. Read via `ragkit.core.config.load_toml`.
- `base_dir` (`Path | None`, keyword-only): directory a relative `path` *option* inside any
  `[<port>]` table is resolved against; defaults to `path.parent`. Makes a config portable — its
  driver paths are relative to the config file, not to the caller's working directory. A literal
  `:memory:` path option is left untouched (never joined with `base_dir`).

**Returns:** `Storage` — one built instance per `[<port>]` table present in the file; `None` for
every port section absent from it.

**Raises:**
- `ConfigError` — the file fails to parse (from `load_toml`); a top-level key other than
  `sql`/`vector`/`pairings`/`run`/`lexicon`/`introspector` is present (from `reject_unknown`); a
  `[<port>]` section is present but is not a table (`"{label} must be a table"`); or a section is a
  table but has no non-blank `driver` string (`"{label} needs a 'driver'"`).
- `RegistryError` — propagated from `Registry.create` if `driver` names no registered/importable/
  entry-point component, or the resolved component's own option validation or port-conformance
  check fails.
- Whatever error a driver's own constructor raises for a bad option value (e.g. `VectorIndexError`
  for a non-positive `dim`) — `_build` does not catch driver-level errors, only shapes the config
  section.

**Side effects:** opens one live connection/handle per configured `[<port>]` table (a SQLite/DuckDB
connection, a LanceDB table handle, a Qdrant client) — the caller owns closing each store returned
in the `Storage` (no context-manager wraps the whole bundle).

---

## ragkit.store.filters

Compiles the framework's own backend-neutral metadata filter (`ragkit.core.ports.Filter`, a
conjunction of `Predicate` values) into a SQL `WHERE` dialect. SQLite and LanceDB both accept a
SQL-shaped predicate and share this compiler; Qdrant's driver (`ragkit.store.vector.qdrant`) has its
own compiler over the same `Filter`/`Predicate` AST rather than reusing this one, because Qdrant's
filter shape is a nested dict, not a SQL string. A driver that cannot honour a predicate refuses it
here, at compile time (the boundary), rather than returning wrong results at query time.

### FilterError

`class FilterError(RagkitError)`. Raised when a filter predicate cannot be compiled for a store — a
bad field identifier or an unsupported value type. Compile-time, not query-time, so a bad filter
never silently returns wrong rows. Carries no dedicated port (it is the shared error type
`to_sql` and each driver's own compiler raise); no fields beyond `RagkitError`'s `reason`/`context`.

#### to_sql(where: Filter) -> str

The `Filter` compiled to a SQL `WHERE` predicate, without the leading `WHERE` keyword.

**Args:**
- `where` (`Filter`, i.e. `tuple[Predicate, ...]`): the conjunction to compile. An empty tuple is
  valid input.

**Returns:** `str` — `""` for an empty filter (so a caller can append it unconditionally); otherwise
each predicate rendered and joined with `" AND "`. String values are single-quote-escaped; numeric
and boolean literals use `repr`/`"1"`/`"0"`. `FilterOp.IN` renders as `field IN (v1, v2, ...)`; the
other ops (`EQ`, `NE`, `LT`, `LE`, `GT`, `GE`) render as their SQL operator.

**Raises:**
- `FilterError` — a predicate's `field` is empty or contains a character that is not alphanumeric or
  `_` (field names reach the SQL string unparameterised, so they are validated as plain identifiers
  rather than escaped); an `IN` predicate's `value` is not a non-empty list/tuple; a predicate's
  `value` is a type `to_sql` cannot literalise (anything but `bool`/`int`/`float`/`str`); or (in
  principle only — every `FilterOp` member is covered) an unrecognised `FilterOp`.

**Side effects:** none — pure string compilation.

---

## ragkit.store.lexical.bm25

BM25-score normalisation and FTS5 query-escaping shared by every lexical pairing-store driver.
Extracted to one module so the score convention and query-escaping exist exactly once (SSOT): both
shipped `PairingStore` drivers — `SqlitePairings` (SQLite/FTS5) and `DuckDBPairings` (DuckDB/`fts`)
— map their engine's raw BM25 through `bm25_to_relevance`, so a ranking bug can only exist in one
place and a `min_score` keeps the same *shape* of meaning across a `[pairings].driver` swap. FTS5
query-escaping (`as_match`) is SQLite-only — DuckDB's `match_bm25` takes the query as a plain string,
with no query language to escape into.

#### bm25_to_relevance(raw: float) -> float

Map a raw, higher-is-better BM25 magnitude to a relevance in `[0, 1)`.

**Args:**
- `raw` (`float`): a BM25 magnitude that is `>= 0`, larger meaning a better match — the convention
  callers normalise *to*. DuckDB's `match_bm25` is already in this convention; SQLite FTS5's
  `bm25()` is negated (`<= 0`, more negative is better), so a caller passes `-bm25()`.

**Returns:** `float` — `raw / (1.0 + raw)`, monotone in `raw` so ranking and RRF fusion are
unaffected by the choice of transform, in `[0, 1)`.

**Raises:** none.

**Cross-driver note — what this does and does not make portable.** One shared transform means a
`min_score` keeps the same shape of meaning on either driver and ranking is identical, but it does
**not** make the absolute numbers equal: the two engines compute different raw BM25 (FTS5 and
DuckDB's `fts` differ in IDF handling), so a term present in every document scores `0.0` on SQLite
and `~0.19` on DuckDB for the same corpus. Converging further would mean reimplementing BM25 instead
of using each engine's native index — so re-check a `min_score` floor after a driver swap; do not
assume it transfers exactly. This rational form (`raw / (1 + raw)`) was chosen over the previously
used exponential form (SQLite used `1 - 2**bm25`, DuckDB used `s / (1 + s)` — two different
functions, invisible to a conformance suite that only asserted `score >= 0`) because it saturates
far more slowly: `1 - 2**-raw` is already `0.999` by `raw = 10`, collapsing every strong match onto
the same score, whereas the rational form still separates them.

#### as_match(query: str) -> str

Turn a free-text query into a SQLite FTS5 `MATCH` expression.

**Args:**
- `query` (`str`): a free-text query, arbitrary user input.

**Returns:** `str` — every whitespace-split word, individually double-quoted (with any embedded `"`
doubled) and OR-combined, e.g. `query="a \"b\" c"` → `"a" OR "\"\"b\"\"" OR "c"`. An empty/blank
query returns `'""'` (an empty quoted term, matching nothing) rather than an empty string, which
FTS5 would reject as a syntax error.

**Raises:** none.

**Side effects:** none — pure string transform. Quoting exists so punctuation in an ordinary query
(a quote, colon, or bare `AND`) cannot be read as FTS5 query syntax and raise on a legitimate user
query.

---

## ragkit.store.lexicon.sqlite

The established-terminology store over SQLite (WAL) — the DB-native replacement for a JSONL lexicon
file as the accumulating source of truth for a project's terminology. Kept as its own small table
rather than folded into the pairing store: a lexicon entry (`term`/`rendering`/`category` keyed on
`(term, category)`) is a different shape from a pairing (`source`/`target`/`context`). Pointing a
`[lexicon]` binding at the same file path as a recipe's `[pairings]` binding still puts both tables
in one physical database file; nothing in this module requires or prevents that.

### LexiconStoreError

`class LexiconStoreError(RagkitError)`. Raised when a lexicon-store operation fails, or the lexicon
database is misconfigured (e.g. an unopenable path). No fields beyond `RagkitError`'s.

### SqliteLexicon

Implements `LexiconStore`. One SQLite table (`lexicon`), WAL journal mode. `add` **upserts**, keyed
on `(term, category)`: re-importing a corrected rendering replaces the stale row. This is the
opposite idempotency rule from `PairingStore.add` (first-write-wins, existing rows untouched) —
deliberate, since a lexicon is a *current* mapping of term to rendering, not a log of distinct
produced facts a pairing store accumulates.

Not a dataclass (custom `__init__`); `CONFIG_KEYS = frozenset({"path"})`. Constructor: `path: str =
":memory:"`. Opens with `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=5000` for the same reason
`SqlitePairings` does (a reader must not fail on "database is locked" just because a writer's commit
is in flight elsewhere).

#### from_config(cls, options: Mapping[str, Any]) -> SqliteLexicon

**Args:** `options` (`Mapping[str, Any]`): `{"path": str}`. `path` is **required** (via
`read_required_path`) — a forgotten `path` is refused rather than silently defaulting to an ephemeral
`:memory:` store; an explicit `":memory:"` is still accepted.

**Returns:** `SqliteLexicon`.

**Raises:**
- `ConfigError` — `path` is absent or blank (via `read_required_path`, label `"[lexicon] store"`).
- `LexiconStoreError` — if the underlying `sqlite3.connect`/schema-creation call fails (wrapped from
  `sqlite3.Error`).

**Side effects:** opens a SQLite connection and creates the `lexicon` table (`CREATE TABLE IF NOT
EXISTS`) if absent; commits once.

#### entries(self) -> list[Entry]

**Args:** none.

**Returns:** `list[Entry]` — every stored `(term, rendering, category, entity_id)` row, as
`ragkit.core.lexicon.Entry` instances. No particular order guaranteed (plain `SELECT`, no
`ORDER BY`).

**Raises:** none directly (a `sqlite3.Error` here would propagate unwrapped — the read path has no
try/except, unlike `add`).

**Side effects:** acquires the instance lock for one read; no write.

#### add(self, entries: Iterable[Entry]) -> int

Add or update entries. Upserts via `INSERT OR REPLACE`, keyed on the table's `(term, category)`
primary key.

**Args:**
- `entries` (`Iterable[Entry]`): entries to add or update. An empty iterable is a valid no-op.

**Returns:** `int` — the number of rows the store's total count grew by (an update to an existing
`(term, category)` does not count as added; `_count_locked()` is diffed before/after the batch).

**Raises:** `LexiconStoreError` — wraps any `sqlite3.Error` from the `executemany`/commit; the
transaction is rolled back first (`_safe_rollback`, itself tolerant of a rollback failing against an
already-broken connection, so that failure cannot mask the original error).

**Side effects:** opens a write transaction (`executemany` + `commit`) covering the whole batch as
one unit.

#### close(self) -> None

**Args:** none. **Returns:** `None`. **Raises:** none (does not guard `sqlite3.Error` from
`.close()`). **Side effects:** closes the SQLite connection.

---

## ragkit.store.migrate

One-time migrations from legacy JSONL/split-store artifacts to the DB-native stores this project
now uses. The split store itself (a relational row table alongside a separate FTS5 index) has been
retired from the live framework — these three functions exist purely to fold artifacts a
pre-retirement recipe left on disk into the new stores. Each streams in batches so a multi-GB corpus
never sits fully in RAM, and none of them touch or delete the old artifact (the caller's safety net
until the new store is verified). `tools/migrate_storage.sh` is the hardened wrapper that actually
runs these against a recipe's data.

#### migrate_documents_to_pairings(documents_path: Path, pairing_store: PairingStore, *, batch_size: int = 5000, on_batch: Callable[[int], None] | None = None) -> int

Fold a legacy row store's rows (`docs(chunk_id, display, meta)` schema, read directly via
`sqlite3`, opened `mode=ro`) into `pairing_store` as **source-only** pairings — no target/context,
since a lexical reference entry never had either.

**Args:**
- `documents_path` (`Path`): path to the legacy row-store SQLite database.
- `pairing_store` (`PairingStore`): destination store; any driver.
- `batch_size` (`int`, keyword-only, default `5000`): rows per `pairing_store.add` call. Must be
  `>= 1`.
- `on_batch` (`Callable[[int], None] | None`, keyword-only): called after each committed batch with
  the destination's new total row count (`floor + total`, cheaply tracked rather than re-queried) —
  a progress signal for a multi-million-row migration run unsupervised.

**Returns:** `int` — the number of pairings actually added (excludes rows skipped as already
migrated).

**Raises:** `ValueError` — `batch_size < 1`.

**Side effects:** opens a read-only connection to `documents_path`; calls `pairing_store.add` (a
write) once per full batch and once more for a non-empty remainder; resumable — the floor is
`pairing_store.count()`, and because both the source table (`rowid` order) and destination
(`ref-<line>` numbering) preserve insertion order, skipping the first `floor` source rows continues
an interrupted migration rather than restarting it (the same resumability
`ragkit.ingest.reference.import_reference` uses).

#### migrate_lexicon_to_store(lexicon_path: Path, lexicon_store: LexiconStore) -> int

Import a JSONL lexicon into a `LexiconStore`.

**Args:**
- `lexicon_path` (`Path`): path to a JSONL lexicon file, read via
  `ragkit.core.lexicon.read_lexicon`.
- `lexicon_store` (`LexiconStore`): destination store; any driver.

**Returns:** `int` — rows `lexicon_store.add` reports as newly added.

**Raises:** none of its own — a missing `lexicon_path` is treated by `read_lexicon` as "no
established terms" (an empty iterable, not an error), so this is a no-op rather than a failure in
that case; whatever `read_lexicon`/`lexicon_store.add` raise for other failures (malformed JSONL,
store errors) propagates unwrapped.

**Side effects:** one `lexicon_store.add` call (a write transaction on the destination).

#### migrate_run_to_store(catalog_path: Path, journal_path: Path, run_store: RunStore) -> int

Fold a legacy catalogue + journal pair into a `RunStore`: add every catalogue record, then replay
the journal's results in write order.

**Args:**
- `catalog_path` (`Path`): legacy record catalogue, read via `ragkit.core.records.read_catalog`.
- `journal_path` (`Path`): legacy append-only result journal, read via
  `ragkit.core.records.read_journal`.
- `run_store` (`RunStore`): destination store; SQLite is currently the only driver.

**Returns:** `int` — the number of records `run_store.add_records` reports as newly added (the
catalogue count; journal replay's count is not separately returned).

**Raises:** none of its own — whatever `read_catalog`/`read_journal`/`run_store.add_records`/
`run_store.append_result` raise (e.g. `CatalogError`, or `RunStoreError` if a journal entry
references a record never added) propagates unwrapped.

**Side effects:** one `add_records` call, then one `append_result` call per journal line, each a
separate write transaction on `run_store`. The `RunStore`'s own later-wins-by-`seq` resolution then
reproduces exactly what the journal encoded (its last line for a record was the one that mattered).
Structured fields a `RunResult` can hold but the journal never recorded (violations, reviews,
captured context) are left at their defaults — there is nothing to migrate into them.

---

## ragkit.store.pairings.common

What every `PairingStore` driver shares: the structured error type and the display-text convention.
Kept in one place so two drivers of the same port cannot compute a hit's display text differently —
the swap property (`[pairings].driver` `sqlite` ↔ `duckdb` with identical retrieval behaviour)
depends on it.

### PairingStoreError

`class PairingStoreError(RagkitError)`. Raised when a pairing-store operation fails, or the pairings
database is misconfigured. No fields beyond `RagkitError`'s.

#### pairing_display(source: str, target: str) -> str

The text a retriever shows for a pairing.

**Args:**
- `source` (`str`): the pairing's input text.
- `target` (`str`): the pairing's paired output text; empty for a reference entry imported before
  it has a target.

**Returns:** `str` — `f"{source} -> {target}"` if `target` is non-empty, else `source` alone.

**Raises:** none.

---

## ragkit.store.pairings.duckdb

A second real `PairingStore` driver, over DuckDB + its `fts` extension — proves the store is
swappable by a config edit exactly like every other port.

**Headline cross-driver divergence: FTS index maintenance timing.** Unlike `SqlitePairings`'s FTS5
triggers (kept in sync incrementally, inside the same transaction as every write), DuckDB's `fts`
extension builds a **separate index structure that must be rebuilt wholesale**
(`PRAGMA create_fts_index(..., overwrite=1)`). Rebuilding it inside every `add()` made a batched load
quadratic — measured: a 7M-row corpus at 5k rows/batch re-indexed the whole growing table ~1,400
times. The rebuild is therefore **deferred**: a write only marks the index stale (`self._fts_stale =
True`), and `search()` rebuilds first if stale. A bulk load followed by queries pays one rebuild
instead of one per batch; `search()` still never observes a base-table row missing from the index,
so the port's contract (`search` reflects everything `add`ed) is unchanged — no caller has to know.
An alternating add/search workload still rebuilds per search (inherent to DuckDB's FTS design); the
realistic import-then-query shape is what this fixes. `get`/`document`/`count`/`all_ids` are
unaffected by any of this — they read the base table directly, never the search index.

Registered as `"duckdb"` in `PAIRING_STORES`. Not a dataclass; `CONFIG_KEYS = frozenset({"path"})`.
Constructor: `path: str = ":memory:"`. Requires the `duckdb` package and its `fts` extension
(installed/loaded lazily on first use, `INSTALL fts` attempted if `LOAD fts` fails).

### DuckDBPairings

Implements `PairingStore` (and, like `SqlitePairings`, also satisfies `SearchIndex` via `search`).

#### from_config(cls, options: Mapping[str, Any]) -> DuckDBPairings

**Args:** `options`: `{"path": str}`. `path` is **required** (via `read_required_path`) — a
forgotten `path` is refused rather than silently defaulting to an ephemeral `:memory:` store; an
explicit `":memory:"` is still accepted.

**Returns:** `DuckDBPairings`.

**Raises:**
- `ConfigError` — `path` is absent or blank (via `read_required_path`, label `"[pairings] store"`),
  raised before the `duckdb`/`fts` checks in `__init__`.
- `PairingStoreError` — the `duckdb` package is not installed; its `fts` extension cannot be
  installed/loaded (offline and not already cached); or the connection/schema-creation call fails
  for any other reason.

**Side effects:** opens a DuckDB connection, loads the `fts` extension, and runs `CREATE TABLE IF
NOT EXISTS pairings(...)`. **Does not build the FTS index here** — an existing on-disk store is not
re-indexed just to be opened, and a caller that only reads rows never pays for an index it does not
use; the FTS-stale flag starts `True` and the first `search()` builds it.

#### add(self, pairings: Iterable[Pairing]) -> int

Add pairings, one row per `Pairing`, idempotent on `chunk_id` (`ON CONFLICT DO NOTHING`).

**Args:** `pairings` (`Iterable[Pairing]`): pairings to add; an empty iterable is a no-op.

**Returns:** `int` — rows actually added (existing `chunk_id`s do not count and are left
untouched).

**Raises:** `PairingStoreError` — wraps any exception from the transaction.

**Side effects:** opens an **explicit** `BEGIN`/`COMMIT`/`ROLLBACK` transaction around the batch —
unlike `SqlitePairings`, which relies on `sqlite3`'s implicit transaction plus an explicit
`.commit()`. This is a genuine, verified divergence: DuckDB does **not** roll back a failed
`executemany` on its own (each row lands or fails independently), so without the explicit wrapper a
mid-batch failure would leave a partial write while still reporting a partial added-count — breaking
`PairingStore.add`'s "one transaction" contract that the SQLite driver honours for free. Also marks
the FTS index stale (rebuilt lazily by the next `search`, not here).

#### search(self, query: str, *, k: int) -> list[tuple[str, float]]

Keyword search via DuckDB's `fts_main_pairings.match_bm25`.

**Args:**
- `query` (`str`): free-text query, passed to `match_bm25` as a plain string (no query-language
  escaping needed or performed — divergent from SQLite's `as_match`, which must escape FTS5 syntax).
- `k` (`int`, keyword-only): max hits to return.

**Returns:** `list[tuple[str, float]]` — `(chunk_id, score)`, best-first (`ORDER BY score DESC`),
`score` in `[0, 1)` via `bm25_to_relevance`. Returns `[]` immediately (no query run) if `k <= 0` or
`query.strip()` is empty.

**Raises:** `PairingStoreError` — any FTS query failure, with `query` attached as context.

**Side effects:** rebuilds the FTS index first if the stale flag is set (`_rebuild_fts`, `PRAGMA
create_fts_index('pairings', 'chunk_id', 'source', 'context', 'target', stemmer='none',
stopwords='none', overwrite=1)`) — `stemmer='none', stopwords='none'` is a deliberate match to
SQLite FTS5's `unicode61` tokenizer default (which does neither), so the two drivers retrieve
identically on a config swap; DuckDB's own default (`porter` stemmer + English stopwords) would
silently drop ordinary words like "hello".

#### document(self, chunk_id: str) -> tuple[str, Mapping[str, Any]] | None

**Args:** `chunk_id` (`str`): id to look up.

**Returns:** `tuple[str, Mapping[str, Any]] | None` — `(pairing_display(source, target),
json.loads(meta))`, or `None` if `chunk_id` is absent.

**Raises:** none explicitly caught (a `duckdb` error here propagates unwrapped, unlike `add`/
`search`).

**Side effects:** one read; reads the base table, never the FTS index.

#### get(self, chunk_id: str) -> Pairing | None

**Args:** `chunk_id` (`str`).

**Returns:** `Pairing | None` — the full row (`chunk_id`, `source`, `target`, `context`, `meta`,
`verified`, `created_at`), or `None` if absent.

**Raises:** none explicitly caught.

**Side effects:** one read; base table only.

#### all_ids(self) -> Iterator[str]

**Args:** none.

**Returns:** `Iterator[str]` — every `chunk_id`, ordered by `chunk_id`, streamed via `fetchmany(1000)`
rather than materialised (a multi-million-row corpus must not be fully resident at once).

**Raises:** `PairingStoreError` — if the initial `cursor.execute` or any `fetchmany` call fails.

**Side effects:** uses `self._conn.cursor()` (a dedicated cursor), not `self._conn.execute()`
directly — a documented, verified DuckDB-specific hazard: `connection.execute()` returns the
connection itself and reuses one shared result-set state, so an unrelated `execute()` call from
another method while this generator is paused between yields would silently redirect a still-open
`fetchmany()` to the wrong result set. `con.cursor()` gives an independent result set, immune to
that. The instance lock is held only for one `fetchmany` call at a time, never across a `yield`, so
a slow consumer cannot starve other threads holding the lock indefinitely. **This is a real,
DuckDB-only concern** — `SqlitePairings.all_ids` does not need the analogous care because
`sqlite3.Connection.execute()` does not share result-set state across calls the same way.

#### count(self) -> int

**Args:** none. **Returns:** `int` — `SELECT count(*) FROM pairings`. **Raises:** none explicitly
caught. **Side effects:** one read.

#### close(self) -> None

**Args:** none. **Returns:** `None`. **Raises:** none explicitly caught. **Side effects:** closes the
DuckDB connection.

---

## ragkit.store.pairings.sink

The reinjection seam: a `Sink` (`ragkit.core.ports.Sink`) that writes accepted, produced records
into a `PairingStore` as new reference pairings — the mechanism that lets a run's outputs become a
later run's worked examples (write-back; see `ragkit.ingest.writeback` for the `RunStore`-reading
orchestration that is its first caller).

#### pairing_chunk_id(source: str, target: str, context: str) -> str

A deterministic id from a pairing's content.

**Args:** `source`, `target`, `context` (`str`): the triple to hash.

**Returns:** `str` — `f"wb-{sha1(source\x00target\x00context)[:16]}"`. Writing back the identical
triple twice therefore produces the same `chunk_id`, so `PairingStore.add`'s idempotency on a
duplicate `chunk_id` makes the second write-back a no-op rather than a growing duplicate — the same
guarantee `ragkit.core.records.make_record_id` gives a catalogue entry, for the same reason.

**Raises:** none.

### PairingSink

Implements `Sink`. Turns each injectable, non-empty-output record into a `Pairing` and adds it to a
`PairingStore`. `Sink.write` is a task-agnostic core port (`Iterable[Record]` only); the context a
record was produced with (needed for the pairing's `context` field) travels via
`record.meta["context_passage"]` (class constant `CONTEXT_META_KEY`) rather than a wider signature —
a caller that has it (write-back, reading a `RunResult`) stashes it there before calling `write`. Not
a dataclass; constructor: `__init__(self, pairing_store: PairingStore, *, clock: Callable[[], float]
= time.time)`.

#### write(self, records: Iterable[Record]) -> None

**Args:** `records` (`Iterable[Record]`): candidate records.

**Returns:** `None`.

**Raises:** none of its own; whatever `pairing_store.add` raises (e.g. `PairingStoreError`)
propagates unwrapped.

**Side effects:** filters to records where `record.status.is_injectable and record.output` is
truthy — a record missing `context_passage` in `meta`, one that is not injectable, or one with no
output, is silently skipped rather than written as a hollow pairing. Calls `pairing_store.add` once
with every accepted record turned into a `Pairing` (`verified=True`, `created_at=self._clock()`); no
call at all if nothing qualifies.

---

## ragkit.store.pairings.sqlite

The co-located pairing store over SQLite — rows and a BM25 search index in one database. An FTS5
**external-content** table (`pairings_fts`) mirrors `pairings` via `AFTER INSERT/UPDATE/DELETE`
triggers, so every write to a row and its search entry happens inside that very statement's implicit
transaction: there is no window where one exists without the other, and an aborted write leaves
neither behind. This is what the retired split store (a relational row table plus a separate FTS5
index, kept in sync only by write ordering) could only approximate.

Registered as `"sqlite"` in `PAIRING_STORES` (the default). Not a dataclass; `CONFIG_KEYS =
frozenset({"path", "tokenizer"})`. Constructor: `__init__(self, path: str = ":memory:", *,
tokenizer: str = "unicode61")`. Opens with `PRAGMA journal_mode=WAL` and `PRAGMA
busy_timeout=5000`.

**Known limitation, documented in source: reads are serialised**, so a retrieval-bound run does not
scale with `--concurrency`. One `sqlite3` connection is shared by every thread, guarded by one lock
(a connection is not safe for concurrent use); WAL means a reader is never blocked by a *writer*, but
the lock still blocks readers against each other. Measured on `legal_procurement` (7,097,288 rows,
2026-08-02): one BM25 query costs ~4.6s, throughput held at ~2.3 records/min with `--concurrency 4`
on a 24-core machine, GPU at 0% — every worker queued behind this lock, not the model. It bites only
where a single query is expensive (a very large FTS5 index); at `med_evidence` scale (597k rows) the
same run was model-bound and scaled with concurrency as expected. The documented fix, if it matters
for a given corpus: thread-local *read* connections (WAL supports many concurrent readers), keeping
the lock for writes only — noted as not applicable to a `:memory:` store, since each connection would
get its own empty database.

### SqlitePairings

Implements `PairingStore` (and `SearchIndex` via `search`, "by design", per the port's own
docstring).

#### from_config(cls, options: Mapping[str, Any]) -> SqlitePairings

**Args:** `options`: `{"path": str, "tokenizer": str}`. `path` is **required** (via
`read_required_path`) — a forgotten `path` is refused rather than silently defaulting to an ephemeral
`:memory:` store that loses everything between runs; an explicit `":memory:"` is still accepted.
`tokenizer` defaults to `"unicode61"`.

**Returns:** `SqlitePairings`.

**Raises:**
- `ConfigError` — `path` is absent or blank (via `read_required_path`, label `"[pairings] store"`);
  the message names the fix (set `path` to a file, or `":memory:"` explicitly for a deliberately
  ephemeral store).
- `PairingStoreError` — wraps a `sqlite3.Error` from opening the connection or running the schema
  script (table + FTS5 virtual table + triggers).

**Side effects:** opens a SQLite connection; runs the full schema script and commits once.

#### add(self, pairings: Iterable[Pairing]) -> int

**Args:** `pairings` (`Iterable[Pairing]`); empty iterable is a no-op.

**Returns:** `int` — rows actually added, via `INSERT OR IGNORE` keyed on the `chunk_id UNIQUE`
constraint (an existing `chunk_id` is left untouched, matching the port's stated idempotency).

**Raises:** `PairingStoreError` — wraps any `sqlite3.Error`; rolls back first.

**Side effects:** one implicit transaction (`executemany` + `.commit()`) covering the whole batch;
the `pairings_ai` trigger fires per inserted row, keeping `pairings_fts` in sync **incrementally,
inside this same transaction** — the direct counterpoint to `DuckDBPairings.add`'s deferred,
separately-triggered rebuild. No explicit `BEGIN`/`ROLLBACK` wrapper is needed here (unlike DuckDB)
because `sqlite3`'s own executemany-then-commit already gives all-or-nothing semantics for this
driver.

#### search(self, query: str, *, k: int) -> list[tuple[str, float]]

**Args:**
- `query` (`str`): free-text query, escaped via `as_match` into an FTS5 `MATCH` expression (every
  word quoted and OR-combined) before use — divergent from DuckDB's `search`, which passes `query`
  to `match_bm25` unescaped, since DuckDB's FTS has no query language to escape into.
- `k` (`int`, keyword-only): max hits.

**Returns:** `list[tuple[str, float]]` — `(chunk_id, score)`, best-first (`ORDER BY score` ascending
over FTS5's own negated `bm25()`, i.e. most-negative/best first), `score` via
`bm25_to_relevance(-score)` in `[0, 1)`. Returns `[]` immediately if `k <= 0` or `query.strip()` is
empty.

**Raises:** `PairingStoreError` — any `sqlite3.Error` from the FTS5 query, with `query` attached as
context.

**Side effects:** one read; **no index maintenance happens here** — unlike `DuckDBPairings.search`,
which may rebuild the whole FTS structure first. This is the concrete manifestation of the
headline divergence: SQLite's FTS5 index is always current by the time any `search()` runs (kept so
incrementally by the triggers on every `add`), so `search()` itself has nothing to do but query.

#### document(self, chunk_id: str) -> tuple[str, Mapping[str, Any]] | None

Same contract and return shape as `DuckDBPairings.document`. **Raises:** none explicitly caught.
**Side effects:** one read of the base `pairings` table.

#### get(self, chunk_id: str) -> Pairing | None

Same contract as `DuckDBPairings.get`. **Raises:** none explicitly caught. **Side effects:** one
read.

#### all_ids(self) -> Iterator[str]

**Args:** none. **Returns:** `Iterator[str]`, ordered by internal `id`, streamed via
`fetchmany(1000)`. **Raises:** `PairingStoreError` on any `sqlite3.Error` from the initial execute or
a `fetchmany`. **Side effects:** uses `self._conn.execute()` directly (no dedicated cursor needed —
unlike `DuckDBPairings.all_ids`, `sqlite3`'s cursor objects returned by `execute()` do not share
result-set state across unrelated calls on the same connection the way DuckDB's do). Lock held only
per-fetch, never across a `yield`.

#### count(self) -> int

Same contract as `DuckDBPairings.count`. **Side effects:** one read.

#### close(self) -> None

Same contract as `DuckDBPairings.close`. **Side effects:** closes the SQLite connection.

---

## ragkit.store.run.sqlite

The run-state store over SQLite — the record catalogue and append-only result history that replace
the JSONL catalogue+journal pair. WAL lets a long-lived `pending()` read run concurrently with the
single writer thread's `append_result()` calls without either blocking the other. `synchronous`
controls fsync discipline: `"FULL"` (default) fsyncs before returning, matching the JSONL journal's
per-line fsync durability; `"NORMAL"` is faster and still crash-safe under WAL (a commit can be
*lost* on power failure, never *corrupted*) — an explicit throughput option, not the default.
`PRAGMA foreign_keys=ON` means `append_result` for a record never added is refused **at the
database** (an `sqlite3.IntegrityError`), rather than discovered later the way the legacy
`merge_journal`'s orphan check used to.

Currently the **only** `RunStore` driver (no DuckDB equivalent registered) — no cross-driver
comparison applies to this port yet.

### RunStoreError

`class RunStoreError(RagkitError)`. Raised when a run-store operation fails, the run database is
misconfigured, or `append_result` is called for a record never added to the catalogue.

### SqliteRunStore

Implements `RunStore`. Not a dataclass; `CONFIG_KEYS = frozenset({"path", "synchronous"})`.
Constructor: `__init__(self, path: str = ":memory:", *, synchronous: str = "FULL", clock:
Callable[[], float] = time.time)` (`clock` is a testing seam, not exposed via `from_config`).

#### from_config(cls, options: Mapping[str, Any]) -> SqliteRunStore

**Args:** `options`: `{"path": str, "synchronous": str}`. `path` is **required** (via
`read_required_path`) — a forgotten `path` is refused rather than silently defaulting to an ephemeral
`:memory:` store; an explicit `":memory:"` is still accepted. `synchronous` defaults to `"FULL"`.

**Returns:** `SqliteRunStore`.

**Raises:**
- `ConfigError` — `path` is absent or blank (via `read_required_path`, label `"[run] store"`).
- `RunStoreError` — `synchronous.upper()` is not one of `{"FULL", "NORMAL"}`; or a `sqlite3.Error`
  from opening the connection/running the schema script.

**Side effects:** opens a connection with `journal_mode=WAL`, `busy_timeout=5000`,
`synchronous=<mode>`, `foreign_keys=ON`; runs the schema script (`records` + `results` tables and
their indexes) and commits once.

#### add_records(self, records: Iterable[Record]) -> int

**Args:** `records` (`Iterable[Record]`); empty iterable is a no-op.

**Returns:** `int` — rows actually added, via `INSERT OR IGNORE` keyed on `record_id` (idempotent —
re-running an import is safe).

**Raises:** `RunStoreError` — wraps any `sqlite3.Error`; rolls back first.

**Side effects:** one implicit transaction (`executemany` + `.commit()`).

#### append_result(self, result: RunResult) -> None

Persist one completed attempt as a new row in `results` — never overwrites an earlier attempt at
the same `record_id`.

**Args:** `result` (`RunResult`): the attempt to persist (`record`, `context_passage`, `retrieved`,
`reviews`, `violations`, `rounds`, `error`).

**Returns:** `None`.

**Raises:**
- `RunStoreError` — specifically for `sqlite3.IntegrityError` (the FK constraint: `record.record_id`
  was never added via `add_records`), with a message naming the exact fix (`"call add_records
  first"`); more generally for any other `sqlite3.Error`. Rolls back first in both cases.

**Side effects:** serialises `notes`/`violations`/`reviews`/`retrieved` to JSON; inserts one `results`
row (`seq` autoincrement) and commits — one transaction, so a torn/partial row is impossible (the
durability the JSONL journal only approximated with a per-line fsync).

#### completed_ids(self) -> set[str]

**Args:** none. **Returns:** `set[str]` — every `record_id` with at least one result (`SELECT
DISTINCT record_id FROM results`) — what a resumed run must skip. **Raises:** none explicitly
caught. **Side effects:** one read, fully materialised (not streamed — a `set`, not an iterator).

#### pending(self) -> Iterator[Record]

**Args:** none. **Returns:** `Iterator[Record]` — records with no result yet, ordered by
`(rel_path, line_no)` (stable catalogue order), streamed via `_stream_rows` (`fetchmany(1000)`, never
materialising the whole catalogue). **Raises:** `RunStoreError` if the query or any fetch fails.
**Side effects:** one streamed read.

#### results(self) -> Iterator[RunResult]

**Args:** none. **Returns:** `Iterator[RunResult]` — the *latest* result per record (via a
`max(seq)` join), each with the full per-attempt detail (retrieved chunk text, reviews, violations),
ordered by `seq`. **Raises:** `RunStoreError` on query/fetch failure. **Side effects:** one streamed
read.

#### latest_records(self) -> Iterator[Record]

Like `results()`, but yields the finished `Record` alone (`source`/`output`/`status`/`notes`/`meta`),
not the full `RunResult`.

**Args:** none.

**Returns:** `Iterator[Record]`, ordered by `seq`.

**Raises:** `RunStoreError` on query/fetch failure.

**Side effects:** one streamed read using a narrower query (`_LATEST_RECORDS_QUERY`) than `results()`
— deliberately does not select the wide per-attempt columns (`retrieved`, `reviews`, `violations`):
a caller that only needs what a record produced (e.g. seeding output memory, a plain export) does
not need them, and reading them into memory on every `ragkit run` invocation just to discard them was
measured as a genuine memory cost at scale.

#### count_records(self) -> int

**Args:** none. **Returns:** `int` — `SELECT count(*) FROM records`. **Raises:** none explicitly
caught. **Side effects:** one read.

#### close(self) -> None

**Args:** none. **Returns:** `None`. **Raises:** none explicitly caught. **Side effects:** closes the
connection.

---

## ragkit.store.sql.duckdb

DuckDB-backed relational store and schema introspector — a second real `SqlStore` driver, proving a
database is swappable by a config edit (`driver = "sqlite"` → `"duckdb"`) with no change to any other
part of the code. DuckDB is an embedded, in-process analytical SQL database (one self-contained
wheel, no server, MIT-licensed), keeping the zero-server self-contained property the SQLite default
has while being a genuinely different engine. The port is honoured identically to SQLite's: a
binding is tagged `read_only` at construction and a write through one is refused **at the port**; the
driver returns rows as plain dicts and normalises any error into the same `SqlStoreError` the SQLite
driver raises, so a caller cannot tell which engine raised. Thread-safe the same way: one connection
shared by the runner's concurrent workers, guarded by a lock (DuckDB's connection is not safe for
concurrent use either).

### DuckDBStore

Implements `SqlStore`. Not a dataclass; `CONFIG_KEYS = frozenset({"path", "read_only",
"schema_sql"})`. Constructor: `__init__(self, path: str = ":memory:", *, read_only: bool = False,
schema_sql: str | None = None)`. Attribute `read_only: bool` (the port's required member).

**Construction order — matched across drivers.** Both `DuckDBStore.__init__` and
`SqliteStore.__init__` (below) check the `schema_sql and read_only` contradiction and raise
**before** opening any connection (`"the error names the real mistake rather than a downstream open
failure"`, per each module's own comment), so a bad config never orphans an open connection handle on
a half-built instance. This once diverged — the SQLite driver used to open its connection first and
raise only afterward, transiently leaving an unclosed connection on the failed instance — which is why
the check's placement is still called out here; the two now agree.

#### from_config(cls, options: Mapping[str, Any]) -> DuckDBStore

**Args:** `options`: `{"path": str, "read_only": bool, "schema_sql": str | None}`; defaults
`":memory:"`, `False`, `None`.

**Returns:** `DuckDBStore`.

**Raises:** `SqlStoreError` — `schema_sql` set together with `read_only=True`
(`SqlStoreError.schema_on_read_only()`); the `duckdb` package is not installed; or the connect/
schema-execute call fails for another reason.

**Side effects:** opens a DuckDB connection — **read-only** at the OS/engine level if `read_only=True`
and `path != ":memory:"` (`duckdb.connect(path, read_only=True)`), matching `SqliteStore`'s rule that
`:memory:` cannot be opened read-only (it is ephemeral and per-connection). Runs `schema_sql` if
given (not possible simultaneously with `read_only`, checked above).

#### query(self, sql: str, params: Sequence[Any] = ()) -> list[Mapping[str, Any]]

**Args:** `sql` (`str`): a read statement. `params` (`Sequence[Any]`, default `()`): positional bind
parameters.

**Returns:** `list[Mapping[str, Any]]` — each row as a `dict` keyed by column name, built by zipping
`cursor.description` names against `fetchall()` rows (`zip(..., strict=True)`) — a different
mechanism from `SqliteStore.query`'s `dict(sqlite3.Row)`, but the same output shape.

**Raises:** `SqlStoreError` — wraps **any** `Exception` from the execute/fetch (deliberately broader
than `SqliteStore.query`'s `except sqlite3.Error`, per the module docstring: "normalise any duckdb
error into the port's `SqlStoreError`" — DuckDB's own exception hierarchy is less narrowly
documented/predictable than `sqlite3.Error`).

**Side effects:** acquires the lock for the query; no write regardless of `read_only` (read is always
allowed).

#### execute(self, sql: str, params: Sequence[Any] = ()) -> None

**Args:** same as `query`.

**Returns:** `None`.

**Raises:**
- `SqlStoreError.write_on_read_only(sql)` — if `self.read_only` is `True`. Checked **before**
  acquiring the lock, so a read-only binding never even attempts the call.
- `SqlStoreError` — wraps any other `Exception` from the execute call.

**Side effects:** runs `sql` under the lock. **No explicit `.commit()` call** — divergent in
mechanism from `SqliteStore.execute()`, which calls `self._conn.commit()` explicitly; functionally
equivalent because DuckDB auto-commits each statement run outside an explicit transaction block, so
a single `execute()` still persists immediately in both drivers.

#### close(self) -> None

**Args:** none. **Returns:** `None`. **Raises:** none explicitly caught. **Side effects:** closes the
DuckDB connection. Also usable as a context manager (`__enter__`/`__exit__` call `close()`).

### DuckDBIntrospector

Implements `SchemaIntrospector`, using `information_schema` so the NL→SQL feature reads a DuckDB
schema exactly as it reads a SQLite one. Not a dataclass; `CONFIG_KEYS = frozenset({"path"})`.
Constructor: `__init__(self, path: str)`.

#### from_config(cls, options: Mapping[str, Any]) -> DuckDBIntrospector

**Args:** `options`: `{"path": str}`, default `":memory:"`.

**Returns:** `DuckDBIntrospector`.

**Raises:** none of its own (construction is trivial — just stores `path`; failures surface from
`schema()`).

**Side effects:** none (no connection opened at construction — divergent from `SqliteIntrospector`,
which is likewise lazy but for the same reason).

#### schema(self) -> Mapping[str, Sequence[tuple[str, str]]]

**Args:** none.

**Returns:** `Mapping[str, Sequence[tuple[str, str]]]` — table name → its ordered `(column_name,
data_type)` pairs, read from `information_schema.tables`/`information_schema.columns` filtered to
`table_schema = 'main'`.

**Raises:**
- `SqlStoreError` — the `duckdb` package is not installed (via `_require_duckdb`, checked **first**,
  so a missing package surfaces before the `:memory:` check).
- `SqlStoreError.memory_introspection()` — `path == ":memory:"`: refused up front rather than opening
  a fresh, empty per-connection in-memory database and silently returning `{}` (see
  `SqlStoreError.memory_introspection`).
- whatever `duckdb.connect(..., read_only=True)` raises for a missing/invalid file (unwrapped — no
  explicit `try`/`except` around the connect in this method).

**Side effects:** opens its **own** short-lived read-only connection and closes it in a `finally`
(distinct from any `DuckDBStore` connection already open on the same path — necessary because DuckDB
refuses two connections to one file with different `read_only` settings, so this always opens
read-only to be compatible with a read-only `DuckDBStore` on the same file). No connection is opened
for a `:memory:` path — it is refused first (see Raises); `SqliteIntrospector.schema` refuses
`:memory:` identically, for the identical reason, so this is a property of `:memory:` databases
generally, not a DuckDB-specific gap.

---

## ragkit.store.sql.sqlite

SQLite-backed relational store and schema introspector — the zero-dependency default. Serves both
the framework's own record/metadata store and the read side of an external task data source (the
database NL→SQL queries, or form-autofill reads). A binding is tagged `read_only` at construction: a
write attempt through one is refused **at the port**, before the database — the first line of the
generated-SQL safety model, not a reliance on database permissions alone. Thread-safe: one
connection shared by the runner's concurrent workers, guarded by a lock, opened with
`check_same_thread=False`; reads serialise.

### SqlStoreError

`class SqlStoreError(RagkitError)`. Raised when a SQL store operation fails, a write is attempted
on a read-only binding, or a schema introspector is pointed at a `:memory:` database. The two
read-only refusals and the `:memory:`-introspection refusal are constructors here so **every**
`SqlStore` driver (SQLite, DuckDB) raises the identical message from one home rather than each
re-typing it — an SSOT for the error text, used by both `sql/sqlite.py` and `sql/duckdb.py`.

#### write_on_read_only(cls, sql: str) -> SqlStoreError

**Args:** `sql` (`str`): the statement that was refused, attached as context.

**Returns:** `SqlStoreError` — a constructed (not raised) instance with the message "write refused:
this SqlStore binding is read_only. An external data source is never written by the framework; only
the framework's own store is writable."

**Raises:** nothing itself (it constructs an exception; the caller raises it).

#### schema_on_read_only(cls) -> SqlStoreError

**Args:** none.

**Returns:** `SqlStoreError` — "a read_only store cannot run schema_sql".

**Raises:** nothing itself.

#### memory_introspection(cls) -> SqlStoreError

**Args:** none.

**Returns:** `SqlStoreError` — "a schema introspector cannot read a ':memory:' database: it is
per-connection and ephemeral, so the introspector's own connection sees an empty database. Point
[introspector].path at the real database file to introspect." Raised by both
`SqliteIntrospector.schema` and `DuckDBIntrospector.schema` when their `path` is `":memory:"`: a
`:memory:` database is per-connection, so an introspector opening its own connection would see a
*different*, empty database and silently return `{}` — the same silent-empty-schema failure the
missing-file guard already refuses, so it is refused too.

**Raises:** nothing itself.

### SqliteStore

Implements `SqlStore`. Not a dataclass; `CONFIG_KEYS = frozenset({"path", "read_only",
"schema_sql"})`. Constructor: `__init__(self, path: str = ":memory:", *, read_only: bool = False,
schema_sql: str | None = None)`. Attribute `read_only: bool`.

#### from_config(cls, options: Mapping[str, Any]) -> SqliteStore

**Args:** `options`: `{"path": str, "read_only": bool, "schema_sql": str | None}`; same defaults as
`DuckDBStore.from_config`.

**Returns:** `SqliteStore`.

**Raises:** `SqlStoreError.schema_on_read_only()` — raised (in `__init__`) **before** any connection
is opened, if both `read_only=True` and `schema_sql` are given — matching `DuckDBStore`; see the
construction-order note above.

**Side effects:** opens a connection — via `_connect_read_only(path)` (a `file:...?mode=ro` URI) if
`read_only`, else a normal read-write `sqlite3.connect`. `_connect_read_only` is also what makes a
missing file an error instead of silently creating one (a plain `sqlite3.connect` would create it).
Sets `row_factory = sqlite3.Row`. Runs `schema_sql` (`executescript` + commit) if given and not
read-only.

#### query(self, sql: str, params: Sequence[Any] = ()) -> list[Mapping[str, Any]]

**Args:** `sql` (`str`), `params` (`Sequence[Any]`, default `()`).

**Returns:** `list[Mapping[str, Any]]` — each row as `dict(sqlite3.Row)`.

**Raises:** `SqlStoreError` — wraps `sqlite3.Error` only (narrower than `DuckDBStore.query`'s bare
`Exception` catch).

**Side effects:** one read under the lock.

#### execute(self, sql: str, params: Sequence[Any] = ()) -> None

**Args:** same as `query`.

**Returns:** `None`.

**Raises:**
- `SqlStoreError.write_on_read_only(sql)` — if `self.read_only`.
- `SqlStoreError` — wraps `sqlite3.Error` from execute or commit.

**Side effects:** runs `sql` under the lock and calls `self._conn.commit()` **explicitly** — see the
`DuckDBStore.execute` note on the (functionally equivalent) mechanism difference.

#### close(self) -> None

**Args:** none. **Returns:** `None`. **Raises:** none explicitly caught. **Side effects:** closes the
connection. Also a context manager (`__enter__`/`__exit__` → `close()`), identical to `DuckDBStore`.

### SqliteIntrospector

Implements `SchemaIntrospector`, using `PRAGMA table_info` — no SQLAlchemy dependency. Not a
dataclass; `CONFIG_KEYS = frozenset({"path"})`. Constructor: `__init__(self, path: str)`.

#### from_config(cls, options: Mapping[str, Any]) -> SqliteIntrospector

**Args:** `options`: `{"path": str}`, default `":memory:"`.

**Returns:** `SqliteIntrospector`.

**Raises:** none of its own (construction just stores `path`).

**Side effects:** none.

#### schema(self) -> Mapping[str, Sequence[tuple[str, str]]]

**Args:** none.

**Returns:** `Mapping[str, Sequence[tuple[str, str]]]` — table name → ordered `(column name,
declared type)` pairs, for every table in `sqlite_master` not matching `sqlite_%` (excludes SQLite's
own internal tables), each read via `PRAGMA table_info("<table>")`.

**Raises:**
- `SqlStoreError.memory_introspection()` — `path == ":memory:"`, checked **first**, before any
  connection is opened: a `:memory:` database is per-connection, so this introspector's own
  connection would see a *different*, empty database and silently return `{}` (see
  `SqlStoreError.memory_introspection`).
- whatever `_connect_read_only` raises for a path that does not exist (unwrapped) — deliberate and
  documented: opening **read-only** is what makes a missing file an error at all. A plain
  `sqlite3.connect` would silently create the file it cannot find, so a typo'd `[introspector].path`
  used to produce an empty file and return `{}` with no error, and NL→SQL then generated against an
  empty schema; the DuckDB introspector already failed loudly on the same misconfiguration, and this
  fix brought the two into agreement — as does the shared `:memory:` refusal above.

**Side effects:** opens its own short-lived read-only connection, closed in a `finally` (none opened
for a `:memory:` path — refused first, see Raises). Table names from `sqlite_master` are
quote-escaped (`"` doubled) before being spliced into the `PRAGMA` string, since `PRAGMA` does not
accept a bound parameter for a table name — safe because the name comes from `sqlite_master`, not
user input. **Refuses `:memory:` identically to `DuckDBIntrospector`**: both raise
`SqlStoreError.memory_introspection` rather than opening a fresh, empty per-connection in-memory
database and silently returning `{}` — a property of `:memory:` databases generally, not a
driver-specific gap, and now named in each `schema()` docstring.

---

## ragkit.store.vector.common

What every `VectorIndex` driver shares: the structured error type and the `upsert` argument
contract. Kept in one place, like `ragkit.store.pairings.common`, so two drivers of the same port
cannot disagree about what a valid call looks like — the swap property (`[vector].driver` `lancedb`
↔ `qdrant` with identical behaviour) depends on the *rejections* matching, not only the successes.

### VectorIndexError

`class VectorIndexError(RagkitError)`. Raised when a vector-index operation fails, or the driver's
backend is unavailable (e.g. the third-party package is not installed).

#### reconcile_against(authoritative: Iterable[str], indexed: Iterator[str], delete: Callable[[Sequence[str]], None]) -> set[str]

Drop orphan vectors and report which authoritative ids the index is missing. Shared by both `Lance
VectorIndex.reconcile` and `QdrantVectorIndex.reconcile` — identical behaviour by construction, since
both simply call this function.

**Args:**
- `authoritative` (`Iterable[str]`): the ids that should be indexed (e.g. every `PairingStore.
  all_ids()`).
- `indexed` (`Iterator[str]`): the ids the index currently holds, consumed as a stream (not
  materialised) so a from-scratch reconcile of a multi-million-row corpus does not hold two full id
  sets in memory at once (each id either strikes itself off `authoritative` or is collected as an
  orphan candidate; what remains in the working set at the end is exactly what is missing).
- `delete` (`Callable[[Sequence[str]], None]`): the driver's own delete method, called **once**, with
  the orphan ids sorted, so the operation is deterministic.

**Returns:** `set[str]` — ids present in `authoritative` but not found in `indexed` (missing a
vector, for the caller to re-embed or refuse).

**Raises:** none of its own; whatever `delete` raises propagates.

**Side effects:** calls `delete(sorted(orphans))` exactly once, only if there is at least one orphan
(indexed ids no longer in `authoritative`).

#### validate_upsert(ids: Sequence[str], vectors: Sequence[Sequence[float]], metas: Sequence[Mapping[str, Any]], *, dim: int) -> None

Check one `upsert` call's arguments before any driver touches its backend, so a bad call is refused
at the port with an identical message whichever driver is configured.

**Args:**
- `ids` (`Sequence[str]`), `vectors` (`Sequence[Sequence[float]]`), `metas` (`Sequence[Mapping[str,
  Any]]`): the three parallel sequences an `upsert` call receives.
- `dim` (`int`, keyword-only): the index's fixed vector dimensionality.

**Returns:** `None`.

**Raises:** `VectorIndexError` —
- lengths of `ids`/`vectors`/`metas` do not all match;
- any vector's length `!= dim`;
- any `id` appears more than once in the batch (which vector should win is undefined for a repeated
  id). **Historical cross-driver note**: before this shared check existed, the two drivers resolved a
  same-batch duplicate id differently — Qdrant's deterministic point id made the *last* occurrence
  silently win, LanceDB's `merge_insert` refused it outright. Silently picking one was judged the
  worse half of that split (an id whose vector depends on batch order is unreproducible), so this
  shared function makes **both** drivers refuse now — this is resolved, not a current divergence.

**Side effects:** none — pure validation, called by both `LanceVectorIndex.upsert` and
`QdrantVectorIndex.upsert` before either touches its backend.

---

## ragkit.store.vector.lancedb

The default `VectorIndex` driver: LanceDB, a real embedded vector database with true ANN indexing,
on-disk columnar storage with versioning, metadata filtering, and hybrid search, all in-process with
no server. `lancedb` and `pyarrow` are imported lazily, inside this module only (vectors are plain Python lists, so no numpy), so the
core import path never pulls them. LanceDB's cosine metric returns a *distance* (`0` identical, `2`
opposite); the port promises a higher-is-better score toward `[0, 1]`, so `_distance_to_score`
converts it here.

Registered as `"lancedb"` in `VECTOR_INDEXES` (the default). Not a dataclass; `CONFIG_KEYS =
frozenset({"path", "table", "dim", "metric", "nprobes"})`. Constructor: `__init__(self, path: str, *,
table: str = "chunks", dim: int = 0, metric: str = "cosine", nprobes: int | None = None)`.

### LanceVectorIndex

Implements `VectorIndex`.

#### from_config(cls, options: Mapping[str, Any]) -> LanceVectorIndex

**Args:** `options`: `{"path": str, "table": str, "dim": int, "metric": str, "nprobes": int |
None}`; `path` is **required** (no default) — like the persistence-critical pairing/run/lexicon
`from_config`s (which require it via `read_required_path`), and unlike the `[sql]` stores,
introspectors, and the Qdrant driver, whose `from_config` still defaults `path` to `":memory:"`.

**Returns:** `LanceVectorIndex`.

**Raises:** `VectorIndexError` — `path` is empty/absent (`"the LanceDB vector index needs a
'path'"`); or (from `__init__`) `dim < 1`; or `nprobes is not None and nprobes < 1`.

**Side effects:** connects to the LanceDB database at `path` and opens (or creates, `exist_ok=True`)
the named table with a fixed 3-column schema (`id: string`, `vector: fixed-size list<float32, dim>`,
`meta: string`) — `exist_ok=True` is used instead of a `list_tables()` check because that check's
result can lag a just-written table across connections.

#### upsert(self, ids: Sequence[str], vectors: Sequence[Sequence[float]], metas: Sequence[Mapping[str, Any]]) -> None

Replace the rows for `ids` and insert the rest, as **one** transaction, via LanceDB's native
`merge_insert("id").when_matched_update_all().when_not_matched_insert_all()`.

**Args:** `ids`, `vectors`, `metas` — parallel sequences (see `validate_upsert`); `meta` values are
JSON-serialised into the `meta` column.

**Returns:** `None`.

**Raises:** `VectorIndexError` — from `validate_upsert` (mismatched lengths, wrong dim, duplicate id
in-batch); or wrapping any exception `merge_insert(...).execute(rows)` raises.

**Side effects:** one `merge_insert` call (no-op if `ids` is empty). Chosen over a delete-then-add
two-statement form specifically because that form left a crash window where old vectors were removed
without the new ones added, wrote two fragments per call instead of one (feeding `compact()`'s
cleanup burden), and had no structural defence against writing the same id twice — the exact shape of
this table's real 5,000-duplicate-row corruption incident (see `compact`). `merge_insert` joins on
`id`, so a matched row is updated in place, not deleted and re-added, and a duplicate cannot be
created via this path.

#### search(self, vector: Sequence[float], *, k: int, where: Filter = ()) -> list[tuple[str, float]]

**Args:**
- `vector` (`Sequence[float]`): query vector; must have length `== self._dim`.
- `k` (`int`, keyword-only): max hits.
- `where` (`Filter`, keyword-only, default `()`): compiled via `to_sql(_supported(where))`.

**Returns:** `list[tuple[str, float]]` — `(id, score)`, best-first, `score` via `_distance_to_score`
(cosine: `max(0, min(1, 1 - distance))`; other metrics, e.g. L2: `1 / (1 + max(0, distance))`).
Returns `[]` immediately if `k <= 0` or the table is empty (`count() == 0`) — no query issued in
either case.

**Raises:**
- `VectorIndexError` — `len(vector) != self._dim`; or (via `_supported`) `where` contains a predicate
  on any field other than `"id"` — **the headline capability divergence**: metadata here is stored as
  one opaque JSON column (`meta`), with no fixed schema (which keys exist varies per record) and no
  JSON-path filter LanceDB exposes to reach into it (checked directly: no `json_extract`/
  `get_json_object`), so there is nothing a metadata predicate could compile to. Qdrant's driver
  stores each metadata key as its own filterable payload field and **can** filter on it — this is a
  genuine, permanent capability difference between the two drivers, not a bug, and it is refused here
  at the boundary (compile time) with a message naming the fix (use `qdrant`), not discovered as a
  raw engine error mid-query the way it used to surface.

**Side effects:** one read; probes `self.search_nprobes(row_count)` partitions of the ANN index if
one exists (falls back to an exact O(n) scan silently if not — see `create_index`).

#### search_nprobes(self, row_count: int) -> int

How many IVF partitions a query probes.

**Args:** `row_count` (`int`): current table row count, used only to derive the default.

**Returns:** `int` — the configured `nprobes` if set at construction; else `max(_MIN_NPROBES,
ceil(_NPROBE_FRACTION * ivf_partitions(row_count)))` where `_NPROBE_FRACTION = 0.05` and
`_MIN_NPROBES = 20` (module constants). Query cost is therefore a roughly fixed *fraction* (5%) of
the corpus rather than a fixed count, deliberately uncapped (an upper bound would silently
reintroduce the recall cliff on the largest tables — the exact bug this exists to fix).

**Raises:** none.

**Side effects:** none (pure calculation). Documented rationale: LanceDB's own default probes a small
fixed number of partitions regardless of table size — fine for a handful of partitions, quietly lossy
for thousands (`create_index` builds `~sqrt(n)` partitions, so a 7.1M-row table has ~2,650 and the
built-in default reads well under 1% of it per query). The observable symptom was a Recall@20 drop on
`legal_procurement` right after its ANN index was first built. Set `nprobes` explicitly in `[vector]`
if `create_index` was given an explicit `num_partitions` — the default's fraction assumes both sides
derive from the same `~sqrt(n)` heuristic. Harmless when no index exists (an exact scan ignores it)
or when it exceeds the partition count (every partition probed, i.e. exact).

#### delete(self, ids: Sequence[str]) -> None

**Args:** `ids` (`Sequence[str]`); empty is a no-op.

**Returns:** `None`.

**Raises:** none explicitly caught (whatever `self._table.delete(...)` raises propagates unwrapped).

**Side effects:** one `DELETE`-style call with a manually quote-escaped `id IN (...)` predicate (`'`
doubled per value) — no parameterised-query API used here, unlike the SQL stores.

#### count(self) -> int

**Args:** none. **Returns:** `int` — `self._table.count_rows()`. **Raises:** none explicitly caught.
**Side effects:** none (read).

#### iter_indexed_ids(self) -> Iterator[str]

**Args:** none.

**Returns:** `Iterator[str]` — every indexed id, streamed via `search().select(["id"]).limit(None).
to_batches()`.

**Raises:** none explicitly caught.

**Side effects:** projects only the `id` column at the scan level (not `table.to_arrow()`, which
would materialise every column — the full vector plus the `meta` JSON — for every row just to
discard all but `id`; at millions of rows that would be tens of GB for one call) and streams batches
rather than building one giant in-memory `pyarrow.Table`.

#### reconcile(self, chunk_ids: Iterable[str]) -> set[str]

**Args:** `chunk_ids` (`Iterable[str]`): the authoritative id set.

**Returns:** `set[str]` — ids in `chunk_ids` missing a vector. Delegates entirely to
`reconcile_against(chunk_ids, self.iter_indexed_ids(), self.delete)` — identical implementation and
behaviour to `QdrantVectorIndex.reconcile`.

**Raises:** propagates whatever `reconcile_against`/`self.delete` raise.

**Side effects:** may call `self.delete` once, for any orphans found.

#### close(self) -> None

**Args:** none. **Returns:** `None` (always — the method body is `return None`). **Raises:** none.
**Side effects:** **none** — LanceDB holds no long-lived handle that needs explicit release for a
local table; this method exists purely for interface symmetry with the other stores. **Divergent
from `QdrantVectorIndex.close`**, which does call `self._client.close()`.

#### create_index(self, *, num_partitions: int | None = None, replace: bool = True) -> None

Build an IVF_FLAT ANN index on the `vector` column. **Not part of the `VectorIndex` port** — Qdrant
has no equivalent method (its HNSW index is built incrementally; there is nothing analogous to call).
A LanceDB-specific maintenance operation.

**Args:**
- `num_partitions` (`int | None`, keyword-only): defaults to `~sqrt(row_count)` (the standard IVF
  heuristic) if not given. Passing an explicit value here **breaks the nprobes/partition coupling**
  `search_nprobes` assumes (both derive from the same heuristic by default) — pair an explicit value
  here with an explicit `nprobes` in `[vector]`.
- `replace` (`bool`, keyword-only, default `True`): whether to replace an existing index.

**Returns:** `None`.

**Raises:** `VectorIndexError` — the table is empty (`row_count == 0`); or wraps any exception from
`self._table.create_index(...)`.

**Side effects:** imports `lancedb.index.IvfFlat` lazily; builds the ANN index (IVF_FLAT, not IVF_PQ
— IVF_FLAT needs no minimum training-set size and loses no precision to quantization, while IVF_PQ
requires ≥256 rows per partition to train and this store's smallest real tables, recipe unit tests,
have far fewer). Not called automatically on `upsert()` — training needs a representative sample of
already-written data and is itself expensive, so callers build it once after a corpus is mostly
loaded. **`search()` silently falls back to an O(n) brute-force scan of every row whenever no index
exists** — confirmed costly at scale: a 7.1M-row/1024-d table without an index made a 956-query eval
CPU-bound and multi-hour at 0% GPU use. **Must not be called while another process is concurrently
writing to the table** — same concurrent-access hazard as `compact()` (this table's real corruption
incident came from exactly this class of concurrent access).

#### compact(self) -> None

Consolidate small fragments left by many incremental `upsert()`/`delete()` calls (each is a separate
write transaction) into a few large ones, and prune old versions. Not part of the `VectorIndex` port
— Qdrant's HNSW index has no on-disk fragment-file model to compact. Called explicitly by
`tools/compact_vector_store.sh`, never on a read/write path.

**Args:** none. **Returns:** `None`.

**Raises:** `VectorIndexError` — `pylance` (a separate package from `lancedb`, needed for
`to_lance()`/`optimize()`) is not installed; or wraps any exception from `self._table.optimize(...)`.

**Side effects:** calls `self._table.optimize(cleanup_older_than=timedelta(0))` — confirmed
necessary, not speculative: `legal_procurement`'s table reached 3,717 versions / 1,858 fragments for
2M rows, at which point simply opening and reconciling against it (before embedding a single new row)
cost multiple GB of RSS per subsequent batch, because every read/write re-scans the whole fragment
list; compacting to 2 fragments fixed that immediately. **Confirmed unsafe against a concurrent
writer — this actually corrupted data once**: calling this (or even running two independent
`upsert()`-driven jobs) while a *different process* is mid-`upsert()`/`delete()` against the same
on-disk table silently produced 5,000 duplicate rows (same id, two rows each) with **no error
raised** — `count_rows()` was inflated by exactly that count, and `reconcile()`'s set-based
orphan/missing logic is blind to it (it compares distinct ids, not row counts). Only caught later by
an exact `pairings.count() == vector.count()` audit. Only call this from the same process that owns
the table's writes, or when no writer is active.

---

## ragkit.store.vector.qdrant

A second real `VectorIndex` driver: Qdrant, in embedded local mode (`QdrantClient(path=...)` /
`":memory:"`, in-process, no server) or pointed at a real Qdrant server via `url` — the plan names
Qdrant as the production tier too, using the same driver code. Ships so swapping the vector store is
a config edit (`driver = "lancedb"` → `"qdrant"`) with no other code change. `qdrant_client` is
imported lazily, inside this module only. Qdrant's cosine similarity is already the port's
higher-is-better convention (`1` identical); clamped into `[0, 1]` to match the other driver exactly.
A chunk id is a string, but a Qdrant point id must be an int or UUID — the chunk id is stored in the
payload (`_CHUNK_ID_FIELD = "_cid"`) and the point id is a deterministic UUID5 of it, so re-upserting
the same chunk id overwrites in place.

Registered as `"qdrant"` in `VECTOR_INDEXES`. Not a dataclass; `CONFIG_KEYS = frozenset({"path",
"url", "collection", "dim"})`. Constructor: `__init__(self, path: str = ":memory:", *, url: str = "",
collection: str = "chunks", dim: int = 0)`.

### QdrantVectorIndex

Implements `VectorIndex`.

#### from_config(cls, options: Mapping[str, Any]) -> QdrantVectorIndex

**Args:** `options`: `{"path": str, "url": str, "collection": str, "dim": int}`; defaults
`":memory:"`, `""`, `"chunks"`, `0`.

**Returns:** `QdrantVectorIndex`.

**Raises:** `VectorIndexError` — `dim < 1`; the `qdrant_client` package is not installed; or the
client/collection-creation call fails.

**Side effects:** if `url` is set, connects to a remote Qdrant server; else opens local mode — fully
in-memory if `path == ":memory:"`, on-disk otherwise. Creates the collection (cosine distance,
size=`dim`) if it does not already exist (`collection_exists` check first).

#### upsert(self, ids: Sequence[str], vectors: Sequence[Sequence[float]], metas: Sequence[Mapping[str, Any]]) -> None

**Args:** `ids`, `vectors`, `metas` — same contract as `LanceVectorIndex.upsert`, validated by the
same shared `validate_upsert`.

**Returns:** `None`.

**Raises:** `VectorIndexError` — from `validate_upsert` (identical rejections to LanceDB's).

**Side effects:** one `self._client.upsert(collection, points=[...])` call (no-op if `ids` is empty).
Each point's id is `uuid5(chunk_id)` (deterministic, so upserting the same chunk id again overwrites
the existing point rather than creating a duplicate — true upsert semantics across calls,
functionally equivalent to LanceDB's `merge_insert`-based upsert despite the different mechanism);
payload is `{**meta, "_cid": chunk_id}`.

#### search(self, vector: Sequence[float], *, k: int, where: Filter = ()) -> list[tuple[str, float]]

**Args:**
- `vector` (`Sequence[float]`): must have length `== self._dim`.
- `k` (`int`, keyword-only).
- `where` (`Filter`, keyword-only, default `()`): compiled by this module's own `_to_filter`/
  `_condition` (not `ragkit.store.filters.to_sql` — Qdrant's filter shape is a nested
  `models.Filter`/`FieldCondition` object graph, not a SQL string).

**Returns:** `list[tuple[str, float]]` — `(chunk_id, score)`, best-first, `score` clamped into `[0,
1]` from Qdrant's own similarity score. Returns `[]` immediately if `k <= 0` or `self.count() == 0`.

**Raises:** `VectorIndexError` — `len(vector) != self._dim`; or, from `_condition`, an `IN` predicate
whose `value` is not a non-empty list/tuple. **Divergence from LanceDB**: `where` here **is honoured**
for any field, including arbitrary metadata keys — `EQ`/`NE` compile to `FieldCondition(match=
MatchValue(...))` (`NE` is placed in the filter's `must_not`, everything else in `must`), `IN` to
`MatchAny`, and `LT`/`LE`/`GT`/`GE` to a `Range`. The special case: a predicate on the port's
canonical `"id"` field is translated to the reserved `_cid` payload key (`_CHUNK_ID_FIELD`) before
compiling — without this translation, `Predicate("id", EQ, ...)` would match nothing on Qdrant (the
real column LanceDB filters on is named `id`, but Qdrant has no such column; the chunk id lives only
in the payload).

**Side effects:** one `query_points` call with `with_payload=True`.

#### delete(self, ids: Sequence[str]) -> None

**Args:** `ids` (`Sequence[str]`); empty is a no-op.

**Returns:** `None`.

**Raises:** none explicitly caught.

**Side effects:** one `self._client.delete(...)` call with a `FilterSelector` matching `_cid IN
(ids)` — a parameterised filter object, unlike LanceDB's manually quote-escaped SQL string for the
same operation.

#### count(self) -> int

**Args:** none. **Returns:** `int` — `self._client.count(collection).count`. **Raises:** none
explicitly caught. **Side effects:** none (read).

#### iter_indexed_ids(self) -> Iterator[str]

**Args:** none.

**Returns:** `Iterator[str]` — every indexed chunk id, read from each point's `_cid` payload field,
paginated via `scroll(limit=256, offset=..., with_vectors=False)` until `offset is None`.

**Raises:** none explicitly caught.

**Side effects:** `with_vectors=False` — the actual vector data is never fetched for this listing,
analogous in intent to LanceDB's `select(["id"])` projection, though the mechanism (pagination
offset vs. columnar projection) differs because the two clients expose different APIs.

#### reconcile(self, chunk_ids: Iterable[str]) -> set[str]

**Args:** `chunk_ids` (`Iterable[str]`).

**Returns:** `set[str]` — same contract as `LanceVectorIndex.reconcile`; delegates to the identical
shared `reconcile_against(chunk_ids, self.iter_indexed_ids(), self.delete)` helper, so behaviour is
identical between the two drivers here (both are correctness-equivalent for this method — only the
underlying `iter_indexed_ids`/`delete` mechanisms differ).

**Raises:** propagates whatever `reconcile_against`/`self.delete` raise.

**Side effects:** may call `self.delete` once for any orphans.

#### close(self) -> None

**Args:** none. **Returns:** `None`. **Raises:** none explicitly caught. **Side effects:** calls
`self._client.close()` — **divergent from `LanceVectorIndex.close`**, which is a documented no-op;
Qdrant's client holds a real handle (a local-mode file lock, or a network connection to a remote
server) that this releases.
