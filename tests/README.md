# tests

The framework's own test suite. It needs **no inference server, no GPU, and no network**: every
model/embedding/rerank call is served by a scripted fake or an in-memory `httpx.MockTransport`, and
every database (sqlite, duckdb, lancedb, qdrant) runs embedded/in-process against `tmp_path`.

```bash
tools/run_tests.sh              # or: PYTHONPATH=src python -m pytest
tools/run_tests.sh --coverage   # statement AND branch coverage, gated at 100% on src/
```

## Layout

| Path | What it covers |
|---|---|
| `core/` | the contract: records & durability, the config loaders, the component registry (all three discovery paths, unknown-key rejection incl. third-party components, protocol conformance, collisions), the lexicon, display width (property tests), placeholders, rule violations, sampling params, and the structured error base |
| `store/` | each driver's behaviour + failure modes, and a **conformance suite** that runs every implementation of each port (sql: sqlite+duckdb; vector: lancedb+qdrant+in-memory; pairings: sqlite+duckdb+in-memory; run: sqlite+in-memory; lexicon: sqlite+in-memory) through identical operations, plus a config dotted-path custom driver. `store/test_pairings.py` covers the co-located `PairingStore` (`SqlitePairings`/`DuckDBPairings`) — the row+FTS5-index atomicity invariant, search, idempotent `add`, and structured errors. `store/test_migrate.py` covers folding a pre-retirement recipe's on-disk artifacts into the new stores |
| `ingest/` | reference-corpus import into a `PairingStore` (`reference.py`); the extractors (text/JSONL/HTML/Markdown), the chunkers and their provenance (fixed/sentence/structure), text normalisation, and deduplication — independent, swappable building blocks for a raw-document ingest pipeline a user writes themselves — plus `from_config` paths and empty/edge inputs |
| `llm/` | the transport client (retry, error taxonomy, JSON-envelope recovery, RAII, a `UsageStats` concurrency-contention test), the backends, and the model pool + thrash guard |
| `harness/` | personas/panel/rules loading, validators, the context blocks + budgeter, memory, output schemas, and the produce→check→panel→revise runner |
| `retrieve/` | fusion (RRF/MMR property tests), rerank, embedding, the retrievers, and `retrieval.toml` parsing |
| `eval/` | the retrieval metrics (hand-verified) + circularity guard, and the blinded A/B judge |
| `cli/` | config-driven assembly, the pre-server checks, and the retrieval/injection seams |
| `test_boundaries.py` | the layer directions by AST walk; `core` is stdlib-only; each driver's third-party dep is confined to its own module; plus a subprocess that imports `core` with the optional deps unimportable |
| `recipes/*/tests/` | each recipe's own suite (against tiny committed fixtures, never the fetched datasets), including adversarial safety/grounding tests and end-to-end runs against **both** real DB drivers of each port |

## Conventions

- Tests are grouped into classes by behaviour, with descriptive method names.
- A docstring explains *why* a non-obvious case matters — several are regression tests, and
  the docstring records the failure.
- Filesystem tests use `tmp_path`; nothing writes outside it.
- Coverage is measured with **branches**, not statements alone: a 100% statement figure once
  hid four untaken branches, two of which were real defects. `src/` is at 100% statement and
  branch.
- Property/fuzz tests (hypothesis) name the *invariant* they defend rather than the
  implementation.
