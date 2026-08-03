# API reference

Exhaustive, per-symbol reference for every public module, class, method, and function in the
framework and its example recipes: real signatures, argument types and meaning, return types and
meaning, side effects, and exceptions raised — verified against the source, not inferred from
names. This is the reference layer; for narrative, runnable-example documentation see
[`../tutorial.md`](../tutorial.md), and for the enforced architecture see
[`../architecture.md`](../architecture.md).

| File | Package | Covers |
|---|---|---|
| [`core.md`](core.md) | `ragkit.core` | The dependency-free foundation: config loading, errors, JSON-shape checks, the lexicon, placeholders, every port `Protocol` (`VectorIndex`, `SqlStore`, `PairingStore`, `RunStore`, `Retriever`, `Backend`, `Provider`, …), records/catalogue durability, the driver registry, rule violations, display-width math. |
| [`llm.md`](llm.md) | `ragkit.llm` | The model-serving/request layer: backends (schema-constrained vs. prompt-described JSON), providers (endpoint kinds — capabilities and request quirks), the HTTP client and its retry/error taxonomy, the model pool (thrash guard, connection reuse), `llama-server` launch-flag rendering. |
| [`store.md`](store.md) | `ragkit.store` | Every persistence port's concrete drivers: `VectorIndex` (LanceDB, Qdrant), `SqlStore` (SQLite, DuckDB), `PairingStore` (SQLite, DuckDB), `RunStore` (SQLite), `LexiconStore` (SQLite), BM25 lexical scoring, metadata filter compilation, legacy-artifact migration. Documents where driver pairs genuinely diverge, not just their shared contract. |
| [`harness.md`](harness.md) | `ragkit.harness` | The orchestration core: personas/panel, the produce → review → revise loop and its exception containment, context assembly (all 9 block kinds), mechanical validators, output schemas, output memory. |
| [`ingest.md`](ingest.md) | `ragkit.ingest` | Turning a raw source into indexable chunks (extractors, chunkers, dedup, normalisation) and importing reference memory (the one-time import → persisted-store distinction, resumable/idempotent). |
| [`retrieve.md`](retrieve.md) | `ragkit.retrieve` | The lexical/dense/hybrid retrieval stack: embedding client, RRF fusion, MMR, reranking, and the retrieval-settings loader. |
| [`eval.md`](eval.md) | `ragkit.eval` | Retrieval-quality metrics (recall/MRR/NDCG/hit-rate), the circularity guard, classification scoring, gold-file loading, and the blinded A/B judge. |
| [`cli.md`](cli.md) | `ragkit.cli` | The four CLI subcommands (import/run/export/writeback) and `assemble()`, which wires a full recipe from its config directory. |
| [`recipes.md`](recipes.md) | `recipes.*` | The 6 worked-example recipes: every custom `Validator` and every recipe's `eval.py`/`reader_eval.py` scoring code. |

Every entry above was produced by first enumerating every public symbol via an AST walk (so
coverage is exhaustive, not sampled), then reading the real source for each one — no docstring or
signature was guessed from a name alone. Where two drivers of the same port behave differently in
a way that matters to a caller, that divergence is called out explicitly rather than papered over.
