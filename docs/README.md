# docs/

Design notes and guides.

| Document | Read it when |
|---|---|
| [`tutorial.md`](tutorial.md) | you want to **build something**: the pipeline end to end (ingest → memory → context → the produce/review/revise loop → evaluation), a new recipe from scratch, every storage and retrieval option with when to pick it, and how to extend every seam — all with runnable examples |
| [`architecture.md`](architecture.md) | you want the shape of the whole thing: the enforced layer boundaries, the one extension mechanism, the `Record` and its durability, where each LLM setting lives, and the evaluation layer |
| [`config.md`](config.md) | you need the exact configuration reference: every file, every key, its type, default, and meaning (wrong types and unknown keys are errors, not silent defaults) |

For runnable specifics, each recipe's `README.md` and its `config/` are closer to the code, and
`.audit/` holds the design investigations kept across sessions.
