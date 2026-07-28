# docs/

Design notes and guides.

| Document | Read it when |
|---|---|
| [`tutorial.md`](tutorial.md) | you want to **build something**: a new pipeline from scratch, and how to extend every seam (a validator, a context block, an output schema, a retriever, a database driver, a model backend) with runnable examples |
| [`architecture.md`](architecture.md) | you want the shape of the whole thing: the enforced layer boundaries, the one extension mechanism, the `Record` and its durability, where each LLM setting lives, and the evaluation layer |
| [`config.md`](config.md) | you need the exact configuration reference: every file, every key, its type, default, and meaning (wrong types and unknown keys are errors, not silent defaults) |

For runnable specifics, each recipe's `README.md` and its `config/` are closer to the code, and
`.audit/` holds the design investigations kept across sessions.
