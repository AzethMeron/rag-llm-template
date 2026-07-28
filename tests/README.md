# tests

The framework's own test suite. It needs **no inference server, no GPU, and no network**:
every model call (once the model layer lands) is served by a scripted fake or an in-memory
HTTP transport, and storage drivers are tested against `tmp_path`.

```bash
tools/run_tests.sh              # or: PYTHONPATH=src python -m pytest
tools/run_tests.sh --coverage   # statement AND branch coverage
```

## Layout

| Path | What it covers |
|---|---|
| `core/` | the contract: records & durability, the config loaders, the component registry (all three discovery paths, unknown-key rejection, conformance, collisions), the lexicon, display width (property tests), placeholders, rule violations, and the structured error base |
| `test_boundaries.py` | the layer directions by AST walk; `core` is stdlib-only; no module imports a concrete driver; plus a subprocess that imports `core` with the optional deps unimportable |

As more layers land, their suites join here: durability under injected failures, error-mode
exit behaviour, concurrency under real contention, and each recipe's own suite (against tiny
committed fixtures, never the fetched datasets).

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
