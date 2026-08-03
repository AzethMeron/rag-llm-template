# .audit/

A hidden but tracked directory holding the results of past audits — reviews, coverage
snapshots, and design investigations worth keeping across sessions. It is version-controlled
on purpose: an audit that is not in the repository is an audit nobody can find later.

Add an audit as a dated Markdown file (e.g. `2026-07-28-core-contract.md`) describing what
was checked, what was found, and what was decided. Findings are recorded as they came out,
including retractions — a wrong conclusion kept on the page, struck through, is more useful
than a deleted one.

## Entries

- `2026-07-28-cross-cutting-sweep.md` — a cross-cutting sweep of the codebase.
- `2026-08-02-deep-audit.md` — whole-codebase deep audit (7 subsystems): bugs, OOM/perf,
  concurrency/durability, contract violations, docs/staleness, SSOT, and interface swappability.
  1 critical, 3 high, 16 medium, 20 low, with a per-port swappability verdict — **all resolved**
  (each finding's status recorded in place in the file itself).
- `2026-08-03-doc-pass-findings.md` — findings surfaced while writing the exhaustive `docs/api/`
  reference (not a dedicated audit): 5 worth fixing (a `ZeroDivisionError` footgun in two eval
  metrics, an unwrapped error path in `reader_eval.py`, a `:memory:` introspector blind spot, an
  unenforced `Persona` invariant, a resource-ordering nit) plus several intentional/informational
  notes. **Resolved**: 4 fixed, 1 (`reader_eval.py`) re-assessed as intentional; status recorded
  per finding in the file.
