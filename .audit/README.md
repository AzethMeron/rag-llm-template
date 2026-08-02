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
  1 critical (broken `fetch_models.sh` repos), 2 high (silent rerank drop; batch poison pill),
  16 medium, 20 low; with a per-port swappability verdict.
