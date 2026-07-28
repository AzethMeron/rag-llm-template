# 2026-07-28 — Cross-cutting sweep (recipes complete, eval harnesses landed)

What was checked: that the four worked recipes exercise the extension API without a core
change, that the two new cross-cutting requirements (configurable sampling; a decide-from-memory
recipe) landed cleanly, and that the generalised evaluation harnesses encode the hardest-won
lesson from `llm-translator`'s own audits rather than repeating it. Full suite green at 100%
statement+branch on `src/`, lint clean, boundary test green with `eval` added to the layer order.

## The circular-metric lesson, now enforced in code

The design rule behind `ragkit.eval` is a lesson carried from `llm-translator`: **a metric must be
independent of the thing it ranks.** There, a "consistency" metric defined ground truth as *what one
retrieval arm returns*, so that arm scored `1.000` by construction — and it *replicated cleanly
across two corpora*, which made it persuasive, not true. A number that is true by construction is
worse than no number, because it looks like evidence.

Decision: the eval harness **refuses to run** a circular configuration rather than warning about it.

- `evaluate_retrieval` raises `CircularEvaluationError` when the ground-truth `Qrels.source` names a
  system under evaluation, and again when any evaluated query has no gold judgments (scoring it
  would invent a number). `Qrels` cannot even be constructed without a `source` label, so the
  independence check always has something to test.
- `evaluate_ab` refuses to A/B a system against itself, and blinds the judge (neutral "Output 1/2"
  in an injected, deterministic order) so a position or provenance bias cannot leak in. A test
  asserts the system names never reach the judge's prompt.

This is the same shape as the other mechanical guards in the codebase (the model-pool thrash guard,
the generated-SQL safety validator, the grounded-decision validator): a failure the framework can
detect is made *impossible to configure into a silent success*, not left to a reviewer to notice.

## Configurable LLM settings — where each knob lives

Requirement added after M8: temperature and other decode settings must be configurable. Resolved by
placing each setting where it takes effect (see `docs/config.md`):

- **Per-request → the persona.** `SamplingParams` (frozen value type in `core/ports.py`) carries
  temperature and the sampling knobs; each `[[persona]]` sets its own `[persona.sampling]`. Default
  temperature is role-based (producer `0.3`, reviewer `0.0`), and every optional knob is unset
  unless named, so a strict-OpenAI endpoint never receives a llama.cpp-only field.
- **Launch-time → `models.toml`.** `[endpoint.<name>].server_args` holds verbatim server flags;
  `serve_models.sh --config` reads the same `EndpointSpec` the pool uses (`ragkit.llm.serveargs`),
  so `models.toml` is the single source of truth for routing *and* serving. `max_tokens` stays a
  computed budget, deliberately not a sampling knob.

No behaviour changed for existing recipes: the old hardcoded `0.3`/`0.0` temperatures are now the
role defaults.

## Recipe set — the abstraction held

Four recipes, each fully implemented, fetching real data on demand, none of which required a core
change; each proves a different facet of the extension API:

| Recipe | Proves | Mandatory guard |
|---|---|---|
| translation | reference-retrieval memory + a task validator plugin | untranslated-echo |
| nl_to_sql | the two-database split (read-only external source) + schema block | generated-SQL safety (single schema-bounded SELECT) |
| form_autofill | multi-field `FormSchema` + retrieval from a relational DB | field-value legality |
| predictive_maintenance | retrieval memory **and** a structured request together | grounded-decision (every citation found in the retrieved manuals) |

Framework additions the recipes motivated were all *generalisations of the module*, not recipe
special-cases: `FormSchema` (incl. an `array` field type), the `schema`/`readings` context blocks,
the retriever exposed to validators, and `models.toml` path resolution against the config dir. Each
is reusable by any recipe, which is the test that the abstraction was drawn in the right place.

## Retractions / notes

- Earlier draft placed sampling on the *model* (`models.toml`). Retracted: temperature is a property
  of the *role*, not the model — a producer and a reviewer sharing one model still want different
  temperatures — so it belongs on the persona. `models.toml` keeps only what a server applies at
  launch.
- The predictive-maintenance grounding check matches evidence by normalised substring against the
  *re-retrieved* passages (deterministic retriever, depth ≥ the block's), not by chunk id, to avoid
  coupling the recipe to how a block renders ids. Noted as a deliberate trade-off: robust to
  rendering, slightly fuzzier than id-equality.
