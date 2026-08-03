# Recipes API reference

The six recipes under `recipes/` are worked examples of the framework's extension seams, not
toys: each wires a real, runnable task — a music-database autofill, a Polish legal-passage
retriever, a biomedical evidence decider, a natural-language-to-SQL generator, a
predictive-maintenance advisor, and a faithful port of an existing translation pipeline — over
the same core (`ragkit.harness`, `ragkit.store`, `ragkit.retrieve`). What each recipe actually
*adds* to the framework is two things: one or more `plugins/validators.py` classes satisfying
`ragkit.core.ports.Validator` (a mechanical, code-decided check registered into the recipe's
`rules.toml`), and a recipe-specific `eval.py` (plus, for `med_evidence`, an additional
`reader_eval.py`) that scores a completed run against held-out gold data. Every recipe's `eval.py`
shares its gold-loading and reporting scaffolding with `ragkit.eval.gold` and, for the four
recipes that predict a single label, `ragkit.eval.classify.ClassificationReport` — but the field
each recipe scores, and which extra rates it reports, are genuinely per-recipe and documented
separately below.

## form_autofill

Fills the missing `genre` and `unit_price` fields of a music-track record from other tracks on
the same album, retrieved live from a real relational database.

### Outcome

`recipes/form_autofill/eval.py` — a frozen, slotted dataclass: one evaluated track's fill outcome.
Not a Protocol implementation (a plain value type).

**Attributes:**
- `record_id: str`
- `produced: bool` — a form was produced for this record (not rejected/pending).
- `genre_ok: bool` — the produced `genre` matched gold (case-insensitive, trimmed).
- `price_ok: bool` — the produced `unit_price` matched gold numerically within `0.005`.

#### both_ok(self) -> bool

Property. True iff both `genre_ok` and `price_ok` are true.

**Args:** none.
**Returns:** `bool`.
**Raises:** nothing.
**Side effects:** none (pure property read).

### Report

`recipes/form_autofill/eval.py` — aggregate over a run's `Outcome`s. Explicitly **not** a
`ClassificationReport`: this recipe scores *several* fields per record, each with its own
comparison (a case-insensitive string, a numeric tolerance), so there is no single predicted
label to count. A plain frozen dataclass, no Protocol implemented.

**Attributes:**
- `outcomes: tuple[Outcome, ...]`

#### total(self) -> int

Property. `len(self.outcomes)`.

**Args:** none. **Returns:** `int`. **Raises:** nothing. **Side effects:** none.

#### produced(self) -> int

Property. Count of outcomes with `produced=True`.

**Args:** none. **Returns:** `int`. **Raises:** nothing. **Side effects:** none.

#### genre_accuracy(self) -> float

Property. Fraction of outcomes with `genre_ok=True`, over `total`; `0.0` if `total` is 0.

**Args:** none. **Returns:** `float` in `[0, 1]`. **Raises:** nothing. **Side effects:** none.

#### price_accuracy(self) -> float

Property. Fraction of outcomes with `price_ok=True`, over `total`; `0.0` if `total` is 0.

**Args:** none. **Returns:** `float` in `[0, 1]`. **Raises:** nothing. **Side effects:** none.

#### both_accuracy(self) -> float

Property. Fraction of outcomes with `both_ok=True` (both fields correct), over `total`; `0.0` if
`total` is 0.

**Args:** none. **Returns:** `float` in `[0, 1]`. **Raises:** nothing. **Side effects:** none.

#### evaluate(pairs: Iterable[Pair]) -> Report

`recipes/form_autofill/eval.py`. Scores `(record_id, produced_output_or_None, gold_fields)`
triples (`Pair` from `ragkit.eval.gold`) — `produced_output` is the record's output string
(canonical JSON matching the form schema), or `None` when nothing usable was produced. Parses the
produced JSON defensively (a parse failure or non-object is treated as an empty form, not a
crash); compares `genre` case-insensitively after trimming and `unit_price` numerically within a
`0.005` tolerance, only when both produced and gold values are the right type (a bool is never
treated as a valid number, even though `bool` is an `int` subclass in Python).

**Args:** `pairs` — iterable of `(record_id, produced: str | None, gold: object)`.
**Returns:** `Report` over one `Outcome` per pair.
**Raises:** nothing (a malformed produced value never raises; it just fails the field check).

#### main(argv: Sequence[str] | None = None) -> int

`recipes/form_autofill/eval.py` CLI entry point. Parses `--journal` and `--gold`
(`ragkit.eval.gold.journal_gold_parser`), loads gold via `load_fields_gold(args.gold,
fields=("genre", "unit_price"))`, joins it against the journal, scores it, and prints
`"filled P/T | genre G | price Pr | both B"`.

**Args:** `argv` — CLI arguments, or `None` to use `sys.argv[1:]` (via `argparse`).
**Returns:** `0` on success, `1` if gold loading/parsing raises `EvalError`.
**Raises:** nothing — `EvalError` is caught by `ragkit.eval.gold.run_report` and reported on
stderr instead of propagating.
**Side effects:** reads the gold file and the run journal from disk; prints one line to stdout
(or an `error: ...` line to stderr).

### FieldRule

`recipes/form_autofill/plugins/validators.py` — a frozen, slotted dataclass: the value contract
for one form field (used only internally by `FieldTypesValidator`, not itself a `Validator`).
`__post_init__` raises `ValueError` if `type` is not one of `string`/`integer`/`number`/`boolean`.

**Attributes:**
- `name: str`
- `type: str` — one of `string`, `integer`, `number`, `boolean`.
- `nonempty: bool = False` — a string must have non-whitespace content.
- `minimum: float | None = None` — inclusive lower bound.
- `maximum: float | None = None` — inclusive upper bound.
- `min_exclusive: float | None = None` — exclusive lower bound (e.g. a price must be `> 0`).
- `enum: tuple[str, ...] = ()` — allowed values, matched case-insensitively for strings.

#### check(self, value: Any) -> list[Violation]

Checks one field's value against this rule's constraints, in order: `None` → not filled; wrong
JSON type (via `ragkit.core.jsonshape.json_type_matches`) → type mismatch; then, only for a
value of the declared type, blank-string, range, and enum checks (each independent — multiple
can fire on one call).

**Args:** `value: Any` — the field's parsed JSON value (or `None` if absent).
**Returns:** `list[Violation]`, each `Severity.ERROR` with rule id `"form_field"`; empty if the
value satisfies every configured constraint.
**Raises:** nothing.
**Side effects:** none.

### FieldTypesValidator

`recipes/form_autofill/plugins/validators.py`. Implements `ragkit.core.ports.Validator`. Checks a
filled form's field *values* against their declared per-field constraints — the contract the
output schema's JSON-shape check cannot express (a field present and correctly typed, but null,
blank, out of range, or not in an enum). Raises `ValueError` from `__init__` if constructed with
no `FieldRule`s.

**Attributes:**
- `CONFIG_KEYS = frozenset({"field"})` — the only recipe-config key the registry accepts for
  this validator; an unknown key is refused before `from_config` runs
  (`ragkit.core.registry.Registry.create`).

#### from_config(cls, options: Mapping[str, Any]) -> FieldTypesValidator

Builds one `FieldRule` per entry of `options["field"]`. Each entry needs at least a `name`;
`type` defaults to `"string"`, `nonempty` to `False`, `enum` to `()`, and the numeric bounds
(`min`, `max`, `min_exclusive`) are optional and parsed via a helper that rejects a non-numeric or
boolean value.

**Args:** `options` — recipe config mapping; must contain a non-empty `"field"` list of rule
dicts.
**Returns:** `FieldTypesValidator`.
**Raises:** `ValueError` if `"field"` is missing, not a list, or empty; if any entry lacks
`"name"`; if any numeric constraint is not a number; or (via `FieldRule.__post_init__`/
`FieldTypesValidator.__init__`) if a rule's `type` is invalid or no rules resulted.
**Side effects:** none.

#### validate(self, record: Record, output: str, context: Mapping[str, Any]) -> list[Violation]

Parses `output` as JSON and checks every configured field against its `FieldRule`. `record` and
`context` are unused (present only to satisfy the `Validator` port's shape).

**Args:** `record: Record` (unused), `output: str` — the filled form's JSON text, `context:
Mapping[str, Any]` (unused).
**Returns:** `list[Violation]` — one `Severity.ERROR` violation (rule id `"form_field"`) if
`output` is not valid JSON or not a JSON object; otherwise the concatenation of every field's
`FieldRule.check` violations. Empty means the form passed.
**Raises:** nothing — a JSON parse failure is returned as a violation, never raised.
**Side effects:** none.

## legal_procurement

Answers a Polish legal or public-procurement question from a corpus of legal passages, citing the
retrieved passage ids it used or abstaining (`"brak podstaw"`) when nothing supports an answer;
refuses any answer that cites a passage it never actually retrieved.

#### evaluate(retriever: Retriever, queries: Sequence[tuple[str, str]], gold: Mapping[str, frozenset[str]], \*, k: int = 20) -> RetrievalScores

`recipes/legal_procurement/eval.py`. Scores the recipe's own retriever (built the same way the
recipe builds it, via `assemble(config).retriever`) against gold-relevant passage ids, routed
through `ragkit.eval.retrieval.evaluate_retrieval` rather than a reimplemented metric loop — so
this recipe also gets that function's circularity and missing-gold guards. Recall is measured at
`k` (default 20); MRR and NDCG at a fixed depth of 10 (module constants `_MRR_K`/`_NDCG_K`); hit
rate at a fixed depth of 10 (`_HIT_RATE_K`, matching PolQA's "top-10 accuracy" convention, Rybak
et al. 2022) — three different, independently useful depths in one pass over the queries.
Ground truth is labelled `source="polqa"` (an independent dataset, not any system under
evaluation).

**Args:** `retriever` — the system to score; `queries` — `(record_id, question)` pairs; `gold` —
`{record_id: {relevant_passage_id, ...}}`; `k` — Recall@k depth, keyword-only, default 20.
**Returns:** `RetrievalScores` (`ragkit.eval.retrieval`) — one system's Recall@k, hit-rate@10,
MRR@10, NDCG@10, averaged over the queries that have gold.
**Raises:** `ragkit.eval.retrieval.CircularEvaluationError` or `IncompleteGroundTruthError` in
principle (propagated from `evaluate_retrieval`), though in practice neither triggers here: the
gold source (`"polqa"`) never matches the evaluated system's name (`"recipe-retriever"`), and
`evaluate` pre-filters `queries` to only those with a non-empty `gold` entry before building
`Qrels`, so the missing-gold guard has nothing left to catch.
**Side effects:** calls `retriever.retrieve` once per query that has gold, at the deepest depth
any metric needs.

#### load_gold(path: Path) -> dict[str, frozenset[str]]

`recipes/legal_procurement/eval.py`. Thin wrapper: `ragkit.eval.gold.load_relevance_gold(path,
field="relevant")`.

**Args:** `path` — gold JSONL file, each row needing `record_id` and a non-empty `relevant` list.
**Returns:** `{record_id: frozenset(relevant_passage_ids)}`.
**Raises:** `ragkit.eval.gold.EvalError` if the file is missing, empty, has invalid JSON, a
non-object row, a row missing a required key, or a `relevant` value that is not a non-empty list.
**Side effects:** reads `path` from disk.

#### load_heldout(path: Path) -> list[tuple[str, str]]

`recipes/legal_procurement/eval.py`. The held-out questions, as `(record_id, question)` in file
order (`question` comes from each gold row's `source` field).

**Args:** `path` — heldout JSONL, each row needing `record_id` and `source`.
**Returns:** `list[(record_id, question)]` in file order.
**Raises:** `EvalError` if `path` is not a file, or a row is invalid JSON / not an object /
missing a required key.
**Side effects:** reads `path` from disk.

#### grounding_rate(retriever: Retriever, journal: Path, gold: Mapping[str, frozenset[str]], queries: Mapping[str, str], \*, k: int, citations_field: str = "citations") -> tuple[int, int]

`recipes/legal_procurement/eval.py`. Of the answers a run actually produced, how many cite only
ids that are within the union of the question's gold passages and the passages freshly retrieved
for it at depth `k`. A record that is not injectable, whose output is not JSON, or whose parsed
output has no list under `citations_field`, is skipped entirely (excluded from both the numerator
and the `produced` denominator — a structurally absent citations field is not the same claim as
"produced but ungrounded").

**Args:** `retriever` — retriever to re-run per question; `journal` — run journal path; `gold` —
`{record_id: relevant_ids}`; `queries` — `{record_id: question}`; `k` — retrieval depth,
keyword-only; `citations_field` — JSON field holding the cited ids, default `"citations"`.
**Returns:** `(grounded, produced)` — counts, not a ratio (the caller divides).
**Raises:** nothing directly; propagates whatever `ragkit.core.records.read_journal` raises for
an unreadable journal file.
**Side effects:** reads `journal` from disk; calls `retriever.retrieve` once per produced,
citing record.

#### main(argv: Sequence[str] | None = None) -> int

`recipes/legal_procurement/eval.py` CLI entry point. Flags: `--config` (recipe config dir,
required), `--heldout` (required), `--gold` (required), `--k` (Recall@k depth, default 20),
`--journal` (optional; when given, also reports the citation-grounding rate). Assembles the
recipe's retriever via `ragkit.cli.app.assemble(args.config)`.

**Args:** `argv` — CLI arguments, or `None` for `sys.argv[1:]`.
**Returns:** `0` on success; `1` if `EvalError` is raised (`--k < 1`, no retriever wired by the
recipe's config, or any of the loaders above failing).
**Raises:** nothing — `EvalError` is caught by `run_report`.
**Side effects:** assembles the full recipe config (may load an embedder, a vector/lexical
index, etc.); reads the gold and heldout files; optionally reads the journal; prints one summary
line (or an error) to stdout/stderr.

### CitationGroundingValidator

`recipes/legal_procurement/plugins/validators.py`. Implements `ragkit.core.ports.Validator`.
Refuses a legal answer that is uncited while substantive, or that cites a passage id never
actually retrieved for the question; accepts an honest abstention (the configured
`abstain_marker`, default `"brak podstaw"`) with no citations. If no retriever is wired into the
run context, grounding cannot be verified at all — the validator blocks rather than passing
silently.

**Attributes:**
- `CONFIG_KEYS = frozenset({"answer_field", "citations_field", "abstain_marker", "k",
  "min_score"})`

#### from_config(cls, options: Mapping[str, Any]) -> CitationGroundingValidator

**Args:** `options` — recipe config; `answer_field` (default `"answer"`), `citations_field`
(default `"citations"`), `abstain_marker` (default `"brak podstaw"`), `k` (default 20),
`min_score` (default 0.0).
**Returns:** `CitationGroundingValidator`.
**Raises:** `ValueError` (via `__init__`) if `answer_field`, `citations_field`, or
`abstain_marker` is blank/whitespace-only, or `k < 1`.
**Side effects:** none.

#### validate(self, record: Record, output: str, context: Mapping[str, Any]) -> list[Violation]

Parses `output` as a JSON object; reads `answer_field` (must be a non-empty string). If the
answer text contains the (lower-cased) abstain marker, returns `[]` immediately — no citations
required or checked. Otherwise requires a non-empty `citations_field` list, re-retrieves the
question (`record.source`) through `context["retriever"]` at the configured `k`/`min_score`, and
flags any cited id not among the retrieved chunk ids.

**Args:** `record: Record`, `output: str` — the answer's JSON text, `context: Mapping[str, Any]`
— must carry `"retriever"` (a `ragkit.core.ports.Retriever`) for a substantive (non-abstaining)
answer to be checkable.
**Returns:** `list[Violation]`, all `Severity.ERROR`, rule id `"ungrounded_citation"`: JSON parse
failure; answer field missing/blank; no citations on a substantive answer; no retriever wired; or
one violation per citation that resolves to no retrieved passage. Empty means the answer passed
(including a valid abstention).
**Raises:** nothing — a JSON parse failure is returned as a violation.
**Side effects:** calls `context["retriever"].retrieve(record.source, k=self._k,
min_score=self._min_score)` and `ragkit.harness.capture.capture_retrieved(context, hits)` — records
those hits into the run's capture sink (if one is wired) so the retrieval this validator performed
is visible in the persisted `RunResult`, even though the validator's own retrieval would otherwise
be invisible to the harness.

## med_evidence

Decides `yes`/`no`/`maybe`/`unsupported` for a biomedical research question from retrieved
abstracts, citing verbatim evidence quotes copied from the retrieved text or abstaining
(`unsupported`) when the abstracts do not address the question.

### Report

`recipes/med_evidence/eval.py` — subclasses `ragkit.eval.classify.ClassificationReport` (a frozen
dataclass over `outcomes: tuple[Labelled, ...]`, `decision_field` scored via
`ragkit.eval.classify.score_labels`). Adds the two rates specific to a task the system may
decline.

#### answered(self) -> int

Property. Count of outcomes that produced a concrete (non-`unsupported`) decision — a miss
(`produced=False` or unparseable) and an abstention both excluded.

**Args:** none. **Returns:** `int`. **Raises:** nothing. **Side effects:** none.

#### abstain_rate(self) -> float

Property. `self.rate(_abstained)` — fraction of *all* outcomes (over `total`) where the system
produced output and predicted `"unsupported"`.

**Args:** none. **Returns:** `float` in `[0, 1]`. **Raises:** nothing. **Side effects:** none.

#### answered_accuracy(self) -> float

Property. `self.rate(lambda outcome: outcome.correct, over=self.answered)` — accuracy computed
only over the records the system actually committed a concrete decision on ("how right when it
commits"); `0.0` if `answered` is 0.

**Args:** none. **Returns:** `float` in `[0, 1]`. **Raises:** nothing. **Side effects:** none.

#### evaluate(pairs: object, \*, decision_field: str = "decision") -> Report

`recipes/med_evidence/eval.py`. `Report(score_labels(pairs, field=decision_field))` — delegates
entirely to the shared classification scorer; `pairs` is typed `object` in the signature but must
actually be `Iterable[Pair]` (`ragkit.eval.gold.Pair`).

**Args:** `pairs` — `(record_id, produced_output_or_None, gold_decision)` triples; `decision_field`
— JSON field to read the predicted decision from, keyword-only, default `"decision"`.
**Returns:** `Report`.
**Raises:** nothing.
**Side effects:** none.

#### main(argv: Sequence[str] | None = None) -> int

`recipes/med_evidence/eval.py` CLI entry point (`--journal`, `--gold`). Loads gold via
`load_label_gold(args.gold, field="decision")` and prints `"decided P/T | accuracy A | abstain
rate R | answered accuracy (over N) AA"`.

**Args:** `argv` — CLI arguments, or `None` for `sys.argv[1:]`.
**Returns:** `0` on success, `1` on `EvalError`.
**Raises:** nothing (`EvalError` caught by `run_report`).
**Side effects:** reads the gold file and journal from disk; prints one line to stdout/stderr.

### GroundedEvidenceValidator

`recipes/med_evidence/plugins/validators.py`. Implements `ragkit.core.ports.Validator`. Refuses a
biomedical decision whose `evidence` quotes are missing (for a non-abstaining decision) or not
found verbatim (after quote/whitespace normalisation, and split on an ellipsis to allow an elided
quote) in the abstracts actually retrieved for the question. If no retriever is wired, blocks
rather than passing silently — grounding is unverifiable without it.

**Attributes:**
- `CONFIG_KEYS = frozenset({"decision_field", "evidence_field", "unsupported_value", "k",
  "min_score", "min_quote_len"})`

#### from_config(cls, options: Mapping[str, Any]) -> GroundedEvidenceValidator

**Args:** `options` — `decision_field` (default `"decision"`), `evidence_field` (default
`"evidence"`), `unsupported_value` (default `"unsupported"`), `k` (default 20), `min_score`
(default 0.0), `min_quote_len` (default 12).
**Returns:** `GroundedEvidenceValidator`.
**Raises:** `ValueError` (via `__init__`) if `k < 1` or `min_quote_len < 1`.
**Side effects:** none.

#### validate(self, record: Record, output: str, context: Mapping[str, Any]) -> list[Violation]

Parses `output` as a JSON object. If the decision equals `unsupported_value` (case-insensitive)
and cites no evidence, passes (abstention with no evidence is exactly the allowed behaviour). If a
retriever is not wired, blocks unconditionally (grounding cannot be verified at all). Otherwise
requires at least one evidence quote, re-retrieves `record.source` through the wired retriever,
and checks each quote (after normalisation, split on ellipsis into segments) against the
concatenated retrieved text; a quote with no segment reaching `min_quote_len` characters, or any
segment absent from the retrieved text, is flagged.

**Args:** `record: Record`, `output: str` — the decision's JSON text, `context: Mapping[str, Any]`
— must carry `"retriever"` for a non-abstaining decision to pass.
**Returns:** `list[Violation]`, all `Severity.ERROR`, rule id `"ungrounded_evidence"`: JSON parse
failure; no retriever wired; a non-abstaining decision with no evidence; a quote too short to
verify; or a quote (or an elided segment of it) absent from the retrieved abstracts. Empty means
the decision passed.
**Raises:** nothing — parse failures are returned as violations.
**Side effects:** calls `context["retriever"].retrieve(record.source, k=self._k,
min_score=self._min_score)` and `capture_retrieved(context, hits)`.

### DecisionEnumValidator

`recipes/med_evidence/plugins/validators.py`. Implements `ragkit.core.ports.Validator`. Refuses
an answer whose `decision` is not one of the allowed labels (default `yes`/`no`/`maybe`/
`unsupported`) or whose JSON cannot be read at all.

**Attributes:**
- `CONFIG_KEYS = frozenset({"field", "allowed"})`

#### from_config(cls, options: Mapping[str, Any]) -> DecisionEnumValidator

**Args:** `options` — `field` (default `"decision"`), `allowed` (default
`("yes", "no", "maybe", "unsupported")`).
**Returns:** `DecisionEnumValidator`.
**Raises:** `ValueError` if `allowed` is present but not a non-empty list (also raised by
`__init__` if `allowed` ends up empty).
**Side effects:** none.

#### validate(self, record: Record, output: str, context: Mapping[str, Any]) -> list[Violation]

Parses `output` as a JSON object and checks `decision[field]` (case-insensitive, trimmed) against
the allowed set. `record` and `context` are unused.

**Args:** `record: Record` (unused), `output: str`, `context: Mapping[str, Any]` (unused).
**Returns:** `list[Violation]`, `Severity.ERROR`, rule id `"invalid_decision"`: JSON parse
failure, non-object JSON, or a value not in the allowed set. Empty means it passed.
**Raises:** nothing.
**Side effects:** none.

### SelfAbstractRetriever

`recipes/med_evidence/reader_eval.py`. Implements `ragkit.core.ports.Retriever`. Returns each
question's *own* gold PubMedQA abstract as the sole retrieved passage — the standard PubMedQA
reader setup (gold context handed to the model), used to isolate decision quality from retrieval
quality and make the reader eval's number directly comparable to the published benchmark.

**Attributes:** none public (holds a private `question -> abstract text` mapping built in
`__init__(self, abstract_of_question: Mapping[str, str])`).

#### retrieve(self, query: str, \*, k: int, min_score: float = 0.0) -> tuple[Retrieved, ...]

**Args:** `query: str` — looked up (stripped) against the question→abstract map; `k`, `min_score`
— accepted for `Retriever` port compatibility but ignored (a self-retriever has exactly one
passage or none, never a depth/floor to apply).
**Returns:** `(Retrieved("gold-0", text, 1.0),)` if `query.strip()` is a known question, else `()`
— a query it does not recognise returns nothing, so the grounding validator blocks it as it would
a genuine retrieval miss.
**Raises:** nothing.
**Side effects:** none.

#### load_reader_set(abstracts: Path, heldout: Path) -> tuple[list[Record], dict[str, str]]

`recipes/med_evidence/reader_eval.py`. Builds the held-out `Record`s and a
`question -> own-abstract-text` map (joined on PMID: each heldout row's `meta.pmid`, falling back
to its `record_id`, looked up in `abstracts`).

**Args:** `abstracts` — `abstracts.jsonl` path (`{"id": ..., "text": ...}` per line); `heldout` —
heldout questions JSONL path (`record_id`, `source`, optional `meta.pmid`).
**Returns:** `(records, by_question)` — `records` is one `Record` per heldout row (`record_id`,
`source`, `meta` carried through); `by_question` maps each `source` string to its PMID's abstract
text.
**Raises:** `ragkit.eval.gold.EvalError` if either file is missing, or a heldout row's PMID has no
matching abstract.
**Side effects:** reads both files from disk.

#### main(argv: Sequence[str] | None = None, \*, client_factory: ClientFactory | None = None) -> int

`recipes/med_evidence/reader_eval.py` CLI entry point. Flags: `--config`, `--abstracts`,
`--heldout`, `--gold` (all required), `--journal` (default `work/med_reader.jsonl`). Unlike
`eval.py`'s `main`, this one does **not** route through `run_report`: only the initial
`load_reader_set`/`load_label_gold` calls are wrapped in a `try/except EvalError`; a failure from
`assemble`, `run_batch`, or the store/journal writes below that point propagates as an ordinary
exception rather than being reported as `error: ...` on stderr.

Builds a `SqliteRunStore` at `<journal>.run.db`, adds the built records, assembles the recipe
with `retriever=SelfAbstractRetriever(by_question)` and the given `client_factory` (so a test can
inject a fake LLM client), runs the harness over every pending record
(`ragkit.harness.run_batch`, signal handlers disabled), exports the journal as a `.jsonl`
artifact, then scores it with `recipes.med_evidence.eval.evaluate` and prints `"[reader /
gold-context] decided P/T | accuracy A | answered accuracy (over N) AA"`.

**Args:** `argv` — CLI arguments, or `None` for `sys.argv[1:]`; `client_factory` — optional
override for how model clients are constructed (`ragkit.llm.pool.ClientFactory`), `None` uses the
recipe's configured default.
**Returns:** `0` on success (including a run where some records were rejected — only data-loading
failures return early); `1` if `load_reader_set` or `load_label_gold` raises `EvalError`.
**Raises:** whatever `ragkit.cli.app.assemble` or `ragkit.harness.run_batch` raise on a genuine
failure (e.g. an unreachable model server) — not caught here.
**Side effects:** creates `<journal>.run.db` and its parent directories; runs the recipe's full
harness against every held-out record (real model calls unless `client_factory` fakes them);
writes the journal JSONL; prints one line to stdout.

## nl_to_sql

Turns a natural-language question into a single, read-only `SELECT` against a real database,
scored by Spider-style execution-match (result sets compared, not SQL text) against gold SQL.

### Outcome

`recipes/nl_to_sql/eval.py` — a frozen, slotted dataclass, plain fields only (no computed
properties). Not a Protocol implementation.

**Attributes:**
- `record_id: str`
- `produced: bool` — a usable (injectable) output existed to evaluate.
- `executed: bool` — the produced query ran without error.
- `matched: bool` — its result set equals the gold query's.
- `detail: str = ""` — a human-readable reason when `matched` is false.

### Report

`recipes/nl_to_sql/eval.py` — aggregate execution accuracy. Plain frozen dataclass, no Protocol.

**Attributes:**
- `outcomes: tuple[Outcome, ...]`

#### total(self) -> int

Property. `len(self.outcomes)`. **Args:** none. **Returns:** `int`. **Raises:** nothing.
**Side effects:** none.

#### matched(self) -> int

Property. Count of outcomes with `matched=True`. **Args:** none. **Returns:** `int`.
**Raises:** nothing. **Side effects:** none.

#### executed(self) -> int

Property. Count of outcomes with `executed=True`. **Args:** none. **Returns:** `int`.
**Raises:** nothing. **Side effects:** none.

#### accuracy(self) -> float

Property. `matched / total`, over *every* gold question (a missing or failing query counts as a
miss, not an exclusion); `0.0` if `total` is 0.

**Args:** none. **Returns:** `float` in `[0, 1]`. **Raises:** nothing. **Side effects:** none.

#### evaluate(pairs: Iterable[Pair], store: SqlStore) -> Report

`recipes/nl_to_sql/eval.py`. For each `(record_id, produced_sql_or_None, gold_sql)` triple, runs
the **gold** query against `store` first (its result set is the baseline `expected`); if the gold
query itself fails to execute, that is treated as a configuration error (wrong database wired)
and raised rather than scored. Then, if `produced` is not `None`, runs it too and compares result
sets: **positionally by row tuple** (column order/count must agree) and, only when the gold SQL
contains an `ORDER BY` (checked via regex), row order matters too — otherwise both result sets are
compared as multisets (`collections.Counter`).

**Args:** `pairs` — `(record_id, produced_sql_or_None, gold_sql)` triples; `store` — the
(read-only) `ragkit.core.ports.SqlStore` to execute both queries against.
**Returns:** `Report`.
**Raises:** `ragkit.eval.gold.EvalError` if a **gold** query fails to execute against `store`
(never for a produced query — that is scored as `executed=False, matched=False` instead).
**Side effects:** executes every gold query, and every produced query for a record that has one,
against `store`.

#### main(argv: Sequence[str] | None = None) -> int

`recipes/nl_to_sql/eval.py` CLI entry point. Flags: `--journal`, `--gold`, and `--config` (recipe
config dir, to resolve the external database via `storage.toml`'s `[sql]` binding). Prints
`"execution accuracy: M/T = A (E executed)"`.

**Args:** `argv` — CLI arguments, or `None` for `sys.argv[1:]`.
**Returns:** `0` on success; `1` on `EvalError` — including `storage.toml` wiring no `[sql]`
store, or that store not being `read_only` (both refused before any query runs, since the
evaluator must never be able to mutate the external data source).
**Raises:** nothing (`EvalError` caught by `run_report`).
**Side effects:** loads `storage.toml` and opens the external SQL store; reads gold and journal
files; executes every gold/produced query against the real database; prints one line to
stdout/stderr.

### SqlSafetyValidator

`recipes/nl_to_sql/plugins/validators.py`. Implements `ragkit.core.ports.Validator`. Defence-in-
depth for a task that emits database commands: refuses anything that is not a single,
schema-bounded `SELECT`/`WITH ... SELECT`, on top of (not instead of) the store's own read-only
binding. Checks, in order: comments stripped first (so a keyword can't hide behind `--`); must be
exactly one statement (splitting on unquoted semicolons); must start with `SELECT`/`WITH`; must
contain no forbidden DDL/DML keyword anywhere (`DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`,
`CREATE`, `REPLACE`, `ATTACH`, `DETACH`, `PRAGMA`, `VACUUM`, `TRUNCATE`, `GRANT`, `REVOKE`, `EXEC`,
`EXECUTE`, `MERGE`, `CALL`); every referenced table (via `FROM`/`JOIN`, excluding CTE names) must
exist in the introspected schema if one is wired; and, if `explain` is enabled and a read-only
store is wired, the statement is dry-run through `EXPLAIN`.

**Attributes:**
- `CONFIG_KEYS = frozenset({"explain"})`

#### from_config(cls, options: Mapping[str, Any]) -> SqlSafetyValidator

**Args:** `options` — `explain` (default `True`).
**Returns:** `SqlSafetyValidator`.
**Raises:** nothing.
**Side effects:** none.

#### validate(self, record: Record, output: str, context: Mapping[str, Any]) -> list[Violation]

`record` is unused (present only for the `Validator` port's shape). Empty (comment-stripped)
output is left to the framework's built-in nonempty check and passes here (returns `[]`).

**Args:** `record: Record` (unused), `output: str` — the generated SQL text, `context:
Mapping[str, Any]` — optionally carries `"introspector"` (`ragkit.core.ports.SchemaIntrospector`)
for the table-existence check and `"sql_store"` (`ragkit.core.ports.SqlStore`) for the `EXPLAIN`
check; either check is skipped (not failed) if its service is not wired.
**Returns:** `list[Violation]`, `Severity.ERROR`, rule id `"sql_unsafe"`: multiple statements; does
not start with `SELECT`/`WITH`; a forbidden keyword present; a referenced table not in the
introspected schema; or an `EXPLAIN` failure. Empty means the query passed every check that
applied.
**Raises:** nothing — an `EXPLAIN` failure from the store is caught and returned as a violation.
**Side effects:** if `explain` is enabled and `context["sql_store"]` is a `SqlStore`, runs
`EXPLAIN <sql>` against it (a read against the real database, never a write).

## predictive_maintenance

Decides a diagnosis, a severity, and a recommended action from equipment manuals and an
operator's sensor readings/fault codes, refusing any decision whose evidence is missing or not
actually present in the retrieved manual passages.

### Report

`recipes/predictive_maintenance/eval.py` — subclasses `ragkit.eval.classify.ClassificationReport`
(scored on `severity_field` via `score_labels`). Gold severity is derived from remaining useful
life (`urgent` <= 30 cycles, `watch` <= 80, else `normal`).

#### exact_accuracy(self) -> float

Property. Alias for the inherited `accuracy` (predicted severity bucket equals gold exactly).

**Args:** none. **Returns:** `float` in `[0, 1]`. **Raises:** nothing. **Side effects:** none.

#### actionable_accuracy(self) -> float

Property. `self.rate(_actionable)` — fraction of outcomes where the needs-attention-vs-normal
call was right (`predicted in {watch, urgent}` matches `gold in {watch, urgent}`), forgiving a
watch/urgent mix-up but not a miss (a record with no decision never counts as a lucky "normal"
match). This is the coarser call that actually drives maintenance action.

**Args:** none. **Returns:** `float` in `[0, 1]`. **Raises:** nothing. **Side effects:** none.

#### evaluate(pairs: object, \*, severity_field: str = "severity") -> Report

`recipes/predictive_maintenance/eval.py`. `Report(score_labels(pairs, field=severity_field))`.

**Args:** `pairs` — `(record_id, produced_output_or_None, gold_severity)` triples; `severity_field`
— JSON field to read the predicted severity from, keyword-only, default `"severity"`.
**Returns:** `Report`. **Raises:** nothing. **Side effects:** none.

#### main(argv: Sequence[str] | None = None) -> int

`recipes/predictive_maintenance/eval.py` CLI entry point (`--journal`, `--gold`). Loads gold via
`load_label_gold(args.gold, field="severity")` and prints `"decided P/T | exact severity E |
actionable (attention vs normal) A"`.

**Args:** `argv` — CLI arguments, or `None` for `sys.argv[1:]`.
**Returns:** `0` on success, `1` on `EvalError`.
**Raises:** nothing (`EvalError` caught by `run_report`).
**Side effects:** reads the gold file and journal from disk; prints one line to stdout/stderr.

### GroundedDecisionValidator

`recipes/predictive_maintenance/plugins/validators.py`. Implements `ragkit.core.ports.Validator`.
Refuses a maintenance decision that is uncited, ungrounded, or wrongly categorised: requires a
non-empty `diagnosis`, a `severity` in the configured allowed set, and at least one `evidence`
quote that (after normalisation: lower-cased, whitespace-collapsed, quote characters stripped)
is a substring of the manual passages actually retrieved for the record. If no retriever is
wired, blocks unconditionally rather than passing silently.

**Attributes:**
- `CONFIG_KEYS = frozenset({"diagnosis_field", "severity_field", "evidence_field", "severities",
  "k", "min_score", "min_quote_len"})`

#### from_config(cls, options: Mapping[str, Any]) -> GroundedDecisionValidator

**Args:** `options` — `severities` (required, non-empty list — no default), `diagnosis_field`
(default `"diagnosis"`), `severity_field` (default `"severity"`), `evidence_field` (default
`"evidence"`), `k` (default 20), `min_score` (default 0.0), `min_quote_len` (default 12).
**Returns:** `GroundedDecisionValidator`.
**Raises:** `ValueError` if `severities` is missing, not a list, or empty (also raised by
`__init__` if `k < 1` or `min_quote_len < 1`).
**Side effects:** none.

#### validate(self, record: Record, output: str, context: Mapping[str, Any]) -> list[Violation]

Parses `output` as a JSON object, then runs three independent checks (all violations
accumulated, not short-circuited): diagnosis non-empty; severity is one of the configured
`severities` (case-insensitive); and evidence — at least one quote, each at least
`min_quote_len` characters after normalisation and a substring of the retrieved manual text
(re-retrieved via `context["retriever"]` at the configured `k`/`min_score`).

**Args:** `record: Record`, `output: str` — the decision's JSON text, `context: Mapping[str,
Any]` — must carry `"retriever"` for the evidence check to pass (its absence is itself a
violation).
**Returns:** `list[Violation]`, all `Severity.ERROR`, rule id `"ungrounded_decision"`: JSON parse
failure; missing/blank diagnosis; invalid severity; no evidence cited; no retriever wired; a
quote too short to verify; or a quote not found in the retrieved manuals. Empty means the
decision passed every check.
**Raises:** nothing — parse failures are returned as violations.
**Side effects:** calls `context["retriever"].retrieve(record.source, k=self._k,
min_score=self._min_score)` and `capture_retrieved(context, hits)`.

## translation

A faithful port of [`AzethMeron/llm-translator`](https://github.com/AzethMeron/llm-translator)
onto the framework: translates a source line and reports exact-match plus character-trigram
similarity against a held-out gold reference.

### Outcome

`recipes/translation/eval.py` — a frozen, slotted dataclass, plain fields only. No docstring in
source; not a Protocol implementation.

**Attributes:**
- `record_id: str`
- `produced: bool`
- `exact: bool` — the produced translation equals the gold reference after
  case/whitespace normalisation.
- `similarity: float` — character-trigram cosine similarity to the gold reference
  (`ragkit.retrieve.trigram_similarity`); `0.0` when `produced` is false.

### Report

`recipes/translation/eval.py` — aggregate over a run's `Outcome`s. Explicitly **not** a
`ClassificationReport`: translation scores *text* against a reference (no predicted label), only
an exact match and a graded similarity. Plain frozen dataclass, no Protocol.

**Attributes:**
- `outcomes: tuple[Outcome, ...]`

#### total(self) -> int

Property. `len(self.outcomes)`. **Args:** none. **Returns:** `int`. **Raises:** nothing.
**Side effects:** none.

#### produced(self) -> int

Property. Count of outcomes with `produced=True`. **Args:** none. **Returns:** `int`.
**Raises:** nothing. **Side effects:** none.

#### exact_match(self) -> float

Property. Fraction of outcomes with `exact=True`, over `total`; `0.0` if `total` is 0. A strict
lower bound on quality — many correct translations legitimately differ from the one gold
rendering.

**Args:** none. **Returns:** `float` in `[0, 1]`. **Raises:** nothing. **Side effects:** none.

#### mean_similarity(self) -> float

Property. Mean of `similarity` over all outcomes (including misses, which contribute `0.0`);
`0.0` if `total` is 0. Read alongside `exact_match`: high similarity with low exact match means
valid but differently-phrased translations; low similarity means the meaning drifted.

**Args:** none. **Returns:** `float` in `[0, 1]`. **Raises:** nothing. **Side effects:** none.

#### evaluate(pairs: Iterable[Pair]) -> Report

`recipes/translation/eval.py`. For each `(record_id, produced_or_None, gold_target)` triple: a
miss (`produced is None`) scores `exact=False, similarity=0.0`; otherwise both the produced text
and the gold reference are normalised (lower-cased, whitespace-collapsed) before comparing.

**Args:** `pairs` — `(record_id, produced: str | None, gold: object)` triples.
**Returns:** `Report`.
**Raises:** nothing.
**Side effects:** none.

#### main(argv: Sequence[str] | None = None) -> int

`recipes/translation/eval.py` CLI entry point (`--journal`, `--gold`). Loads gold via
`load_label_gold(args.gold, field="target")` and prints `"translated P/T | exact match E | mean
trigram similarity to gold S"`.

**Args:** `argv` — CLI arguments, or `None` for `sys.argv[1:]`.
**Returns:** `0` on success, `1` on `EvalError`.
**Raises:** nothing (`EvalError` caught by `run_report`).
**Side effects:** reads the gold file and journal from disk; prints one line to stdout/stderr.

#### available_scripts() -> list[str]

`recipes/translation/plugins/validators.py`. The names of the Unicode scripts
`ScriptValidator` recognises out of the box — sorted keys of the module's script→code-point-range
table (`han`, `hiragana`, `katakana`, `japanese`, `cyrillic`, `greek`, `arabic`, `hebrew`,
`hangul`, `devanagari`, `thai`).

**Args:** none.
**Returns:** `list[str]`, sorted.
**Raises:** nothing.
**Side effects:** none.

### EchoValidator

`recipes/translation/plugins/validators.py`. Implements `ragkit.core.ports.Validator`. Flags an
output that is a whole-line verbatim echo of the input (placeholders masked, Unicode-normalised,
casefolded) — i.e. the model returned the source untranslated. Abstains (no violation) when the
source is shorter than `min_words` words, since a short "translation" legitimately equalling the
source (a proper noun) cannot be told apart from a real echo on that little evidence. A partial
(non-whole-line) overlap is deliberately left to `ScriptValidator`, since some tokens (names,
numbers, "OK") legitimately survive translation.

**Attributes:**
- `CONFIG_KEYS = frozenset({"min_words"})`

#### from_config(cls, options: Mapping[str, Any]) -> EchoValidator

**Args:** `options` — `min_words` (default 2).
**Returns:** `EchoValidator`.
**Raises:** nothing.
**Side effects:** none.

#### validate(self, record: Record, output: str, context: Mapping[str, Any]) -> list[Violation]

`context` is unused. Normalises both `record.source` and `output` (placeholders like `[[3]]`
masked to a space, NFC-normalised, casefolded, whitespace-collapsed) and compares them for exact
equality.

**Args:** `record: Record` — `source` is the original-language line, `output: str` — the
produced translation, `context: Mapping[str, Any]` (unused).
**Returns:** `[]` if the source has fewer than `min_words` words (abstain — too little evidence);
otherwise `[]` unless the normalised output equals the normalised source, in which case one
`Severity.ERROR` violation, rule id `"untranslated"`.
**Raises:** nothing.
**Side effects:** none.

### ScriptValidator

`recipes/translation/plugins/validators.py`. Implements `ragkit.core.ports.Validator`. Flags
source-script characters surviving into a target of a different script (e.g. Han characters
surviving into an English translation) — needs no language profile, only the named source script
and the code-point ranges in the module's `_SCRIPTS` table. Only judges when the source actually
contains at least one character of the configured script (positive evidence); otherwise the check
does not apply and abstains.

**Attributes:**
- `CONFIG_KEYS = frozenset({"source_script", "max_survivors"})`

#### from_config(cls, options: Mapping[str, Any]) -> ScriptValidator

**Args:** `options` — `source_script` (required; must be a key of `available_scripts()`),
`max_survivors` (default 0 — a few tolerated stray characters, e.g. a quoted name kept on
purpose).
**Returns:** `ScriptValidator`.
**Raises:** `ValueError` (via `__init__`) if `source_script` is not a recognised script name
(message lists `available_scripts()`).
**Side effects:** none.

#### validate(self, record: Record, output: str, context: Mapping[str, Any]) -> list[Violation]

`context` is unused. Counts characters in `output` (placeholders masked out first) whose code
point falls in the configured script's ranges.

**Args:** `record: Record` — `source` is checked for at least one character of the configured
script before this validator applies at all, `output: str` — the produced translation,
`context: Mapping[str, Any]` (unused).
**Returns:** `[]` if `record.source` contains no character of the configured script (abstain —
does not apply), or if the count of surviving script characters in `output` is `<=
max_survivors`; otherwise one `Severity.ERROR` violation, rule id `"untranslated"`, naming the
survivor count and the script.
**Raises:** nothing.
**Side effects:** none.

## Package `__init__.py` files

`recipes/__init__.py` and, for each of the six recipes, its top-level `<recipe>/__init__.py` and
`<recipe>/plugins/__init__.py` are all empty (0 bytes). This is intentional: they exist only to
mark `recipes` and each recipe's `plugins/` as an importable Python package (so
`recipes.<name>.plugins.validators:ClassName` resolves as a dotted path for the registry — see
`ragkit.core.registry`), and carry no public API surface of their own. `recipes/<name>/tests/
__init__.py` files exist for the same reason and are out of scope for this reference (tests are
not public API).
