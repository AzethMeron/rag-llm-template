# `ragkit.eval`

`ragkit.eval` is the retrieval-metric, classification-scoring, and blinded-A/B-judge evaluation
layer. It scores what a recipe produced against ground truth: rank-quality metrics for a
retriever (`ragkit.eval.retrieval`), per-record label accuracy for a decision/classification task
(`ragkit.eval.classify`), the gold-loading and journal-joining scaffolding every recipe's `eval.py`
shares (`ragkit.eval.gold`), and an impartial-LLM A/B comparison between two systems' outputs
(`ragkit.eval.judge`). A recurring discipline runs through all four: a metric or comparison must
not be entangled with the thing it measures — `evaluate_retrieval` and `evaluate_ab` both refuse a
configuration that would score `1.000` (or a meaningless "win rate") by construction, rather than
silently reporting one.

The package `__init__` re-exports every public name from its four modules; nothing besides that
re-export lives at the top level.

## ragkit.eval.retrieval

Rank-quality metrics for a ranked list of chunk ids against a set of known-relevant ids, and
`evaluate_retrieval`, the harness that runs those metrics over one or more named retrievers and
refuses a circular configuration before doing any work. Every metric function is pure —
`(ranked_ids, relevant_ids[, k]) -> float` — so each is testable in isolation, independent of how
the ranking was produced.

### CircularEvaluationError

`RagkitError` subclass. Raised by `evaluate_retrieval` when the evaluation is configured so a
result would be true by construction.

The guard behind this is **nominal, not a provenance check** — this is the class's own documented
scope, not an implementation detail. It compares `Qrels.source` against the names of the systems
being ranked: a label check. Ground truth genuinely produced by a system under evaluation, but
labelled anything else, passes the guard. It catches the mistake of an evaluator wiring its own
retriever's output back in as gold; it cannot verify provenance it is not told about, and it is not
defense against an adversary who mislabels `source` on purpose.

### IncompleteGroundTruthError

`RagkitError` subclass. Raised by `evaluate_retrieval` when a query being evaluated has no
ground-truth relevant ids. Deliberately a distinct type from `CircularEvaluationError`: "the gold
set is incomplete" is a data problem with a different fix from "the gold set is not independent,"
and a caller that catches one should not silently absorb the other.

### Qrels

Ground-truth relevance judgments: for each query id, the set of relevant document ids, plus a
`source` label naming where the judgments came from (a gold dataset, an annotator). The label is
not decoration — `evaluate_retrieval` checks it is not one of the systems being ranked.

**Attributes:**
- `relevant: Mapping[str, frozenset[str]]` — query id → set of relevant chunk/document ids.
- `source: str` — who/what produced these judgments. Must be non-empty (see `__post_init__`
  below); checked for identity against the names in `evaluate_retrieval`'s `systems` mapping.

Frozen, slotted dataclass. `__post_init__` raises `ValueError` if `source` is empty or
whitespace-only — "an unlabelled ground truth cannot be checked for independence."

#### recall_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float

Fraction of the relevant ids that appear in the top `k` of `ranked`.

**Formula:** `|top_k(ranked) ∩ relevant| / |relevant|`, where `top_k(ranked) = set(ranked[:k])`.

**Args:** `ranked` — chunk ids best-first; `relevant` — the known-relevant id set; `k` — cutoff
depth.

**Returns:** the recall fraction, in `[0, 1]`.

**Raises:** none explicitly; `ZeroDivisionError` propagates if `relevant` is empty (the docstring
calls this case "undefined" — `evaluate_retrieval` excludes it before calling this, via
`IncompleteGroundTruthError`, but a direct caller gets no such guarantee).

#### hit_rate_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float

Binary per-query hit/miss at depth `k`, unlike `recall_at_k`'s fraction of *all* relevant ids
captured. This is the "top-k accuracy" metric reported by PolQA (Rybak et al., 2022) and similar
OpenQA retrieval papers, kept alongside recall/MRR/NDCG so a recipe can report a
literature-comparable number in addition to this framework's own convention.

**Formula:** `1.0` if `set(ranked[:k]) ∩ relevant` is non-empty, else `0.0`.

**Args/Returns:** same shape as `recall_at_k`.

**Raises:** none; unlike `recall_at_k`, safe when `relevant` is empty (returns `0.0`, no division).

#### reciprocal_rank(ranked: Sequence[str], relevant: frozenset[str]) -> float

The per-query term of MRR (Mean Reciprocal Rank).

**Formula:** `1 / rank` of the first relevant id in `ranked`, rank counted from 1; `0.0` if no
relevant id appears anywhere in `ranked`.

**Raises:** none; safe when `relevant` is empty (loop never matches, returns `0.0`).

#### average_precision(ranked: Sequence[str], relevant: frozenset[str]) -> float

The per-query term of MAP (Mean Average Precision).

**Formula:** for each rank `i` (1-indexed) where `ranked[i-1]` is relevant, accumulate
`hits_so_far / i` (precision at that rank); the result is that sum divided by `|relevant|` —
"mean of the precision values taken at each rank where a relevant id is hit, normalised by the
number of relevant ids."

**Raises:** none explicitly; `ZeroDivisionError` propagates if `relevant` is empty, same caveat as
`recall_at_k`.

#### ndcg_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float

NDCG@k with binary relevance (a hit contributes gain 1, a miss contributes 0 — no graded
relevance).

**Formula:** `DCG@k = Σ 1/log2(i+1)` over ranks `i` (1-indexed, within `ranked[:k]`) where
`ranked[i-1]` is relevant. `IDCG@k = Σ 1/log2(i+1)` for `i` in `1..min(k, |relevant|)` (the ideal
ordering: every relevant id ranked first). Returns `DCG@k / IDCG@k`, or `0.0` if `IDCG@k` is `0`
(i.e. `relevant` is empty or `k < 1`). `1.0` when the top `k` hold as many relevant ids as possible.

**Raises:** none; the only metric among the five that guards its own empty-`relevant` case
internally.

### RetrievalScores

Metrics for one system, averaged over the evaluated queries. Produced by `evaluate_retrieval`, one
per system name.

**Attributes:**
- `k: int` — the recall depth this run was scored at.
- `queries: int` — number of queries averaged over.
- `recall_at_k: float`, `hit_rate_at_k: float`, `mrr: float`, `map: float`, `ndcg_at_k: float` —
  mean of each metric across queries.
- `hit_rate_k: int` — the depth `hit_rate_at_k` was actually measured at (equal to `k` unless the
  caller passed a different `hit_rate_k` to `evaluate_retrieval`).
- `rank_k: int` — the depth `mrr`, `map`, and `ndcg_at_k` were actually measured at (equal to `k`
  unless the caller passed a different `rank_k`).

Each depth is recorded alongside its metric specifically so a report stays self-describing when
the three depths differ.

#### evaluate_retrieval(systems: Mapping[str, Retriever], queries: Mapping[str, str], qrels: Qrels, *, k: int = 10, hit_rate_k: int | None = None, rank_k: int | None = None) -> dict[str, RetrievalScores]

Scores each named retriever in `systems` over `queries` against `qrels`, returning one
`RetrievalScores` per system name.

**Args:**
- `systems` — system name → `Retriever` (from `ragkit.core.ports`) to evaluate.
- `queries` — query id → query text.
- `qrels` — the ground truth (see `Qrels`).
- `k` — recall's cutoff depth (default 10).
- `hit_rate_k` — `hit_rate_at_k`'s depth; defaults to `k` when `None`.
- `rank_k` — MRR/MAP/NDCG's depth; defaults to `k` when `None`. The three depths exist separately
  because conventional depths genuinely differ across metrics — a recipe can report Recall@20
  alongside a literature-comparable Acc@10 and MRR@10 in one call.

**Returns:** `{system_name: RetrievalScores}`, one entry per key of `systems`.

**Raises:**
- `ValueError` — if `k`, `hit_rate_k`, or `rank_k` is `< 1`; if `systems` is empty ("no systems to
  evaluate").
- `CircularEvaluationError` — if `qrels.source` is also a key of `systems` (checked by identity
  against the system name, before any retrieval runs).
- `IncompleteGroundTruthError` — if any query id in `queries` has no non-empty entry in
  `qrels.relevant` (checked before any retrieval runs, listing up to 3 offending query ids).

**Behavior notes:**
- All checks (arg validation, circularity, ground-truth completeness) run before any retriever is
  called — a misconfiguration is reported as itself, not masked by a downstream metric anomaly.
- Retrieval happens once per query, at `depth = max(k, hit_rate_k or k, rank_k or k)` (the deepest
  of the three), via `retriever.retrieve(text, k=depth)`; each metric then slices the ranked list
  (or is passed the appropriate cutoff) to its own required depth. This means increasing only one
  of the three depths does not multiply retrieval calls.
- Per-query scores are unweighted-averaged (arithmetic mean; `0.0` for an empty query set) into
  each system's `RetrievalScores`.

## ragkit.eval.judge

Blinded A/B output evaluation: an impartial LLM judge compares two systems' outputs per item. Two
independence disciplines apply, both instances of "a metric must not be entangled with what it
ranks":

- **Blinding** — the judge never sees which system produced which output; the two are shown as
  "Output 1"/"Output 2" in an order decided by an injected function (no hidden RNG), and the
  mapping back to A/B is recorded, not guessed.
- **Distinctness** — A/B-ing a system against itself is refused (`evaluate_ab` raises
  `CircularEvaluationError`, imported from `ragkit.eval.retrieval`, for the same "true by
  construction" reason).

The judge call goes through `ragkit.llm.client.LlmClient.complete_json` with a strict JSON schema
(`{"choice": "1"|"2"|"tie", "reason"?: str}`), so a reply that is not one of the three allowed
verdicts is a content error (`JudgeError`), not a silent miscount.

### JudgeError

`RagkitError` subclass. Raised when the judge returns a verdict outside the allowed set
(`"1"`/`"2"`/`"tie"`) — reachable in practice despite the JSON schema declaring an `enum`, because
`LlmClient`'s shape check verifies `required` and `type` but not `enum`; a backend running in
`json_object` mode (shape described only in the prompt, not grammar-constrained) can return any
string in `choice`.

### AbItem

One comparison to judge: the task input and the two systems' outputs for it.

**Attributes:**
- `item_id: str` — identifies this comparison in the resulting `AbVerdict`.
- `task_input: str` — the input shown to the judge for context.
- `output_a: str`, `output_b: str` — the two candidate outputs (blinded from the judge as "Output
  1"/"Output 2").

Frozen, slotted dataclass.

### AbVerdict

The de-blinded verdict for one item.

**Attributes:**
- `item_id: str` — matches the source `AbItem.item_id`.
- `winner: str` — `"a"`, `"b"`, or `"tie"`, resolved back from the judge's blind `"1"`/`"2"`/`"tie"`
  choice using `a_shown_first`.
- `reason: str` — the judge's stated reason (empty string if the judge omitted it).
- `a_shown_first: bool` — whether system A was shown as "Output 1" for this item.

Frozen, slotted dataclass.

### AbSummary

Aggregate A/B outcome for a batch of comparisons. Returned by `evaluate_ab`.

**Attributes:**
- `system_a: str`, `system_b: str` — the two system names being compared.
- `verdicts: tuple[AbVerdict, ...]` — one per compared item, in input order.

Frozen, slotted dataclass.

#### total(self) -> int

Count of verdicts (`len(self.verdicts)`).

#### a_wins(self) -> int

Count of verdicts with `winner == "a"`.

#### b_wins(self) -> int

Count of verdicts with `winner == "b"`.

#### ties(self) -> int

Count of verdicts with `winner == "tie"`.

#### a_win_rate(self) -> float

**Formula:** `(a_wins + 0.5 * ties) / total`, or `0.0` if `total == 0`. Ties count as half a win to
each side, so `a_win_rate` and the implied `b_win_rate` sum to 1.

#### evaluate_ab(client: LlmClient, items: Sequence[AbItem], *, system_a: str, system_b: str, criterion: str, model: str | None = None, order: Callable[[int], bool] = _alternate) -> AbSummary

Judges every item's two outputs and returns the aggregate.

**Args:**
- `client` — the `LlmClient` used for every judge call.
- `items` — the comparisons to judge.
- `system_a`, `system_b` — names of the two systems (recorded in the returned `AbSummary`, and
  checked for distinctness).
- `criterion` — what "better" means for this task, e.g. `"a more faithful, fluent translation"`;
  interpolated directly into the judge's system prompt.
- `model` — optional model override passed through to `client.complete_json`.
- `order` — `order(index) -> bool`; decides whether system A is shown first ("Output 1") for the
  item at `index`. Injected for determinism. Default `_alternate`: `index % 2 == 0` (A first on
  even indices, B first on odd) — balances position bias across the batch without a random source.

**Returns:** `AbSummary` with one `AbVerdict` per item, in input order.

**Raises:**
- `CircularEvaluationError` — if `system_a == system_b`, raised immediately, before any judge call
  is made.
- `JudgeError` — if any judge reply's `choice` is not `"1"`, `"2"`, or `"tie"` (see `JudgeError`
  above); raised mid-batch, so verdicts for items already judged are not returned (the function
  does not catch this to return a partial `AbSummary`).

**Side effects:** issues one `client.complete_json` call per item, each with
`SamplingParams(temperature=0.0)` and `role="judge"`.

## ragkit.eval.classify

Scoring a recipe that predicts a **label** per record: a decision, a severity bucket, a class.
Several shipped recipes do exactly this and previously had near-identical `Outcome`/`Report` pairs
to prove it, differing only in the field name and which extra rates each reported. What's genuinely
shared is the shape (did it produce anything, what did it predict, what was the gold, what fraction
got it right); what's genuinely per-recipe is *which* rates matter. `ClassificationReport` supplies
the shared shape and a generic `rate()`; a recipe subclasses it to add its own named properties
rather than re-implementing the counting.

### Labelled

One record's verdict: what the system predicted (`None` when it produced nothing usable) against
the gold label.

**Attributes:**
- `record_id: str`
- `produced: bool` — whether the run produced any usable output for this record.
- `predicted: str | None` — the extracted prediction, or `None` if nothing usable was produced.
- `gold: str` — the gold label (already lower-cased/stripped by `score_labels`).

Frozen, slotted dataclass.

#### correct(self) -> bool

`self.predicted == self.gold`. A record that produced nothing has `predicted=None`, which can never
equal a (non-`None`) gold string, so a miss can never score as correct.

### ClassificationReport

Counts and rates over a run's `Labelled` outcomes. Meant to be subclassed: add a recipe's own
metrics as properties built on `rate()` (e.g. an abstention rate, an "actionable" rate that
forgives a within-group confusion) rather than recounting from scratch.

**Attributes:**
- `outcomes: tuple[Labelled, ...]` — every evaluated record's outcome.

Frozen, slotted dataclass.

#### total(self) -> int

`len(self.outcomes)`.

#### produced(self) -> int

Count of outcomes with `produced == True` — how many records the run produced any usable output
for.

#### accuracy(self) -> float

`self.rate(lambda outcome: outcome.correct)` — correct over *every* evaluated record, so a record
the run failed to produce counts against the score rather than vanishing from the denominator.

#### rate(self, predicate: Callable[[Labelled], bool], *, over: int | None = None) -> float

The fraction of outcomes satisfying `predicate`.

**Args:** `predicate` — tested against each `Labelled`; `over` — the denominator, defaulting to
`self.total`; pass this explicitly for a rate scored only over the records a recipe committed to
(a subset of `total`).

**Returns:** `sum(1 for o in outcomes if predicate(o)) / denominator`, or `0.0` if `denominator <=
0` — an empty run reports `0.0` rather than raising a division error.

#### score_labels(pairs: Iterable[Pair], *, field: str) -> tuple[Labelled, ...]

Turns `(record_id, produced, gold)` triples (the `Pair` shape from `ragkit.eval.gold`) into
`Labelled` outcomes.

**Args:** `pairs` — iterable of `(record_id, produced_json_str_or_None, gold_value)`; `field` — the
key read from each produced JSON object via `json_field`.

**Returns:** a tuple of `Labelled`, one per input pair, with `predicted = json_field(produced,
field)` and `gold = str(gold).strip().lower()` — gold is normalized so casing differences in a gold
file cannot silently cost accuracy.

**Raises:** none; delegates all defensiveness to `json_field` (any malformed `produced` becomes
`predicted=None`, never an exception).

## ragkit.eval.gold

Gold data and the scaffolding every recipe's `eval.py` needs: read the judgments, line them up with
a run journal, and report a failure the same way. Each recipe scores something different (a
decision label, a severity bucket, execution-equivalent SQL, retrieval rank quality), but the frame
around that is identical — load a JSONL gold file (validating each row and naming `path:line` when
it's wrong), pair each gold id with whatever the journal produced for it (`None` when the record
was rejected or never ran), and turn an `EvalError` into one stderr line and exit 1. This module is
the single home for that frame, extracted after several near-identical copies had already drifted
(the required-key check differed per recipe, and a fix to one copy's error message hadn't reached
the others).

**`Pair` type alias:** `tuple[str, str | None, object]` — `(record_id, produced output or None,
gold value)`, the shape every scorer in this package consumes.

### EvalError

`RagkitError` subclass. Raised when a recipe's evaluation cannot run as configured: missing or
malformed gold, or a journal that cannot be read. Being a `RagkitError`, a host embedding a recipe
eval catches it alongside every other framework error rather than a bare `Exception`.

#### gold_rows(path: Path, *, required: Sequence[str]) -> Iterator[Mapping[str, object]]

Yields each non-blank line of a JSONL gold file, parsed and checked for `required` keys. The single
home for the per-row checks every recipe used to write itself, so the "a gold row needs ..."
message and the `path:line` prefix are identical everywhere.

**Args:** `path` — the gold file; `required` — keys every row must contain.

**Returns:** an iterator (generator) of parsed JSON objects (as `Mapping`), one per non-blank line,
in file order.

**Raises (lazily, only as the generator is iterated — not when `gold_rows(...)` is called):**
- `EvalError` — `path` is not a file; a line is not valid JSON (`json.JSONDecodeError` chained via
  `from exc`); a parsed row is not a JSON object (e.g. a JSON array or scalar); a row is missing one
  or more `required` keys; or the file has no non-blank lines at all ("gold file is empty").

Note the laziness: a caller that does `gold_rows(path, required=(...))` without iterating gets no
exception at all — only `for row in gold_rows(...)` (or wrapping it in `list(...)`, as the
`load_*_gold` functions below do) actually triggers the checks.

#### load_label_gold(path: Path, *, field: str) -> dict[str, str]

`{record_id: label}` — the single-label gold shape (a decision, a severity bucket, a reference
translation, a gold SQL statement).

**Args:** `path` — gold file; `field` — the label column name.

**Returns:** dict built by iterating `gold_rows(path, required=("record_id", field))`; `record_id`
and the label are both coerced with `str(...)`.

**Raises:** `EvalError`, propagated from `gold_rows` (iteration happens immediately here, inside
the dict comprehension, so this call itself can raise).

#### load_fields_gold(path: Path, *, fields: Sequence[str]) -> dict[str, dict[str, object]]

`{record_id: {field: value, ...}}` — the several-fields-per-record shape (a form's held-out
columns). Values keep their JSON type (a numeric field is compared numerically, not as a string).

**Args:** `path` — gold file; `fields` — the column names to extract.

**Returns:** dict via `gold_rows(path, required=("record_id", *fields))`.

**Raises:** `EvalError`, same as `load_label_gold`.

#### load_relevance_gold(path: Path, field: str = "relevant") -> dict[str, frozenset[str]]

`{record_id: {relevant_chunk_id, ...}}` — the retrieval gold shape.

**Args:** `path` — gold file; `field` — the column holding the list of relevant ids (default
`"relevant"`).

**Returns:** dict via `gold_rows(path, required=("record_id", field))`, converting each row's
`field` value to a `frozenset[str]`.

**Raises:** `EvalError` — from `gold_rows`, or (this function's own check) if a row's `field` value
is not a list, or is an empty list — "an empty list is refused here rather than becoming a query
that scores 0 by construction."

#### join_journal_with_gold(journal: Path, gold: Mapping[str, object]) -> list[Pair]

Produces one `(record_id, produced, gold_value)` `Pair` per entry of `gold`, in `gold`'s iteration
order.

**Args:** `journal` — a run journal (JSONL), read via `ragkit.core.records.read_journal`; `gold` —
`{record_id: gold_value}` (from one of the `load_*_gold` functions above).

**Returns:** `produced` is the record's `output` when the journal has an entry for that
`record_id` *and* its `status.is_injectable` is true; `None` otherwise (rejected, skipped, or never
ran). This distinction is why the join is shared code: scoring a rejection as an empty-string
answer rather than a miss would quietly flatter every recipe's numbers. A `record_id` present in
`gold` but absent from the journal also yields `produced=None`.

**Raises:** propagates whatever `ragkit.core.records.read_journal` raises for a malformed journal
(a `CatalogError`); this function itself has no explicit `raise`.

#### json_field(produced: str | None, field: str) -> str | None

Extracts `field` from a produced JSON object, lower-cased and stripped.

**Args:** `produced` — a record's raw output string, or `None`; `field` — the key to read.

**Returns:** `None` if `produced` is `None`; `None` if `produced` is not valid JSON
(`json.JSONDecodeError` is caught, not propagated); `None` if the parsed value is not a JSON
object, or `field` is absent, or `parsed[field]` is not a string; otherwise the string value,
`.strip().lower()`'d.

**Raises:** none — every failure mode is a miss (`None`), never an exception, because "an eval must
survive whatever a model returned."

#### journal_gold_parser(description: str, *, gold_help: str) -> argparse.ArgumentParser

Builds the `--journal`/`--gold` argument parser every recipe eval script starts from.

**Args:** `description` — the parser's `description`; `gold_help` — help text for `--gold` (since
what the gold file contains differs per recipe).

**Returns:** an `argparse.ArgumentParser` with two required arguments: `--journal` (`Path`, help
"run journal (JSONL)") and `--gold` (`Path`, help = `gold_help`).

#### run_report(build: Callable[[], str]) -> int

The shared tail of every recipe eval's `main()`.

**Args:** `build` — a zero-argument callable that computes and returns the report line to print.

**Returns:** `0` on success (after printing `build()`'s return value to stdout); `1` if `build()`
raises `EvalError` (after printing `f"error: {exc}"` to stderr).

**Raises:** none itself — it is the boundary that turns `EvalError` into an exit code instead of
letting it propagate; any other exception from `build` is *not* caught and propagates normally.

**Side effects:** writes to stdout (success) or stderr (failure).
