# `ragkit.harness`

The orchestration core. This package owns the **personas** that act under fixed system prompts
(a producer and a panel of reviewers, `roles.py`), the **produce -> mechanical check -> review
panel -> revise loop** that sequences them over one record at a time (`agents.py`), the
declarative **context assembly** that builds each prompt's grounding passage under a character
budget (`context/`), the **mechanical validators** and rule set that decide what is decidable in
code before any GPU time is spent on an opinion (`rules.py`, `validators.py`), the **run-local
output memory** that lets duplicate and neighbouring records reuse an already-accepted output
(`memory.py`), the **resumable batch runner** that drives many records to completion durably
(`runner.py`), and the **output-schema** implementations that describe what a producer returns
(`schemas.py`). The package is task-agnostic: what the output *is* (a translation, a SQL
statement, a filled form) is an `ragkit.core.ports.OutputSchema`; the reviewers, budgets, rules
and context blocks are all configuration. It depends on the `llm` and `core` layers and never on
a concrete store or retriever — those are injected as `ragkit.core.ports` implementations.

`ragkit/harness/__init__.py` re-exports the package's public surface (`Harness`, `Outcome`,
`Review`, `Attempt`, `learn_memory`, `Panel`, `Persona`, `Limits`, `Leniency`, `load_panel`,
`RuleSet`, `ValidatorPipeline`, `VALIDATORS`, `check_mechanical`, `blocking` (re-exported from
`ragkit.core.rules`), `ContextAssembler`, `load_context`, `OUTPUT_SCHEMAS`, `JsonFieldSchema`,
`FormSchema`, `FormField`, `OutputMemory`, `run_batch`, `Progress`, `RunnerError`,
`group_duplicates`) in its `__all__`. `ragkit/harness/context/__init__.py` does the same for the
context subpackage (`ContextAssembler`, `load_context`, `CONTEXT_BLOCKS`, `ContextBlockError`,
and the nine built-in block classes).

---

## ragkit.harness.agents

The bounded produce -> mechanical-check -> review-panel -> revise loop, lifted from a translation
engine where it turned out to have nothing to do with translation. The mechanical check sits
between the model steps deliberately: placeholder integrity, emptiness, forbidden patterns and
width are decidable in code, so they are settled before any GPU time is spent on an opinion, and a
failing output is sent straight back with a precise reason. Only what needs judgement reaches the
review panel. Errors are handled by blast radius (`ragkit.llm.errors`): an `LlmContentError` on
the produce path records the record `REJECTED` and lets the run continue; a bare `LlmError`
propagates and stops the run with the journal intact. Everything is bounded — `max_repairs`
mechanical-repair rounds per generation, `max_revisions` review rounds per record — after which
the output is recorded rather than retried forever. Prompt assembly obeys one rule throughout: an
optional section is included only when it has content (see `context/blocks.py`'s no-empty-section
contract).

**Module-level constant:**

- `PLACEHOLDER_CONTRACT: str` — the fixed instruction block appended to the producer's and each
  reviewer's system prompt telling the model how to treat `[[0]]`, `[[1]]`, ... placeholders:
  reproduce every one exactly, use each exactly once, never renumber or interpret one, but it may
  be moved to fit the output's word order. Appended only when `record.source` actually contains at
  least one placeholder (`placeholder_indices(record.source)` is non-empty), in both
  `Harness.produce` and `Harness.review`.

Also present but not in the AST checklist (module-level, no leading underscore, so technically
public, though undocumented by the extractor): `REVIEW_SCHEMA: dict[str, Any]` — the JSON schema a
reviewer's reply must satisfy: `acceptable: bool` (required), `issues: array[string] | null`
(nullable because a `json_object`-mode reviewer that accepts encodes "no issues" as `null`, which
the caller coerces to an empty tuple), `improved_output: string | null`.

### Review

One reviewer persona's verdict on a produced output. Frozen dataclass (`slots=True`).

**Attributes:**

- `role: str` — the reviewing persona's `id`.
- `acceptable: bool` — whether the reviewer accepts the output. Forced `True` when `error` is set:
  an abstention is not the same as acceptance, but it cannot veto either.
- `issues: tuple[str, ...] = ()` — concrete problems the reviewer named. `Harness.review`
  guarantees this is non-empty whenever `acceptable` is `False` (see below).
- `improved: str | None = None` — the reviewer's proposed rewrite, kept only when it differs from
  the output it reviewed.
- `error: str | None = None` — set when this reviewer's reply could not be evaluated: an
  abstention that cannot veto, but means the panel did not fully run, so the record is kept for
  review rather than verified.

No public methods beyond the dataclass-generated ones.

### Attempt

A rejected output and everything known about why, carried into the next try so the model revises
rather than re-produces blind. Frozen dataclass (`slots=True`).

**Attributes:**

- `target: str` — the output text that was rejected. May be `""` when a first-round content or
  truncation error left nothing to show (see `PreviousAttemptBlock.render`, which relies on this).
- `issues: tuple[str, ...]` — why it was rejected: mechanical violation messages, or reviewer
  objections prefixed `"{role}: {issue}"`.
- `suggestions: tuple[str, ...] = ()` — reviewer-proposed rewrites carried forward as hints, not
  adopted verbatim (the model "owns the final output").

### Outcome

Result of running one record through the full harness. Frozen dataclass (`slots=True`).

**Attributes:**

- `record: Record` — the record this outcome is for.
- `status: Status` — `VERIFIED`, `PRODUCED`, `REJECTED`, or `SKIPPED`.
- `output: str | None` — the produced text, or `None` for `SKIPPED`/some `REJECTED` outcomes.
- `violations: tuple[Violation, ...] = ()` — mechanical-check results against the accepted/final
  candidate.
- `reviews: tuple[Review, ...] = ()` — every `Review` collected across every revision round.
- `rounds: int = 0` — how many revision rounds ran (1-based).
- `error: str | None = None` — a human-readable reason, set whenever the status is not a clean
  `VERIFIED` (a flag, an exhaustion, a caught exception's diagnostic).
- `context_passage: str = ""` — diagnostic/reproducibility capture of the assembled prompt passage
  that actually produced `output` (see `.capture.Capture`), not part of the pass/fail verdict.
  Empty when the record was skipped, no context block was configured, or nothing was retrieved.
- `retrieved: tuple[Retrieved, ...] = ()` — chunks a retriever returned along the way, same
  diagnostic status as `context_passage`.

#### `applied(self) -> Record`

Returns: `self.applied_to(self.record)` — this outcome written onto its own record.

Raises: none.

#### `applied_to(self, record: Record) -> Record`

This outcome written onto `record` (used to share one output across duplicates: only the verdict
travels, so each occurrence keeps its own id and provenance).

**Args:** `record: Record` — typically a different (duplicate) record than `self.record`, sharing
the same `source`.

**Returns:** `Record` — a copy of `record` with `status`/`output` replaced and `notes` built from
every violation (`"{rule_id}: {message}"`), every reviewer issue (`"{role}: {issue}"`), every
abstaining reviewer (`"{role}: could not be evaluated ({error})"`), and, if `self.error` is set, a
trailing `"error: {error}"` note.

Raises: none directly; propagates `Record.__post_init__`'s `CatalogError` if the resulting
combination of `status`/`output` would violate `Record`'s invariant (e.g. an injectable status
with `output=None` — cannot occur here since `replace` always supplies `self.output` alongside
`self.status`).

### Harness

The personas and the loop that sequences them, over one record at a time.

Thread-safe for `ragkit.harness.runner.run_batch`'s concurrent workers: `process` is free of
shared mutable state except the per-reviewer leniency window (guarded here by
`self._history_lock`), the model pool's usage stats and lazily-built per-model clients (guarded
inside `ModelPool`), and the injected `memory` (guarded inside `OutputMemory`). The context
retriever and SQL store are read-only and need no lock.

**Constructor:** `__init__(self, pool: ModelPool, panel: Panel, ruleset: RuleSet, output_schema:
OutputSchema, validators: ValidatorPipeline, context: ContextAssembler, *, memory: OutputMemory |
None = None, retriever: Retriever | None = None, sql_store: SqlStore | None = None, introspector:
SchemaIntrospector | None = None, sanitize: Callable[[str], str] = lambda text: text, input_label:
str = "Input to act on:", stand_in: str = "they") -> None`

Stores every argument as a same-named public attribute (`self.pool`, `self.panel`, ...). Raises
`ValueError` if any reviewer in `panel.reviewers` has `from_rules=True` but `ruleset.
advisory_rules` is empty ("a `from_rules` reviewer is configured but the rule set has no
`[[advisory]]` criteria, so it would judge against nothing"). Initializes the private
`self._reply_history: dict[str, deque[bool]]` and `self._history_lock: threading.Lock` used by
`_register_reply`.

#### `produce(self, record: Record, previous: Attempt | None = None, *, capture: Capture | None = None) -> str`

Generate one candidate output from the panel's producer persona.

**Args:**
- `record: Record` — `record.source` is the input; also drives the placeholder contract (only
  added when the source contains at least one `[[N]]` placeholder) and the produce token budget
  (`Limits.produce_budget(len(record.source))`).
- `previous: Attempt | None = None` — feedback from a prior rejected attempt, folded into the user
  prompt via `PreviousAttemptBlock` (through `_user_prompt` -> `context.assemble`).
- `capture: Capture | None = None` — keyword-only; if given, the assembled prompt passage is
  recorded onto it via `capture.note_passage`.

**Returns:** `str` — `self.sanitize(self.output_schema.extract(reply))`: the schema-defined output
field, extracted from the producer's parsed JSON reply and passed through the injected
`sanitize` callable.

**Raises:** propagates whatever `LlmClient.complete_json` raises for the producer's request:
`LlmRefusalError` / `LlmTruncationError` / `LlmIncompleteJsonError` (all `LlmContentError`
subclasses) for unusable output, or a bare `LlmError` if the transport/server itself is broken.
Also propagates a `KeyError` (or any other exception) from a custom `OutputSchema.extract` or the
injected `sanitize`.

**Side effects:** none beyond the optional `capture.note_passage` write (see `capture` arg above).

#### `review(self, record: Record, output: str, reviewer: Persona) -> Review`

Get one reviewer persona's verdict on `output`.

**Args:**
- `record: Record` — the record `output` was produced for.
- `output: str` — the candidate to judge.
- `reviewer: Persona` — which reviewer runs. If `reviewer.from_rules`, the system prompt is built
  from `self.ruleset.advisory_rules` (`_rule_instructions`) instead of `reviewer.instructions`.

**Returns:** `Review` — `role=reviewer.id`; `acceptable` from the reply's `"acceptable"` field;
`issues` from `"issues"` (`None`/absent coerced to `()`); if the reviewer marked the output
unacceptable but supplied no issues, `issues` is forced to a single generic placeholder message
("marked unacceptable without a specific reason..."); `improved` is the reply's
`"improved_output"`, kept only when it is truthy and differs from `output`.

**Raises:** same as `produce`: propagates `LlmContentError` subclasses and bare `LlmError` from
`client.complete_json`.

**Side effects:** none.

#### `review_panel(self, record: Record, output: str) -> list[Review]`

Run reviewers in configured order, stopping at the first objection. An unreadable reply is
isolated as an abstention (it cannot veto) so it does not abort the panel; the record is then kept
for review rather than verified on a panel that did not fully run.

**Args:** `record: Record`, `output: str` — the candidate every configured reviewer judges.

**Returns:** `list[Review]` — one entry per reviewer actually consulted. Stops after the first
`Review` with `acceptable=False`; an abstaining reviewer (`error` set, `acceptable` forced `True`)
does *not* stop the panel, since it cannot veto.

**Raises:** does not raise on an individual reviewer's `LlmContentError` (caught and converted to
an abstaining `Review`, see below); a bare `LlmError` from any reviewer call propagates uncaught
(infrastructure failure, same blast-radius rule as `produce`/`review`).

**Side effects:**
- On an `LlmContentError` from `self.review(...)`, logs a warning (`logger.warning`, with a clipped
  input/reply-body diagnostic and `extra={"leniency_suppress_console": not surfaces}`) and appends
  an abstaining `Review(role=reviewer.id, acceptable=True, error=...)`.
- Calls `self._register_reply(reviewer, bad=...)` for every reviewer consulted (both the abstained
  and the normal path), which mutates `self._reply_history[reviewer.id]` — a `deque[bool]` bounded
  to `reviewer.leniency.window`, rebuilt if the configured window size changed — under
  `self._history_lock`. This is the one piece of state `process()` mutates that is shared across
  concurrent calls to `Harness.process` for different records but the same reviewer persona.

#### `process(self, record: Record) -> Outcome`

Run one record through production, mechanical checks, review and revision — the harness's single
entry point per record.

**Args:** `record: Record`.

**Returns:** `Outcome` — the verdict for *this* record. Short-circuits to
`Outcome(status=Status.SKIPPED, output=None)` immediately when `record.status is Status.SKIPPED`
or `record.source.strip()` is empty (no `Capture` is even created in that case). Otherwise creates
one fresh `Capture()` for the whole call (never shared across records — see `.capture.Capture`),
and loops up to `panel.max_revisions + 1` times:

1. Calls the private `self._generate(record, feedback, capture=capture)`, which itself produces
   and mechanically repairs up to `panel.max_repairs + 1` sub-rounds via `self.produce` +
   `self.validators.check`, returning `(target, violations, from_repair)` — so one revision round
   can itself cost up to `max_repairs + 1` calls to `produce`, and a full `process()` call up to
   `(max_revisions + 1) * (max_repairs + 1)` in the worst case.
2. Splits the mechanical `violations` via `partition_on_exhaustion(violations, self.ruleset)`:
   - Any `reject` violation (a blocking rule not in `keep_flagged_rules`, only possible on the
     final repair round) -> returns `Outcome(REJECTED, output=target, error="mechanical rules
     still violated after repairs")`.
   - Else any `keep` violation (a blocking rule in `keep_flagged_rules`, exhausted) -> returns
     `Outcome(PRODUCED, output=target, error="kept despite a flagged rule (the alternative is
     showing nothing)")`.
   - Else runs `self.review_panel(record, target)`, accumulating every round's reviews into
     `all_reviews`.
     - No objections -> returns `self._settle(...)` (see below): `VERIFIED`, or `PRODUCED` with a
       flag if the panel didn't fully run or the candidate came from a repaired-but-unverified
       truncated envelope (`from_repair`).
     - Objections and this was the last revision round -> returns `Outcome(PRODUCED,
       error="review objections unresolved within budget")`.
     - Objections with rounds remaining -> builds the next `Attempt` from the objecting reviewers'
       issues/suggestions and loops again.

   Every returned `Outcome` is wrapped through the local `captured(...)` closure, which fills in
   `context_passage`/`retrieved` from the round's `Capture` before returning.

**Raises:**
- A bare `LlmError` (not one of the more specific `LlmContentError` subclasses caught below)
  propagates uncaught: infrastructure/transport is broken, every remaining record would fail
  alike, so `run_batch` must stop the run with its journal intact rather than journal a false
  `REJECTED`.
- `AssertionError("unreachable: revision loop always returns")` after the `for` loop — purely
  defensive; the loop's own branches return in every iteration (the last-round branch always
  returns before falling through to build another `Attempt`), so this is not expected to fire.
- Never raises for a record-local defect (see the exception-handling note below) — it is caught
  and turned into a `REJECTED` `Outcome` instead.

**Side effects:**
- Creates and threads one `Capture()` through every `produce`/mechanical-check call in the round
  that produced the returned outcome.
- Mutates `self._reply_history` / `self._history_lock` via `review_panel` -> `_register_reply`,
  for each reviewer actually consulted.
- Uses `self.pool.client_for(...)`, which lazily builds and caches a per-model client and updates
  the pool's own usage stats (guarded inside `ModelPool`, not by `Harness`).
- Logs a warning per abstaining reviewer (inside `review_panel`) and `logger.exception` (full
  traceback) on an unexpected exception (see below).
- Does **not** itself write into `self.memory` — that is `learn_memory`'s job, called by the
  caller (`run_batch`) after `process()` returns.

**Exception-handling contract (the produce-review-revise loop's most load-bearing behaviour):**
the `try`/`except` around the whole loop distinguishes three tiers, in this order:

1. `except LlmRefusalError as exc:` -> `Outcome(REJECTED, output=None, error=f"model refused:
   {exc}")`.
2. `except LlmContentError as exc:` (any other content-error subclass that escaped `_generate`'s
   own repair budget, e.g. an `LlmIncompleteJsonError` with `repair_truncated_json=False` on the
   last round) -> `Outcome(REJECTED, output=None, error=str(exc))`.
3. `except LlmError:` (a bare, broader-blast-radius error — not caught by 1 or 2, since those are
   subclasses) -> **re-raised** ("infrastructure: every remaining record would fail alike --
   stop the run").
4. `except Exception as exc:` — anything else that escapes: a `re.error` from a pluggable
   `Validator`, a `KeyError` in a custom `OutputSchema.extract`, a bug in an injected `sanitize`.
   This is a **defect in code this layer called, not a statement about the server**, and is caught
   and returned as a `REJECTED` `Outcome` carrying `_unexpected(exc)` as its diagnostic (the
   exception's type, message, and the innermost raising frame's function/file/line — see the
   private `_unexpected` helper). `logger.exception(...)` logs the full traceback first.

   This is the fix the module docstring calls out explicitly: letting such an exception propagate
   instead made **one bad record permanently poison a whole batch** — `run_batch` skips
   `append_result` for a worker whose `Future.result()` raised (see `runner.run_batch`'s `except
   BaseException as exc: failure = failure or exc; continue`), so the record stayed `PENDING`,
   the run aborted, and every resume hit the same deterministic exception and aborted again. Now
   that same defect journals as a normal `REJECTED` result for *that one record* and the run
   continues — the trade is deliberate (a plugin bug that hits every record now journals every
   record `REJECTED` rather than stopping at the first), but it is the loud kind: every occurrence
   carries the failing frame and is logged with a full traceback, and containment is scoped to one
   record, never to a whole batch.

#### `learn_memory(memory: OutputMemory | None, outcome: Outcome) -> None`

Module-level function. Record an accepted output into memory, so a later record can reuse it as
context. A no-op unless memory is in use and the output is injectable.

**Args:** `memory: OutputMemory | None`; `outcome: Outcome`.

**Returns:** `None`.

**Raises:** none.

**Side effects:** if `memory is not None and outcome.output and outcome.status.is_injectable`
(i.e. `VERIFIED` or `PRODUCED`), calls `memory.record(outcome.record.source, outcome.output)` —
mutates the shared `OutputMemory` under its own lock. Otherwise does nothing.

---

## ragkit.harness.capture

The context-capture sink: records what one `Harness.process` call actually retrieved and
assembled, without ever triggering a retrieval of its own. A block or validator that already calls
a `ragkit.core.ports.Retriever` (`RetrievedBlock`, a recipe's grounding validator) writes its hits
here through `capture_retrieved`; nothing new is fetched for capture's sake. This lives in its own
module, separate from `.agents`, so both `.agents` and `.context.blocks` can import it without a
cycle (`agents` builds the context assembler, which is built from `context.blocks`).

### Capture

A fresh instance is created per `Harness.process` call and threaded through every round (produce,
mechanical check); never shared across records, so it needs no lock. Dataclass (`slots=True`, not
frozen — its fields are mutated in place).

**Attributes:**

- `passage: str = ""` — reflects whichever call wrote it *last*. By construction that is always
  the final `Harness.produce` of the round that produced the returned outcome, since production
  always precedes settling and `Harness.review`'s own prompt assembly does not write here.
- `retrieved: dict[str, Retrieved] = {}` (`field(default_factory=dict)`) — keyed by `chunk_id`,
  merged across every call within one `process()` call rather than reset per round: retrieval is
  deterministic on `record.source`, which does not change across repair/revision rounds, so every
  call within one record contributes the same hits and a later write simply confirms an earlier
  one.

#### `note_passage(self, passage: str) -> None`

**Args:** `passage: str`. **Returns:** `None`. **Raises:** none. **Side effects:** overwrites
`self.passage`.

#### `note_retrieved(self, hits: Sequence[Retrieved]) -> None`

**Args:** `hits: Sequence[Retrieved]`. **Returns:** `None`. **Raises:** none. **Side effects:**
for each hit, `self.retrieved[hit.chunk_id] = hit` — mutates `self.retrieved`, overwriting any
prior entry for the same `chunk_id` (harmless, since retrieval is deterministic per the class
docstring).

#### `capture_retrieved(context: Mapping[str, Any], hits: Sequence[Retrieved]) -> None`

Module-level function. Record hits a context block or validator already retrieved into the run's
capture sink, if one is wired into `context` under the `"capture"` key — a no-op otherwise (no
capture wired, or no hits). Never performs a retrieval of its own.

**Args:** `context: Mapping[str, Any]` — the shared block/validator context mapping; only its
`"capture"` key is read. `hits: Sequence[Retrieved]` — chunks to record.

**Returns:** `None`.

**Raises:** none.

**Side effects:** if `context.get("capture")` is a `Capture` instance and `hits` is non-empty,
calls `capture.note_retrieved(hits)` (see above). Otherwise nothing.

---

## ragkit.harness.context.assembler

The declarative context builder: an ordered list of blocks, assembled into one prompt passage
under a character budget. The list of blocks and the budget are config (`context.toml`); the
assembler renders each block in order, drops the ones with no content (no empty sections), and —
if the result exceeds the character budget — trims whole blocks by a configured priority rather
than truncating text mid-section. A block whose kind is not named in `trim_order` is never
dropped, so the essential sections survive any budget.

### ContextAssembler

Renders a record's configured context blocks into one passage, within a character budget. Not a
dataclass (plain class).

**Constructor:** `__init__(self, blocks: Sequence[_PlacedBlock], *, max_chars: int = 0,
trim_order: Sequence[str] = (), separator: str = "\n\n") -> None`
- `blocks` — ordered `(kind, ContextBlock)` pairs (the private `_PlacedBlock`; normally built by
  `load_context`, not constructed directly by a caller).
- `max_chars: int = 0` — character budget; `0` disables it (every block with content is shown).
- `trim_order: Sequence[str] = ()` — block kinds most-trimmable-first.
- `separator: str = "\n\n"` — joiner between rendered sections.

No validation is performed in the constructor itself (`load_context` validates `max_chars >= 0`
before calling it).

#### `assemble(self, record: Record, context: Mapping[str, Any]) -> str`

**Args:**
- `record: Record`.
- `context: Mapping[str, Any]` — the shared services mapping each block may read from: `lexicon`,
  `memory`, `retriever`, `sql_store`, `introspector`, `previous_attempt`, `stand_in`, `capture`
  (see `context.blocks`'s module docstring for the full list and their types).

**Returns:** `str` — every configured block's `render(record, context)` is called in order; a
result of `None`, or a result that is blank after `.strip()`, is dropped. The surviving
`(kind, text)` pairs are then passed through `self._fit(...)` (see below) and joined with
`self._separator`. Returns `""` if nothing survives.

**Raises:** propagates any exception a block's own `render()` raises — most notably
`ContextBlockError` from a block whose required external service (`retriever`, `sql_store`,
`introspector`) is not wired into `context` (see `context.blocks`).

**Side effects:** none of its own; a block's `render()` may have side effects (e.g.
`RetrievedBlock` issuing a live retrieval and writing into `context["capture"]`).

**Budget-fitting rule** (private `_fit`/`_most_trimmable`, documented here since it is load-bearing
behaviour of `assemble`): if `max_chars <= 0`, no trimming happens. Otherwise, while the assembled
length (section lengths plus `len(separator)` between them) exceeds `max_chars`, the "most
trimmable" rendered section is dropped — the one whose `kind` has the lowest rank in `trim_order`
(ties broken toward the *last* occurrence of that kind, so a repeated kind sheds its farthest
section first) — repeated until it fits or no remaining section's kind appears in `trim_order`, in
which case trimming stops even if still over budget (the guaranteed, non-trimmable sections
always survive).

#### `load_context(path: Path) -> ContextAssembler`

Module-level function. Build a `ContextAssembler` from a `context.toml`: `[[context.block]]`
entries in order, each selecting a registered block by `kind` and validated against that block's
own options, plus an optional `[context.budget]`.

**Args:** `path: Path` — a `context.toml` file.

**Returns:** `ContextAssembler` built from the file's blocks (resolved through the
`CONTEXT_BLOCKS` registry) and budget.

**Raises:** `ConfigError` for: the file not found or not valid TOML; unknown top-level keys (only
`context` is allowed) or unknown `[context]` keys (only `block`/`budget`); a `[[context.block]]`
entry with no `kind`; zero `[[context.block]]` entries defined at all; unknown `[context.budget]`
keys; a negative `budget.max_chars`. Also propagates whatever `CONTEXT_BLOCKS.create(kind, options,
path=path)` raises for an unknown `kind` or invalid per-block options (a `RegistryError`, or the
specific block's own `ContextBlockError`/`ValueError` from its `from_config`).

**Side effects:** none (pure construction); reads the file at `path`.

---

## ragkit.harness.context.blocks

Built-in context blocks, and the registry a config selects them through. A context block renders
one prompt section from the record and the run's shared services, or returns `None` when it has no
content — the no-empty-section rule lives here, so an absent block contributes nothing rather than
a dangling heading. The ordered list of blocks a run assembles is config (`context.assembler`); a
bespoke block is named by dotted path and resolved through `CONTEXT_BLOCKS` exactly like a
built-in.

The shared services a block may read from the `context` mapping (all optional): `lexicon` (the
established terminology, `list[Entry]`), `memory` (an `OutputMemory`), `retriever` (a
`ragkit.core.ports.Retriever`), `sql_store` (a read-only `ragkit.core.ports.SqlStore`),
`previous_attempt` (on revision, the `Attempt` being fixed), `stand_in` (a readable replacement for
a placeholder shown in a *neighbour* line), `capture` (the run's `Capture` sink, if wired).

**Module-level registry:** `CONTEXT_BLOCKS: Registry[ContextBlock]` — registers built-in and
third-party (via the `ragkit.context_blocks` entry-point group) context-block components.

**Strict option parsing (every block's `from_config`).** Each built-in block's `from_config` reads
its options through the strict `ragkit.core.config` readers (`read_string`, `read_int`,
`read_float`, `read_string_list`), so an option present with the wrong TOML type is refused with a
`ConfigError` rather than silently coerced (a non-integer `limit`, a `min_score` given as a string,
and so on). The per-block `from_config` "Raises" notes below list only each block's own
constructor-level validation (the `ContextBlockError`s); this `ConfigError` type-check applies to
every block ahead of its constructor and is not repeated per entry.

**The no-empty-section render contract, precisely:** a block returns `None` (never a heading with
nothing under it) whenever it has nothing to contribute for this record. This is enforced
differently depending on the block:

- `LiteralBlock` never returns `None` — its content is guaranteed non-empty *at construction*
  (`ContextBlockError` if configured with blank `text`), so it is always present once configured.
- `LexiconBlock`, `NeighboursBlock`, `RetrievedBlock`, `SqlRowsBlock`, `SchemaBlock`,
  `ReadingsBlock` each check for content (a lexicon hit, a neighbour line, a retrieval hit, a SQL
  row, a schema table, a meta key) and return `None` directly when none exists.
- `EstablishedBlock` additionally treats a recorded-but-*empty* output as no content — it filters
  on truthiness (`if (output := memory.get(line))`), not `is not None`, specifically because
  showing `"line -> "` with nothing after the arrow would teach the model that producing nothing
  is an acceptable answer.
- `PreviousAttemptBlock` omits its "your previous attempt" heading+target sub-section specifically
  when `attempt.target` is blank, even though `attempt` itself is present (a first-round
  content/truncation error can produce an `Attempt` with `target=""`); it still returns `None`
  outright when there is no `previous_attempt` in context at all (first pass).

Separately, a block whose **required external service** is simply absent from `context` — the
`retriever` for `RetrievedBlock`, the `sql_store` for `SqlRowsBlock`, the `introspector` for
`SchemaBlock` — treats that as a **misconfiguration**, not an empty section, and raises
`ContextBlockError` rather than returning `None` (each also `isinstance`-checks the wired object
against the corresponding port and raises if it doesn't satisfy it). A block whose service is
legitimately optional instead degrades silently: `LexiconBlock` treats an absent `lexicon` as an
empty sequence (`context.get("lexicon", ())`), and `EstablishedBlock` returns `None` outright when
`context.get("memory")` is `None` — neither raises.

### ContextBlockError(RagkitError)

Raised when a context block is misconfigured, or a service it needs is not wired in. No
attributes or methods beyond `RagkitError`.

### LiteralBlock

A fixed block of text (a grounding instruction, a format reminder). Implements
`ragkit.core.ports.ContextBlock`. Always present once configured (see the render contract above).
`CONFIG_KEYS = {"text", "heading"}`.

**Constructor:** `__init__(self, text: str, heading: str = "") -> None`. Raises `ContextBlockError`
("a literal context block needs non-empty 'text'") if `text.strip()` is empty.

#### `from_config(cls, options: Mapping[str, Any]) -> LiteralBlock`

**Args:** `options` — `text` (default `""`), `heading` (default `""`). **Returns:** `LiteralBlock`.
**Raises:** `ContextBlockError` via the constructor if `text` is blank.

#### `render(self, record: Record, context: Mapping[str, Any]) -> str | None`

Ignores both arguments (required only to satisfy the `ContextBlock` port; marked `# noqa: ARG002`
in source). **Returns:** the heading (if set) plus the configured text, joined by `_section`;
never `None`. **Raises:** none.

### LexiconBlock

The established terms whose *term* occurs in the input, rendered as "use these renderings".
Implements `ContextBlock`. `CONFIG_KEYS = {"heading", "limit"}`. Default heading: `"Established
terminology (use these renderings):"`.

**Constructor:** `__init__(self, heading: str = _DEFAULT_HEADING, limit: int = 12) -> None`. No
validation (unlike most other blocks, a non-positive `limit` is not rejected here).

#### `from_config(cls, options: Mapping[str, Any]) -> LexiconBlock`

**Args:** `options` — `heading`, `limit` (default `12`). **Returns:** `LexiconBlock`. **Raises:**
none.

#### `render(self, record: Record, context: Mapping[str, Any]) -> str | None`

Reads `context.get("lexicon", ())`, calls `relevant_entries(record.source, list(lexicon),
limit=self._limit)`. **Returns:** `None` if no term of the lexicon occurs in `record.source`;
otherwise one line per hit, `"  {term} -> {rendering}"` (plus `"  [{category}]"` if the entry has
one), under the heading. **Raises:** none of its own.

### NeighboursBlock

The surrounding lines (`context_before`/`context_after` in the record's meta) as one continuous
passage, masked, with this record's own `source` inserted unmasked in the middle — a record's line
is often a fragment split across boundaries, and a model reads continuous prose the way it read the
source. Implements `ContextBlock`. `CONFIG_KEYS = {"before", "after", "heading"}`.

**Constructor:** `__init__(self, before: int = 3, after: int = 2, heading: str = _DEFAULT_HEADING)
-> None`. Raises `ContextBlockError` ("neighbours block: before/after must be >= 0") if either is
negative.

#### `from_config(cls, options: Mapping[str, Any]) -> NeighboursBlock`

**Args:** `options` — `before` (default `3`), `after` (default `2`), `heading`. **Returns:**
`NeighboursBlock`. **Raises:** `ContextBlockError` via the constructor if `before`/`after` is
negative.

#### `render(self, record: Record, context: Mapping[str, Any]) -> str | None`

Reads `context.get("stand_in", "they")`; collects up to `before` items from `record.meta
["context_before"]` (the *last* `before` entries, i.e. closest to this record) and up to `after`
items from `record.meta["context_after"]` (the *first* `after` entries). **Returns:** `None` if
both are empty (an edge-of-document record); otherwise each neighbour line has its `[[N]]`
placeholders replaced with `stand_in` (via the private `_mask`, since a placeholder in a neighbour
line refers to *that* line's own lookup, not this record's), joined as `before-lines + record.
source (unmasked) + after-lines`, under the heading. **Raises:** none of its own.

### EstablishedBlock

Neighbour lines that already have a produced output (from `context["memory"]`, an `OutputMemory`),
paired with it — so the model sees how a neighbour was actually rendered, keeping terminology and
pronouns continuous. Implements `ContextBlock`. `CONFIG_KEYS = {"before", "after", "heading"}`.

**Constructor:** `__init__(self, before: int = 3, after: int = 2, heading: str = _DEFAULT_HEADING)
-> None`. Raises `ContextBlockError` ("established block: before/after must be >= 0") if either is
negative — the same guard as `NeighboursBlock` (a negative window would otherwise fall through
`_meta_strings`'s `limit <= 0` path and silently drop the established context).

#### `from_config(cls, options: Mapping[str, Any]) -> EstablishedBlock`

**Args:** `options` — `before` (default `3`), `after` (default `2`), `heading`. **Returns:**
`EstablishedBlock`. **Raises:** `ContextBlockError` via the constructor if `before`/`after` is
negative.

#### `render(self, record: Record, context: Mapping[str, Any]) -> str | None`

**Returns:** `None` immediately if `context.get("memory")` is `None` (no memory wired this run —
a legitimate absence, not a misconfiguration; unlike the "required service" blocks below, no
`ContextBlockError` is raised, and no `isinstance` check is performed against whatever object *is*
present under `"memory"`). Otherwise gathers the same before/after neighbour lines as
`NeighboursBlock`, looks each up via `memory.get(line)`, and keeps only lines whose recorded output
is truthy (see the render-contract note above for why). Returns `None` if nothing survives that
filter; otherwise renders `"  {masked line} -> {masked output}"` per surviving neighbour, under the
heading. **Raises:** none of its own; would propagate an exception from a non-conforming `memory`
object's `.get(...)`.

### RetrievedBlock

Reference examples retrieved for this record, as worked examples to align with. Implements
`ContextBlock`. Requires a `retriever` in `context`. `CONFIG_KEYS = {"heading", "k", "min_score"}`.

**Constructor:** `__init__(self, heading: str = _DEFAULT_HEADING, k: int = 3, min_score: float =
0.3) -> None`. Raises `ContextBlockError` if `k < 1` ("retrieved block: k must be >= 1") or
`min_score` not in `[0, 1]` ("retrieved block: min_score must be in [0, 1]").

#### `from_config(cls, options: Mapping[str, Any]) -> RetrievedBlock`

**Args:** `options` — `heading`, `k` (default `3`), `min_score` (default `0.3`). **Returns:**
`RetrievedBlock`. **Raises:** `ContextBlockError` via the constructor for an invalid `k`/`min_score`.

#### `render(self, record: Record, context: Mapping[str, Any]) -> str | None`

**Raises:** `ContextBlockError` ("a 'retrieved' context block is configured but no retriever was
wired into the run; supply one or remove the block") if `context.get("retriever")` is `None`; or
("the wired 'retriever' does not satisfy the Retriever port") if it is present but fails an
`isinstance(retriever, Retriever)` check. Otherwise propagates whatever `retriever.retrieve(...)`
itself raises.

**Returns:** calls `retriever.retrieve(record.source, k=self._k, min_score=self._min_score)`,
records the hits via `capture_retrieved(context, hits)`, then returns `None` if there are no hits
(no-empty-section), else `"  {hit.text}"` per hit under the heading.

**Side effects:** performs a live retrieval call every time `render()` runs (not cached across
repair/revision rounds within one `process()` call — repeated calls simply re-confirm the same
hits in the shared `Capture`, since retrieval is deterministic on `record.source`); writes into
`context["capture"]` if a `Capture` is wired.

### PreviousAttemptBlock

On revision, the exact attempt being fixed and the objections against it — so the model revises
rather than re-produces blind. Absent on the first pass. Implements `ContextBlock`.
`CONFIG_KEYS = {"heading"}`.

**Constructor:** `__init__(self, heading: str = "Your previous attempt:") -> None`. No validation.

#### `from_config(cls, options: Mapping[str, Any]) -> PreviousAttemptBlock`

**Args:** `options` — `heading`. **Returns:** `PreviousAttemptBlock`. **Raises:** none.

#### `render(self, record: Record, context: Mapping[str, Any]) -> str | None`

Ignores `record` (marked `# noqa: ARG002`). Reads `context.get("previous_attempt")` (an `Attempt |
None`). **Returns:** `None` if absent (first pass). Otherwise assembles up to three parts, joined
with `"\n\n"`: the heading+target section, included only when `attempt.target.strip()` is truthy
(see the render-contract note above); an issues section when `attempt.issues` is non-empty ("It
was sent back for these reasons. Fix exactly these and change nothing else that already works:"
plus a bullet per issue); a suggestions section when `attempt.suggestions` (read defensively via
`getattr(attempt, "suggestions", ())`, since the port doesn't guarantee the attribute) is
non-empty ("A reviewer proposed this rewrite; adopt whatever is right about it, but you own the
final output:" plus a bullet per suggestion). If none of the three parts apply (an `Attempt` with
blank `target`, no `issues`, no `suggestions`), the join yields `""`, which
`ContextAssembler.assemble` treats the same as `None` since it drops any text that is blank after
`.strip()`. **Raises:** none.

### SqlRowsBlock

Rows retrieved from an external read-only database for this record — the form-autofill task's
historical-record context. Implements `ContextBlock`. Requires a `sql_store` in `context`.
`CONFIG_KEYS = {"heading", "query", "param_keys", "limit"}`.

**Constructor:** `__init__(self, query: str, heading: str = "Relevant historical records:",
param_keys: Sequence[str] = (), limit: int = 20) -> None`. Raises `ContextBlockError` if
`query.strip()` is empty ("sql_rows block needs a non-empty 'query'") or `limit < 1` ("sql_rows
block: limit must be >= 1").

#### `from_config(cls, options: Mapping[str, Any]) -> SqlRowsBlock`

**Args:** `options` — `query`, `heading`, `param_keys` (tuple, default `()`), `limit` (default
`20`). **Returns:** `SqlRowsBlock`. **Raises:** `ContextBlockError` via the constructor.

#### `render(self, record: Record, context: Mapping[str, Any]) -> str | None`

**Raises:** `ContextBlockError` ("a 'sql_rows' context block is configured but no sql_store was
wired into the run") if `context.get("sql_store")` is `None`; or ("the wired 'sql_store' does not
satisfy the SqlStore port") if present but not an `isinstance(store, SqlStore)`; or ("sql_rows
block: record ... is missing param_key(s) ... needed by the query; a missing bind would silently
match nothing"), naming the record and the missing keys, if any declared `param_key` is absent from
`record.meta`. This last check is a documented fix: a declared `param_key` missing from the record
previously bound SQL NULL silently, and `= NULL` matches nothing, so the block rendered no rows —
indistinguishable from a genuine no-match, the historical context silently dropped. Otherwise
propagates whatever `store.query(...)` raises.

**Returns:** builds `params = [record.meta[key] for key in self._param_keys]` (direct indexing,
safe because every key is guaranteed present by the missing-key check above), then runs
`SELECT * FROM (<query, trailing ';' stripped>) LIMIT ?` with `[*params, self._limit]` — the row
limit is applied by the database itself via the wrapping subquery, not by slicing the full result
set in Python (a documented fix: the previous form fetched every matching row across the port and
discarded all but `limit`, so a broad query materialised its whole result set just to show twenty
lines). Returns `None` if no rows come back; otherwise `"  key=value, key=value, ..."` (`!r` repr
of each value) per row, under the heading.

**Side effects:** issues a live SQL query every `render()` call.

### SchemaBlock

The introspected schema of the external database, rendered as `CREATE TABLE`-like text, so the
NL->SQL producer sees what tables and columns exist. Implements `ContextBlock`. Requires an
`introspector` in `context`. `CONFIG_KEYS = {"heading"}`.

**Constructor:** `__init__(self, heading: str = "Database schema (tables and their columns):") ->
None`. No validation.

#### `from_config(cls, options: Mapping[str, Any]) -> SchemaBlock`

**Args:** `options` — `heading`. **Returns:** `SchemaBlock`. **Raises:** none.

#### `render(self, record: Record, context: Mapping[str, Any]) -> str | None`

Ignores `record`. **Raises:** `ContextBlockError` ("a 'schema' context block is configured but no
introspector was wired into the run") if `context.get("introspector")` is `None`; or ("the wired
'introspector' does not satisfy the port") if present but not an
`isinstance(introspector, SchemaIntrospector)`. Otherwise propagates whatever
`introspector.schema()` raises.

**Returns:** calls `introspector.schema()` (`Mapping[str, Sequence[tuple[str, str]]]`); `None` if
empty; otherwise one line per table, `"  {table}({col1} {type1}, {col2} {type2}, ...)"`, under the
heading.

**Side effects:** calls the introspector every `render()` call.

### ReadingsBlock

Named fields of *this record's own* request, rendered into the prompt — the sensor telemetry and
fault codes a decision task is given (`record.meta`). Unlike `SqlRowsBlock` (rows fetched from a
database) this reads only the record itself, so it needs no wired service. Implements
`ContextBlock`. `CONFIG_KEYS = {"heading", "keys"}`.

**Constructor:** `__init__(self, keys: Sequence[str] = (), heading: str = "Reported readings and
codes:") -> None`. No validation.

#### `from_config(cls, options: Mapping[str, Any]) -> ReadingsBlock`

**Args:** `options` — `keys` (tuple, default `()`), `heading`. **Returns:** `ReadingsBlock`.
**Raises:** none.

#### `render(self, record: Record, context: Mapping[str, Any]) -> str | None`

Ignores `context` (marked `# noqa: ARG002`). `keys = self._keys or tuple(record.meta.keys())` —
an empty configured `keys` means "show every meta field". **Returns:** `None` if no configured key
is present in `record.meta` (absent keys are silently skipped, not an error); otherwise `"  {key}:
{rendered_value}"` per present key, under the heading, where the private `_render_value` renders a
`Mapping` as `"k=v, k=v"`, a `list`/`tuple` as a comma-joined line, and anything else via `str()`.
**Raises:** none.

#### `register_builtins() -> None`

Module-level function. Register the built-in blocks. Idempotent, so importing this module more
than once is harmless; the registry refuses a genuine name collision.

**Args:** none. **Returns:** `None`.

**Raises:** would propagate `RegistryError` from `CONTEXT_BLOCKS.register` if a name were already
registered to a *different* class (not expected in normal use; re-registering the same class under
the same name is a no-op).

**Side effects:** registers all nine built-in block classes (`literal`, `lexicon`, `neighbours`,
`established`, `retrieved`, `previous_attempt`, `sql_rows`, `schema`, `readings`) into the
module-level `CONTEXT_BLOCKS` registry. Called automatically once at import time (invoked at the
bottom of this module), so a caller normally never calls it directly.

---

## ragkit.harness.memory

Outputs already produced this run, looked up by the input that produced them. Two things need
this: duplicate inputs are produced once and shared, and a neighbouring record that has already
been produced can be shown to the model *with* its output — not only in the input it was extracted
as — so terminology and phrasing stay continuous across a scene. The masked input text is a sound
key: two records share a key only when they present the model with the identical problem.

### OutputMemory

Accepted outputs, keyed by the input that produced them. Safe to share across worker threads: the
runner records from its own thread while workers read, both through the lock. Not a dataclass
(plain class); also supports `len(memory)` via `__len__` (not part of the documented checklist, but
present and lock-guarded like every other access).

**Constructor:** `__init__(self, initial: dict[str, str] | None = None) -> None`. `initial` — an
optional pre-seeded `{source: output}` mapping (e.g. built from a prior run's results), copied into
the private `self._by_source: dict[str, str]`. Also creates `self._lock: threading.Lock`.

#### `from_records(cls, records: Iterable[Record]) -> OutputMemory`

Build from previously journalled results. Only injectable records (verified/produced) with an
output contribute; a rejected record's output failed its checks, so offering it as context would
spread a known-bad rendering.

**Args:** `records: Iterable[Record]`.

**Returns:** `OutputMemory` — a new instance populated by calling `.record(record.source, record.
output)` for every record where `record.output` is truthy and `record.status.is_injectable`.

**Raises:** none.

**Side effects:** none on its inputs; builds and returns a new instance.

#### `get(self, source: str) -> str | None`

**Args:** `source: str` — the exact masked input text key. **Returns:** the previously recorded
output for `source`, or `None` if none recorded. **Raises:** none. **Side effects:** none (read
under `self._lock`).

#### `record(self, source: str, output: str) -> None`

**Args:** `source: str`, `output: str`. **Returns:** `None`. **Raises:** none. **Side effects:**
sets `self._by_source[source] = output` under `self._lock`, overwriting any prior entry for the
same `source` (last writer wins).

---

## ragkit.harness.roles

Who the personas are and how much room they get, loaded from `personas.toml`. A **persona** is an
LLM acting under a fixed system prompt — the producer that generates the output, and the reviewers
that judge it. This is distinct from the *harness* (the loop that sequences them) and from a chat
message's `role` field; the naming keeps all three apart. Every persona names the logical model it
runs on, so a run can put its cheap reviewers on a small model and its producer on a larger one;
personas naming the same model share it. The panel consults reviewers in file order and stops at
the first objection, so order them most-decisive-first. A `from_rules` reviewer judges against the
rule set's advisory criteria rather than its own instructions, keeping policy in one place.

### Leniency

How tolerant the *console* is of a reviewer's unusable replies before it warns. Every unusable
reply is always recorded; this bounds only the *printed* warning, so a flaky reviewer neither
hides a systemic failure nor buries it in noise. Within a rolling window of the last `window`
replies, up to `max_bad` unusable ones pass without a console warning. Frozen dataclass
(`slots=True`).

**Attributes:**

- `window: int = 20` — size of the rolling reply-history window per reviewer. Must be `>= 1`
  (`ValueError` otherwise).
- `max_bad: int = 2` — how many unusable replies within the window are tolerated before the
  console is warned. Must be `>= 0` (`ValueError` otherwise).

#### `surfaces(self, bad_in_window: int) -> bool`

Whether a bad reply, given the bad count now in the window, reaches the console.

**Args:** `bad_in_window: int` — the count of unusable replies currently in the rolling window
(including the one just registered). **Returns:** `bool` — `bad_in_window > self.max_bad`.
**Raises:** none.

### Limits

Output token budgets. The producer's scales with the input; each reviewer has a ceiling. Frozen
dataclass (`slots=True`).

**Attributes:**

- `produce_tokens_per_source_char: int = 8` — the producer's token budget per input character.
  Must be `>= 1`.
- `produce_tokens_floor: int = 256` — minimum producer token budget regardless of input length.
  Must be `>= 1`.
- `produce_tokens_ceiling: int = 1024` — maximum producer token budget regardless of input length.
  Must be `>= 1`, and must be `>= produce_tokens_floor` (`ValueError` otherwise: "...so every
  budget would be the ceiling").
- `review_tokens: int = 1024` — default reviewer token ceiling, used when a persona does not set
  its own `max_tokens`. Must be `>= 1`.

#### `produce_budget(self, source_length: int) -> int`

Token ceiling for producing from a source of `source_length` characters, bounded at both ends: the
floor keeps a short input able to produce a full answer, the ceiling stops an unusually long one
asking for an unbounded response.

**Args:** `source_length: int` — typically `len(record.source)`.

**Returns:** `int` — `max(produce_tokens_floor, min(produce_tokens_ceiling, source_length *
produce_tokens_per_source_char))`.

**Raises:** none (no validation of `source_length`; a negative value clamps to the floor rather
than raising).

#### `review_budget(self, persona: Persona) -> int`

**Args:** `persona: Persona` — the reviewer persona. **Returns:** `int` — `persona.max_tokens` if
it is set, else `self.review_tokens`. **Raises:** none.

### Persona

One member of the run: the producer, or a reviewer. Frozen dataclass (`slots=True`).

**Attributes:**

- `id: str` — the persona's identifier; also the `role` tag sent to the LLM client and the key
  into `Harness._reply_history` / `Review.role`.
- `kind: str` — `"producer"` or `"reviewer"`; validated in `__post_init__` against
  `{"producer", "reviewer"}` (`ValueError` otherwise).
- `model: str` — the logical model name, resolved through `ModelPool.client_for`.
- `instructions: str = ""` — this persona's system-prompt text. Validated by `Persona.
  __post_init__`: a non-`from_rules` persona with blank `instructions` raises `ValueError` ("has no
  instructions and does not set from_rules, so it would act against nothing"), and setting both
  `from_rules` and non-empty `instructions` raises `ValueError` ("sets both from_rules and
  instructions; one would be silently ignored"). These invariants live on the type, so an invalid
  `Persona` cannot be constructed directly (not only via the loader); the config loader
  (`_one_persona`) relies on them and wraps the `ValueError` into a `ConfigError` naming the persona
  and file.
- `from_rules: bool = False` — reviewer-only: build instructions from the rule set's advisory
  criteria instead of its own (see `Harness._rule_instructions`). Also validated by `Persona.
  __post_init__`: `from_rules=True` on a non-reviewer raises `ValueError` ("from_rules applies only
  to a reviewer"), and it is mutually exclusive with non-empty `instructions` (see above).
- `max_tokens: int | None = None` — an explicit override of the reviewer token ceiling. If set,
  must be `>= 1` (`ValueError` otherwise, raised by `__post_init__`).
- `leniency: Leniency = Leniency()` — per-reviewer leniency window. The class-level default is a
  single shared `Leniency()` instance; safe only because `Leniency` is itself frozen/immutable
  (an ordinary dataclass would refuse a mutable default here).
- `sampling: SamplingParams = field(default_factory=SamplingParams)` — per-request decode settings.
  The config loader (`load_panel`) fills a kind-appropriate default temperature (0.3 for a
  producer, 0.0 for a reviewer); a directly-constructed `Persona` instead gets `SamplingParams`'s
  own default (temperature `0.2`).

#### `is_reviewer` (property, not a callable method)

**Returns:** `bool` — `self.kind == "reviewer"`. Accessed as `persona.is_reviewer` (no
parentheses); used e.g. by the config loader to sort personas into producer/reviewer lists.
**Raises:** none.

### Panel

The producer, the reviewers that judge it, and the budgets they operate under. Frozen dataclass
(`slots=True`).

**Attributes:**

- `producer: Persona` — the single producer persona.
- `reviewers: tuple[Persona, ...]` — reviewers, consulted in this order; the panel stops at the
  first objection.
- `limits: Limits = field(default_factory=Limits)` — token budgets.
- `leniency: Leniency = field(default_factory=Leniency)` — the panel-wide default leniency, used
  by any persona that does not set its own `[persona.leniency]`.
- `max_revisions: int = 2` — revision rounds per record in `Harness.process`. Must be `>= 0`.
- `max_repairs: int = 2` — mechanical-repair rounds per generation in `Harness._generate`. Must be
  `>= 0`.
- `repair_truncated_json: bool = True` — whether `Harness._generate` attempts to recover an
  `LlmIncompleteJsonError`'s truncated envelope as a last-resort candidate rather than treating it
  as an ordinary content error.

#### `models_in_use(self) -> set[str]`

Every logical model named by a persona — the set the pool's thrash guard checks.

**Returns:** `set[str]` — `{self.producer.model, *(r.model for r in self.reviewers)}`. **Raises:**
none.

#### `load_panel(path: Path, substitutions: dict[str, str] | None = None) -> Panel`

Module-level function. Load the personas and budgets, filling any `{placeholder}` tokens by literal
replacement (not `str.format`, because instructions are prose that legitimately contains braces).
An unfilled placeholder is an error, never a brace reaching the model.

**Args:** `path: Path` — a `personas.toml` file. `substitutions: dict[str, str] | None = None` —
`{key: value}` pairs substituted for `{key}` tokens (matched by the regex `\{([a-z_][a-z0-9_]*)\}`
— deliberately narrow, lowercase identifiers only, so ordinary prose braces are never mistaken for
a placeholder) inside every persona's `instructions`.

**Returns:** `Panel`.

**Raises:** `ConfigError` for, among others: the file not found or not valid TOML; unknown
top-level keys (only `persona`/`limits`/`revision`/`leniency` allowed); an instruction referring to
an unknown `{placeholder}`; invalid `[leniency]`/`[limits]`/`[revision]` keys or values (each
wrapped from the underlying dataclass's `ValueError`); a `[[persona]]` entry with no/duplicate
`id`, an unknown key, an invalid `kind`, no/blank `model`, `from_rules=True` on a non-reviewer,
non-string `instructions`, both `from_rules` and non-blank `instructions` set, blank
`instructions` with `from_rules` unset, or an invalid `[persona.sampling]` key/value; not exactly
one producer persona; zero reviewer personas ("every output would be accepted unread. Set
`revision.max_revisions` to 0 if that is genuinely what you want"); an invalid final `Panel`
construction (`max_revisions`/`max_repairs` negative).

**Side effects:** none (pure construction); reads the file at `path`.

---

## ragkit.harness.rules

Configurable policy over an acceptable output, in three tiers: **numeric** (`[limits]`, line width
and its tolerance, whether an empty output is allowed), **pattern** (`[[forbidden]]`, regexes an
output must not match, each with a reason), and **prose** (`[[advisory]]` and `[style]`, criteria a
reviewer persona judges, because judging them is exactly what a language model is for). The
pattern and numeric tiers are decided by code (`ragkit.harness.validators`) and cost no GPU time;
the prose tier reaches the review panel. Two further knobs generalise hard-won translation
behaviour: `keep_flagged_rules` names the blocking rules that mean "imperfect, but still worth
keeping" rather than "unusable" (kept and flagged when the repair budget runs out instead of
discarded, the `on_exhausted` distinction), and `lexicon_severity` sets whether a missing
established term blocks or merely warns.

**Module-level constant:**

- `MIN_LINE_COLUMNS = 20` — below this, a "line" cannot hold even a short clause, so a smaller
  `max_line_columns` is treated as a mistyped value rather than a tight one — and would reject
  every output offered against it. Enforced by `RuleSet.__post_init__`.

### RuleSet

Policy governing an acceptable output. Frozen dataclass (`slots=True`).

**Attributes:**

- `max_line_columns: int = 110` — project-wide guideline column width, checked per output line
  (only used when a record carries no per-record hard budget; see
  `ragkit.harness.validators.check_mechanical`). Must be `>= MIN_LINE_COLUMNS` (`ValueError`
  otherwise).
- `max_columns_tolerance: float = 0.12` — how far an output may exceed a record's *hard* column
  budget (`record.meta["max_columns"]`, distinct from `max_line_columns`) and still be accepted as
  a warning rather than an error. Must be a fraction in `[0.0, 1.0]`.
- `require_nonempty: bool = True` — whether a non-blank input producing a blank output is an
  error.
- `forbidden_patterns: tuple[tuple[str, str], ...] = ()` — `(regex, reason)` pairs the output must
  not match: model failure modes (a preamble, a refusal, a label prefix), not domain rules.
- `style_directives: tuple[str, ...] = ()` — short directives injected into every persona's system
  prompt; anything long or nuanced belongs in an advisory criterion instead.
- `advisory_rules: tuple[tuple[str, str], ...] = ()` — `(id, description)` criteria a `from_rules`
  reviewer judges, all in one pass.
- `lexicon_severity: Severity = Severity.WARNING` — whether a missing established term is an
  error (blocks) or a warning (does not).
- `keep_flagged_rules: frozenset[str] = frozenset()` — blocking rule ids whose failure, once the
  repair budget is spent, keeps the output (flagged for review) rather than rejecting it, because
  the only alternative is showing nothing. Empty means every blocking rule rejects on exhaustion.

#### `load(cls, path: Path) -> RuleSet`

**Args:** `path: Path` — a `rules.toml` file.

**Returns:** `RuleSet`.

**Raises:** `ConfigError` for: the file not found or not valid TOML; unknown top-level keys (only
`limits`/`lexicon`/`style`/`forbidden`/`advisory` allowed); unknown `[limits]`/`[style]`/
`[lexicon]` keys; a non-integer `limits.max_line_columns`; a `[[forbidden]]` entry with no/blank
`pattern` or an invalid regex; an `[[advisory]]` entry missing `id` or `description`; an invalid
`lexicon.severity` (must be `"error"` or `"warning"`); any `ValueError` from the final `RuleSet`
construction (`max_line_columns` below `MIN_LINE_COLUMNS`, or `max_columns_tolerance` outside
`[0, 1]`), wrapped as `ConfigError`.

**Side effects:** none (pure construction); reads the file at `path`.

---

## ragkit.harness.runner

Resumable batch execution over a run store. A full run can be tens of thousands of records, each
several sequential model calls, so it spans hours and will be interrupted. Durability is a design
requirement: each result is appended to the injected `ragkit.core.ports.RunStore` as it completes,
in one transaction — restarting reads the store's own `RunStore.pending()` and skips what already
has a result, and a crash mid-write leaves a complete result or none at all (the store's job, not
this module's). Duplicate inputs are grouped so identical text is produced once and shared, and
only this thread appends results or touches the progress counters, so the append-per-result
durability needs no locking even under concurrent workers.

`Clock = Callable[[], float]` — injected so a test can drive time deterministically; defaults to
the monotonic wall clock.

### RunnerError(RagkitError)

The run cannot start or continue. No attributes or methods beyond `RagkitError`.

### Progress

Live counters for a job that may span many runs. `done` counts what *this* run produced (the rate
is measured from it); `already_done` is what earlier runs finished, so the reported position
describes the whole job rather than this session. Ordinary (non-frozen) dataclass.

**Attributes:**

- `total: int` — the job's total record count.
- `already_done: int = 0` — completed by earlier runs, before this call.
- `done: int = 0` — completed by this call so far.
- `verified: int = 0`, `produced: int = 0`, `rejected: int = 0`, `skipped: int = 0` — per-status
  counters, incremented by `record(...)`.
- `started_at: float = 0.0` — the clock's reading when the run began.
- `_clock: Clock = lambda: 0.0` (`field(repr=False, compare=False)`) — the injected time source
  used by `elapsed`. Leading underscore despite being a dataclass field with no `__init__`
  override — present in the AST checklist because dataclass field enumeration includes it, but not
  part of the public API (excluded from `repr`/equality).

#### `record(self, outcome: Outcome) -> None`

**Args:** `outcome: Outcome`.

**Returns:** `None`.

**Raises:** `RunnerError` if `outcome.status` is not one of `VERIFIED`/`PRODUCED`/`REJECTED`/
`SKIPPED` (i.e. `PENDING`, which `Harness.process` never actually returns, so this is a defensive
branch: `f"Progress.record: outcome for record {record_id!r} carries non-terminal status
{unhandled!r}, which has no counter"`). Counted **last**, so a status with no bucket leaves every
counter untouched rather than a `done` that outruns its categories.

**Side effects:** increments the matching status counter, then `self.done += 1`.

#### `elapsed` (property)

**Returns:** `float` — `self._clock() - self.started_at`. **Raises:** none.

#### `completed` (property)

**Returns:** `int` — `self.already_done + self.done`. **Raises:** none.

#### `remaining` (property)

**Returns:** `int` — `max(0, self.total - self.completed)`. **Raises:** none.

#### `summary(self) -> str`

**Returns:** `str` — a one-line human-readable summary: `"{completed}/{total} done, {remaining}
left (verified {verified} | needs review {produced} | rejected {rejected})"`. **Raises:** none.

#### `group_duplicates(records: Iterable[Record]) -> list[tuple[Record, tuple[Record, ...]]]`

Module-level function. Group records that should receive one output, keeping input order. Keyed on
`(source, discriminator)` where the discriminator is `meta['speaker']` if present, so two speakers
of the same line can still diverge; members share the first member's output, and each is
journalled individually so resume and reinjection stay per-record.

**Args:** `records: Iterable[Record]`.

**Returns:** `list[tuple[Record, tuple[Record, ...]]]` — one `(representative, members)` pair per
distinct `(source, speaker)` key, `representative` being the first record seen for that key, in
first-seen order.

**Raises:** none.

**Side effects:** none.

#### `run_batch(harness: Harness, records: Iterable[Record], run_store: RunStore, *, total: int | None = None, already_done: int = 0, on_progress: Callable[[Progress], None] | None = None, report_every: int = 25, concurrency: int = 1, clock: Clock = lambda: 0.0, install_signal_handlers: bool = True) -> Progress`

Module-level function. Produce for `records`, appending each result to `run_store` as it
completes. `concurrency` records are produced at once against the same pool; the personas within
one record still run in sequence. Only this thread appends results or touches the counters.
`clock` is injectable for deterministic tests; `install_signal_handlers` is off in a worker thread
where signals cannot be caught.

**Args:**
- `harness: Harness` — runs one record (`harness.process`) at a time, per worker.
- `records: Iterable[Record]` — consumed into a list, then grouped via `group_duplicates`.
- `run_store: RunStore` — durable sink; each completed group's members are appended here as
  `RunResult`s.
- `total: int | None = None` — keyword-only; defaults to `len(records) + already_done`.
- `already_done: int = 0` — records finished by earlier runs (reporting only).
- `on_progress: Callable[[Progress], None] | None = None` — called every `report_every`
  completions, and once more unconditionally at the end.
- `report_every: int = 25` — must be `>= 1`.
- `concurrency: int = 1` — `ThreadPoolExecutor(max_workers=concurrency)` worker count; must be
  `>= 1`.
- `clock: Clock = lambda: 0.0` — time source for `Progress.started_at`/`elapsed`.
- `install_signal_handlers: bool = True` — install cooperative SIGINT/SIGTERM handling for the
  duration of the call; must be `False` when called from a non-main thread (Python can only
  install signal handlers on the main thread).

**Returns:** `Progress` — the final counters.

**Raises:**
- `RunnerError` — `concurrency < 1`, or `report_every < 1`.
- Whatever a worker's `harness.process(...)` raised (its `Future.result()`), **re-raised only
  after every already-in-flight result has been appended to `run_store`** — so an infrastructure
  failure (a bare `LlmError`, or the defensive `AssertionError` from `Harness.process`) never
  discards completed work. Per-record plugin/validator defects never reach here, since `Harness.
  process` itself contains those to a `REJECTED` `Outcome` (see `agents.Harness.process`).

**Side effects:**
- For each completed duplicate group, appends one `RunResult` per member (via the private
  `_as_run_result`, which projects `outcome.applied_to(member)` plus `context_passage`,
  `retrieved` (as `RetrievedRef`), `reviews` (flattened via `dataclasses.asdict`), `violations`,
  `rounds`, `error`) to `run_store.append_result(...)`, and calls `progress.record(outcome)` once
  per member.
- Calls `learn_memory(harness.memory, outcome)` once per completed group.
- Calls `on_progress(progress)` periodically (`progress.done % report_every == 0`) and once more
  at the end regardless.
- Installs OS signal handlers for `SIGINT`/`SIGTERM` for the call's duration (via the private
  `_Interruptible`, restoring the previous handlers on exit) unless `install_signal_handlers=
  False` (in which case the private `_NullInterrupt` no-op stand-in is used instead). On the first
  signal, stops submitting new work and lets in-flight work finish, then returns normally — it
  does not raise.
- Creates and tears down a `ThreadPoolExecutor(max_workers=concurrency)`.

---

## ragkit.harness.schemas

Ready-made `ragkit.core.ports.OutputSchema` implementations. An output schema describes what the
producer generates (a JSON schema the backend constrains or prompts for) and how to pull the
output string out of the parsed reply. Most tasks produce a single string field — a translation, a
SQL statement — for which `JsonFieldSchema` is enough; a recipe with a richer shape supplies its
own component through the `OUTPUT_SCHEMAS` registry.

**Module-level registry:** `OUTPUT_SCHEMAS: Registry[OutputSchema]` — registers `"json_field"` ->
`JsonFieldSchema` and `"form"` -> `FormSchema` as built-ins, plus any third-party component
published under the `ragkit.output_schemas` entry-point group.

### JsonFieldSchema

A one-string-field output: `{"<field>": "<the output>"}`. The common case. Implements
`ragkit.core.ports.OutputSchema`. Not a dataclass. `CONFIG_KEYS = {"field", "description"}`.

**Constructor:** `__init__(self, field: str = "output", description: str = "") -> None`. Raises
`ValueError` ("JsonFieldSchema needs a non-empty field name") if `field.strip()` is empty. Sets
`self.name = field` (the `OutputSchema.name` attribute) and the private `self._field`,
`self._description`.

#### `from_config(cls, options: Mapping[str, Any]) -> JsonFieldSchema`

**Args:** `options` — `field` (default `"output"`), `description` (default `""`). **Returns:**
`JsonFieldSchema`. **Raises:** `ValueError` via the constructor if the resolved field name is
blank.

#### `json_schema(self) -> dict[str, Any]`

**Returns:** `dict[str, Any]` — `{"type": "object", "additionalProperties": False, "required":
[field], "properties": {field: {"type": "string" [, "description": ...]}}}`. **Raises:** none.

#### `extract(self, reply: Mapping[str, Any]) -> str`

**Args:** `reply: Mapping[str, Any]` — the parsed producer JSON reply. **Returns:** `str(reply
[self._field])`. **Raises:** `KeyError` if `self._field` is absent from `reply` (the JSON schema
declares it `required`, but nothing here re-validates that at extraction time; this is exactly the
kind of exception `Harness.process`'s generic `except Exception` clause is built to contain to one
record).

### FormField

One field of a form to fill: its name, JSON type, an optional prompt description, and whether the
model must supply it. `array` fields are arrays of strings. Frozen dataclass (`slots=True`).

**Attributes:**

- `name: str` — must be non-empty (`ValueError` otherwise).
- `type: str = "string"` — one of `{"string", "integer", "number", "boolean", "array"}`
  (`ValueError` otherwise); `"array"` means an array of strings.
- `description: str = ""` — optional prompt description surfaced in the JSON schema.
- `required: bool = True` — whether the producer must supply this field.

No public methods beyond the dataclass-generated ones.

### FormSchema

A multi-field form output: `{field1: ..., field2: ...}`. Each field is declared with a name and
JSON type, so the producer is asked (or grammar-constrained) to fill exactly those fields.
`extract` returns the filled form as a **canonical JSON string** (sorted keys), which becomes the
record's single `output` — it round-trips through the journal and can be scored field by field by
a validator or an eval. Field *value* checks (type conformance, enum membership, ranges) are a
validator's job, layered on top. Implements `ragkit.core.ports.OutputSchema`. Not a dataclass.
`CONFIG_KEYS = {"name", "fields"}`.

**Constructor:** `__init__(self, fields: Sequence[FormField], name: str = "form") -> None`. Raises
`ValueError` if `fields` is empty ("FormSchema needs at least one field") or contains a duplicate
field name ("FormSchema has a duplicate field {name!r}"). Sets `self.name = name` and
`self._fields = tuple(fields)`.

#### `from_config(cls, options: Mapping[str, Any]) -> FormSchema`

**Args:** `options` — `fields` (required, a non-empty list of `{name, [type], [description],
[required]}` mappings), `name` (default `"form"`).

**Returns:** `FormSchema`.

**Raises:** `ValueError` if `fields` is missing or not a non-empty list ("FormSchema needs a
non-empty 'fields' array"), if an entry is not a `Mapping` or lacks `"name"` ("each form field
needs at least a 'name'"), or whatever `FormField.__post_init__`/`FormSchema.__init__` raise for
a bad type, duplicate name, or empty field list. Each field's `name`/`type`/`description` and the
top-level `name` are read through the strict `read_string` reader and `required` through
`read_bool`, so a mistyped field option — e.g. a non-string `type`, or a `required` that is not a
boolean — raises `ConfigError` rather than being coerced. (`JsonFieldSchema.from_config`, by
contrast, still uses plain `str(...)` coercion — see above.)

#### `json_schema(self) -> dict[str, Any]`

**Returns:** an object schema with one property per field (`type` = the field's declared type;
`"array"` fields additionally get `"items": {"type": "string"}`; `"description"` included when
set); `"required"` lists the names of fields with `required=True`. **Raises:** none.

#### `extract(self, reply: Mapping[str, Any]) -> str`

**Args:** `reply: Mapping[str, Any]`. **Returns:** `json.dumps({f.name: reply.get(f.name) for f in
fields}, sort_keys=True, ensure_ascii=False)` — only the declared fields (extra reply keys are
dropped), a missing field becomes `None`, output is deterministic (sorted keys) so two equal forms
serialise identically. **Raises:** none in practice (all values come from an already-parsed JSON
reply, which `json.dumps` can always re-serialise).

---

## ragkit.harness.validators

Mechanical (code-decidable) checks over a produced output, and the pluggable-validator seam. Two
kinds meet here. The **built-in** checks are intrinsic to the harness and driven entirely by the
`ragkit.harness.rules.RuleSet` (emptiness, placeholder integrity, control characters, forbidden
patterns, established-term presence, column budget) — cheap, deterministic, and settled before any
GPU time. The **pluggable** checks are task-specific and satisfy the `ragkit.core.ports.Validator`
port, resolved through `VALIDATORS`: an untranslated-echo detector, a generated-SQL safety check, a
form-field-type check. A `ValidatorPipeline` runs the built-ins and then the pluggables over one
output. Two disciplines the pluggable validators must honour, promoted to framework rules because
they were the hard-won part of the original engine: a blocking check must **abstain unless it has
positive evidence** (never fire on input too thin to judge), and a check that **cannot be
evaluated is skipped, never reported as passed**.

**Module-level registry:** `VALIDATORS: Registry[Validator]` — registers third-party, task-specific
`Validator` components (via the `ragkit.validators` entry-point group); built-in mechanical checks
are not registered here, since they are driven by the `RuleSet` directly rather than being
pluggable components.

#### `check_mechanical(record: Record, output: str, ruleset: RuleSet, *, required_terms: Sequence[str] = (), max_columns: int | None = None) -> list[Violation]`

Every code-decidable rule violation in `output` against `ruleset`.

**Args:**
- `record: Record` — `record.source` drives the emptiness and placeholder checks.
- `output: str` — the candidate to check.
- `ruleset: RuleSet` — drives `require_nonempty`, `forbidden_patterns`, `max_line_columns` /
  `max_columns_tolerance`, `lexicon_severity`.
- `required_terms: Sequence[str] = ()` — established-term renderings this input triggered that
  must appear (case-insensitively) in `output`.
- `max_columns: int | None = None` — the record's own hard column budget (from `record.meta`),
  distinct from `ruleset.max_line_columns`'s project-wide guideline: a hard budget blocks past its
  tolerance, the guideline only warns.

**Returns:** `list[Violation]`, built in this order:
- `"nonempty"` (`ERROR`) — `ruleset.require_nonempty` and `record.source.strip()` is non-blank but
  `output.strip()` is blank.
- `"placeholders"` (`ERROR`) — the set of `[[N]]` indices in `output` differs from those in
  `record.source`, or (a separate check) the same index repeats in `output`.
- `"control_character"` (`ERROR`) — `output` contains a raw disallowed control character (any of
  `\x00`-`\x08`, `\x0b`, `\x0c`, `\x0e`-`\x1f`, `\x7f`; tab/LF/CR are deliberately allowed here,
  since whether a *newline* is acceptable is task-specific and left to a recipe's own validator).
- `"line_width"` (`ERROR` or `WARNING`) — see the width rule below.
- `"forbidden"` (`ERROR`) — per `ruleset.forbidden_patterns` regex that matches `output` (message
  includes the pattern's `reason` if one was given).
- `"lexicon"` (`ruleset.lexicon_severity`) — per `required_terms` entry that does not appear
  (case-insensitively) in `output`.

**Width rule** (private `_width_violations`, documented here as part of `check_mechanical`'s
contract): if `max_columns` is given, `display_columns(output)` is measured over the **whole**
payload — within budget: no violation; over budget but within `ruleset.max_columns_tolerance`: one
`WARNING`; past tolerance: one `ERROR` ("say it more briefly"). Otherwise (no per-record hard
budget), each `"\n"`-split line of `output` is measured against `ruleset.max_line_columns`
independently, and each over-long line produces its own `WARNING` (never `ERROR`) naming its
1-based line number — splitting on the real newline character, since `output` has already been
JSON-decoded by the time this runs.

**Raises:** propagates `re.error` if one of `ruleset.forbidden_patterns`'s regexes is invalid
(`RuleSet.load` already validates every pattern compiles at load time, so this only matters for a
hand-built `RuleSet`).

**Side effects:** none.

### ValidatorPipeline

Runs the built-in mechanical checks and then the pluggable validators over one output. The
built-ins are driven by the `ruleset` and the `lexicon` (established terms are required to
appear); the pluggables are task-specific components. `lexicon_limit` caps how many established
terms a single output is required to carry, so a long input cannot demand an unbounded set. The
shared context handed to each pluggable carries the ruleset, the required terms, the record's
column budget, and (when the caller passes one) the run's `capture` sink, so a validator need not
recompute them. Not a dataclass (plain class).

**Constructor:** `__init__(self, ruleset: RuleSet, lexicon: list[Entry] | None = None, *, extra:
Sequence[Validator] = (), lexicon_limit: int = 12, shared: Mapping[str, Any] | None = None) ->
None`
- `ruleset: RuleSet` — drives the built-in mechanical checks.
- `lexicon: list[Entry] | None = None` — established terms; defaults to `[]`.
- `extra: Sequence[Validator] = ()` — pluggable, task-specific `Validator`-port components run
  after the built-ins.
- `lexicon_limit: int = 12`.
- `shared: Mapping[str, Any] | None = None` — run-wide services (e.g. a `SqlStore`/
  `SchemaIntrospector` for a SQL-safety check) merged into every pluggable validator's context
  mapping; copied into `self.shared`.

No validation is performed in the constructor.

#### `check(self, record: Record, output: str, *, capture: Capture | None = None) -> list[Violation]`

**Args:** `record: Record`, `output: str`, `capture: Capture | None = None` — keyword-only,
forwarded into the pluggable validators' shared context under `"capture"`.

**Returns:** `list[Violation]` — the built-in `check_mechanical(record, output, self.ruleset,
required_terms=..., max_columns=...)` violations (`required_terms` from `relevant_entries(record.
source, self.lexicon, limit=self.lexicon_limit)`; `max_columns` from `record.meta.get
("max_columns")`, only honoured when it is a positive `int`) followed, in order, by every
violation each of `self.extra`'s validators returns from `.validate(record, output, context)`
(`context` = `{**self.shared, "capture": capture, "ruleset": self.ruleset, "required_terms":
required, "max_columns": max_columns}`).

**Raises:** propagates whatever a pluggable `validator.validate(...)` raises — a defect in a
task-specific validator, exactly the kind of exception `Harness.process`'s generic `except
Exception` clause is designed to contain to one record; also propagates `check_mechanical`'s own
possible `re.error`.

**Side effects:** none of its own; a pluggable validator may have its own (e.g. a grounding check
that retrieves and writes into `context["capture"]`).

#### `partition_on_exhaustion(violations: Sequence[Violation], ruleset: RuleSet) -> tuple[list[Violation], list[Violation]]`

Module-level function. Split blocking violations into `(reject, keep_flagged)` by the rule set's
`keep_flagged_rules`. When the repair budget runs out, a keep-flagged blocker means the output is
imperfect but still worth keeping (the alternative is showing nothing); a reject blocker means the
output is unusable.

**Args:** `violations: Sequence[Violation]`, `ruleset: RuleSet`.

**Returns:** `tuple[list[Violation], list[Violation]]` — `(reject, keep)`, computed by filtering
`blocking(violations)` (only `ERROR`-severity violations; `ragkit.core.rules.blocking`) and
sorting each into `keep` if its `rule_id` is in `ruleset.keep_flagged_rules`, else `reject`.

**Raises:** none.
