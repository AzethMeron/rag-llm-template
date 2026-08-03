# ragkit.core API Reference

`ragkit.core` is the dependency-free foundation every other layer of the framework builds on:
records and their durable JSON Lines catalogue/journal, the `Protocol` ports that decouple every
replaceable component (store drivers, retrievers, chunkers, model backends, ...), the one
extension mechanism (`Registry`) those components are resolved through, and a handful of
stdlib-only primitives (strict TOML config loading, JSON type checking, terminology lookup,
placeholder preservation, rule-violation vocabulary, display width). Nothing in this package
imports anything outside the standard library — `tests/test_boundaries.py` enforces this by
importing the package with optional dependencies made unimportable — so the contract it defines
can never break because a machine-learning or storage dependency changed. `store`, `ingest`,
`retrieve`, `llm`, and `harness` all depend on `ragkit.core`; it depends on none of them.

## ragkit.core.config

Strict TOML loading primitives — the single source of truth for how every config file in the
framework (personas, rules, context, retrieval, models, storage) and every third-party
component's own option block is parsed. Centralizing this here means an unknown key is always an
error (never a silently-ignored typo), a `bool` is never coerced into the `int`/`float` it happens
to subclass, and the `[[section]]` vs `[section]` mistake is always named explicitly rather than
surfacing as a bare iteration-yields-keys bug. Range checks are deliberately left to the receiving
dataclass, so a directly constructed object is exactly as hard to make invalid as a loaded one.

### ConfigError

A configuration file is missing, malformed, or internally inconsistent. Subclasses
`ragkit.core.errors.LocatedError` and adds its own `__init__`, so it carries `path` and `label`
(the section at fault) in addition to the inherited `reason`/`context`/`line_no`, pointing an
operator at the exact place to fix rather than at a bare traceback.

**Attributes:**
- `reason` (`str`): the stable, human/machine-readable statement of what went wrong (from `RagkitError`).
- `context` (`Mapping[str, Any]`): diagnostic detail keyed by name, `None` values dropped (from `RagkitError`).
- `path` (`Path | None`, default `None`): the config file at fault, when known (from `LocatedError`).
- `line_no` (`int | None`, default `None`): always `None` for this class — `ConfigError` never passes a line number (from `LocatedError`).
- `label` (`str | None`, default `None`): the name of the section/key at fault, when known.

#### `ConfigError(reason: str, *, path: Path | None = None, label: str | None = None) -> None`

**Args:**
- `reason` (`str`): what went wrong, e.g. `"invalid TOML: ..."`.
- `path` (`Path | None`, default `None`): the file at fault.
- `label` (`str | None`, default `None`): the section/key at fault.

**Returns:** none (constructor).

#### `load_toml(path: Path, *, what: str = "configuration file") -> dict[str, Any]`

Read and parse a TOML file, turning both failure modes into a `ConfigError` rather than letting a
raw `FileNotFoundError` or `TOMLDecodeError` reach the caller.

**Args:**
- `path` (`Path`): the file to read.
- `what` (`str`, default `"configuration file"`): a noun used in the "not found" message (e.g. `"models file"`), so the error names the kind of file that's missing.

**Returns:** `dict[str, Any]` — the parsed TOML document.

**Raises:**
- `ConfigError`: `path` does not point to a regular file, or the file's contents are not valid TOML (`tomllib.TOMLDecodeError`).

#### `as_table(value: object, *, label: str, path: Path | None = None) -> dict[str, Any]`

The value as a TOML table (`dict`), or a `ConfigError` naming what it actually is. Answers "is this
even a table?" once, centrally, rather than letting a bare `TypeError` surface deep inside a
loader when someone writes `limits = 5` instead of `[limits]`.

**Args:**
- `value` (`object`): the value to check.
- `label` (`str`): the section name, used in the error message.
- `path` (`Path | None`, default `None`): the file the value came from, for the error.

**Returns:** `dict[str, Any]` — `value`, unchanged, once confirmed to be a `dict`.

**Raises:**
- `ConfigError`: `value` is not a `dict`.

#### `reject_unknown(section: object, allowed: AbstractSet[str], *, label: str, path: Path | None = None) -> dict[str, Any]`

The section as a table, with any key outside `allowed` refused rather than dropped — a typo that
silently left a default in force is exactly the silent failure the project forbids.

**Args:**
- `section` (`object`): the value to validate (must be a table; checked via `as_table`).
- `allowed` (`AbstractSet[str]`): the set of key names this section may contain.
- `label` (`str`): the section name, used in the error message.
- `path` (`Path | None`, default `None`): the file the section came from.

**Returns:** `dict[str, Any]` — the section, confirmed to contain only allowed keys.

**Raises:**
- `ConfigError`: `section` is not a table (via `as_table`), or it contains a key not in `allowed` (names the offending keys and the allowed set).

#### `tables(data: dict[str, Any], key: str, *, path: Path | None = None) -> list[dict[str, Any]]`

The entries of an array-of-tables (`[[key]]`), each validated to be a table. `[key]` written where
`[[key]]` was meant is the likeliest structural mistake in a config file; it is named explicitly
here rather than left to produce its old failure (iterating a table yields its keys, so each
"entry" silently became a bare string).

**Args:**
- `data` (`dict[str, Any]`): the parsed document (or a sub-table) to read `key` from.
- `key` (`str`): the array-of-tables key, e.g. `"endpoint"`.
- `path` (`Path | None`, default `None`): the file, for the error.

**Returns:** `list[dict[str, Any]]` — `data.get(key, [])`, each element confirmed to be a table.

**Raises:**
- `ConfigError`: `data[key]` exists and is not a `list`, or any entry in it is not a `dict`.

#### `read_int(section: dict[str, Any], key: str, default: int, *, label: str, path: Path | None = None) -> int`

A TOML integer, refused rather than coerced when mistyped. `bool` is explicitly excluded despite
being an `int` subclass in Python, so `= true` reading as `1` cannot silently turn a token budget
into a plausible-but-wrong value. Range checking is left to the dataclass that receives the value.

**Args:**
- `section` (`dict[str, Any]`): the table to read from.
- `key` (`str`): the key to read.
- `default` (`int`): value used when `key` is absent.
- `label` (`str`): the section name, for the error message (rendered as `label.key`).
- `path` (`Path | None`, default `None`): the file, for the error.

**Returns:** `int` — the value at `key`, or `default`.

**Raises:**
- `ConfigError`: the value present is not an `int`, or is a `bool`.

#### `read_float(section: dict[str, Any], key: str, default: float, *, label: str, path: Path | None = None) -> float`

A TOML number, refused rather than coerced when mistyped. An `int` is accepted and widened to
`float` (0 and 1 are natural values for a fraction); `bool` is not, since `= true` reading as `1.0`
could silently turn a similarity floor into "accept only exact matches."

**Args:**
- `section` (`dict[str, Any]`): the table to read from.
- `key` (`str`): the key to read.
- `default` (`float`): value used when `key` is absent.
- `label` (`str`): the section name, for the error message.
- `path` (`Path | None`, default `None`): the file, for the error.

**Returns:** `float` — the value at `key` (widened from `int` if needed), or `default`.

**Raises:**
- `ConfigError`: the value present is a `bool`, or is neither `int` nor `float`.

#### `read_bool(section: dict[str, Any], key: str, default: bool, *, label: str, path: Path | None = None) -> bool`

A TOML boolean, refused rather than coerced — `bool("false")` is `True` in Python, so coercion here
could silently invert a safety gate.

**Args:**
- `section` (`dict[str, Any]`): the table to read from.
- `key` (`str`): the key to read.
- `default` (`bool`): value used when `key` is absent.
- `label` (`str`): the section name, for the error message.
- `path` (`Path | None`, default `None`): the file, for the error.

**Returns:** `bool` — the value at `key`, or `default`.

**Raises:**
- `ConfigError`: the value present is not a `bool`.

#### `read_string(section: dict[str, Any], key: str, default: str, *, label: str, path: Path | None = None) -> str`

A TOML string, refused rather than coerced.

**Args:**
- `section` (`dict[str, Any]`): the table to read from.
- `key` (`str`): the key to read.
- `default` (`str`): value used when `key` is absent.
- `label` (`str`): the section name, for the error message.
- `path` (`Path | None`, default `None`): the file, for the error.

**Returns:** `str` — the value at `key`, or `default`.

**Raises:**
- `ConfigError`: the value present is not a `str`.

#### `read_string_list(section: dict[str, Any], key: str, *, label: str, path: Path | None = None) -> tuple[str, ...]`

A TOML array of strings. A bare string is refused rather than accepted, because `tuple("abc")`
would silently split it into one-character entries instead of failing.

**Args:**
- `section` (`dict[str, Any]`): the table to read from.
- `key` (`str`): the key to read.
- `label` (`str`): the section name, for the error message.
- `path` (`Path | None`, default `None`): the file, for the error.

**Returns:** `tuple[str, ...]` — `section.get(key, [])` as a tuple, or `()` when absent.

**Raises:**
- `ConfigError`: the value present is not a `list`, or contains a non-`str` element.

#### `read_required_path(section: dict[str, Any], key: str, *, label: str, path: Path | None = None) -> str`

A required filesystem path, refused rather than defaulted when absent or blank — there is
deliberately no `default` parameter. Unlike `read_string`, a forgotten `path` for a store whose
whole purpose is durable persistence must not silently become an ephemeral `:memory:` database
that loses everything between runs, with no error anywhere. An explicit `":memory:"` is a valid,
deliberate value (tests, a scratch store) and is accepted.

**Args:**
- `section` (`dict[str, Any]`): the table to read from.
- `key` (`str`): the key to read (the database file path).
- `label` (`str`): the section name, used in the error message.
- `path` (`Path | None`, default `None`): the file the section came from, for the error.

**Returns:** `str` — the value at `key`, once confirmed to be a non-blank string; an explicit `":memory:"` is returned unchanged.

**Raises:**
- `ConfigError`: `key` is absent, or its value is not a `str`, or is blank/whitespace-only (the message names the key and explains the in-memory-fallback hazard).

## ragkit.core.errors

The one base every deliberate error in the framework derives from. Errors here are structured and
diagnostic, not bare strings: each carries a machine-readable `reason` and a `context` mapping, so
a failure can be caught, inspected, and debugged without reproducing it. The subclasses that name
each subsystem's specific failure (`ConfigError`, `RegistryError`, `CatalogError`, ...) live beside
their own code, not here, so this module depends on nothing and every layer can raise a framework
error without importing half the framework.

### RagkitError

Base for every error the framework raises deliberately — never raised directly. A caller wanting
to handle "any error this framework raised on purpose" catches this; one wanting a specific
failure catches the subsystem's subclass.

**Attributes:**
- `reason` (`str`): the stable, human-and-machine-readable statement of *what* went wrong.
- `context` (`Mapping[str, Any]`): keyword diagnostic detail (*where*, expected vs. actual, offending input); entries whose value is `None` are dropped so an absent optional does not print as `key=None`.

#### `RagkitError(reason: str, **context: Any) -> None`

**Args:**
- `reason` (`str`): the stable statement of what went wrong.
- `**context` (`Any`): arbitrary named diagnostic values; any value equal to `None` is dropped before storage.

**Returns:** none (constructor). Also builds the exception's rendered message (`reason` plus `key=value, ...` for non-empty `context`) and passes it to `Exception.__init__`.

### LocatedError

An error attributable to a place in a file: the path, and the line within it when known. One home
for that shape — `path` and `line_no` used to be represented inconsistently across subsystems
(`ConfigError` put `path` directly in `context`; others folded path+line into one pre-formatted
string), so a handler reading `exc.context["path"]` worked for one and raised `KeyError` for
others. They are now separate, structured attributes everywhere.

**Attributes:**
- `path` (`Any`, default `None`): the file the error is attributable to (typically a `pathlib.Path`).
- `line_no` (`int | None`, default `None`): the line within `path`, when known.
- (inherits `reason` and `context` from `RagkitError`.)

#### `LocatedError(reason: str, *, path: Any = None, line_no: int | None = None, **extra: Any) -> None`

**Args:**
- `reason` (`str`): what went wrong.
- `path` (`Any`, default `None`): the file at fault.
- `line_no` (`int | None`, default `None`): the line within `path`, when known.
- `**extra` (`Any`): a subclass's own additional context keys (e.g. `ConfigError`'s `label`), merged into `context` without re-implementing the location half.

**Returns:** none (constructor).

## ragkit.core.jsonshape

JSON value-type checking, shared by the LLM client's schema-shape validation
(`ragkit.llm.client`) and any validator that must check a declared field type against a value. One
home for the rule that `bool` is not an `integer` or `number` even though Python makes `bool` a
subclass of `int` — kept here so a validator and the client's shape check cannot drift apart.
Stdlib-only.

### JsonShapeError

A declared JSON type is not one this module knows how to check. Subclasses `RagkitError` directly
(no custom `__init__`); carries only the inherited `reason`/`context`.

#### `json_type_matches(value: Any, declared: Any) -> bool`

Whether `value` has the JSON type `declared` names — a single type name (`"string"`, `"integer"`,
`"number"`, `"boolean"`, `"array"`, `"object"`, `"null"`), or a list of them treated as a union. An
unrecognised name raises rather than silently passing — returning `True` for a typo like
`"boolena"` used to mean the check was effectively switched off for that field.

**Args:**
- `value` (`Any`): the value to check.
- `declared` (`Any`): a JSON type name (`str`), or a `list` of type names (union).

**Returns:** `bool` — whether `value` matches `declared`.

**Raises:**
- `JsonShapeError`: `declared` (or, for a union, any member of it) is not one of the known type names.

## ragkit.core.lexicon

Domain terminology as a term-to-rendering mapping retrieved by literal occurrence — the
generalisation of a translation glossary: a body of established terms and the exact form each
should take in the output (a house translation, a canonical spelling, a required SQL identifier, a
fixed form-field label). Retrieval (`relevant_entries`) is deliberately simple and deterministic:
the entries whose term literally occurs in the input, longest first, capped. A component decides
what to do with the matches — usually inject them into the prompt, and, for ones that must appear
verbatim, enforce their presence with a mechanical validator.

### LexiconError

A lexicon source could not be read or parsed. Subclasses `LocatedError` directly (no custom
`__init__`); carries the inherited `reason`, `context`, `path`, `line_no`.

### Entry

One term and its established rendering. A frozen, `slots`-based dataclass.

**Attributes:**
- `term` (`str`): the term to look for, matched by literal substring occurrence.
- `rendering` (`str`): the established rendering to use in place of `term`.
- `category` (`str`, default `""`): free-form and task-defined; the core never reads it — a recipe uses it to route entries to its own validators or prompt sections.
- `entity_id` (`int`, default `0`): a caller-side handle back to whatever the entry came from; the framework never interprets it.

#### `to_json(self) -> str`

**Args:** none.

**Returns:** `str` — the entry as a single-line JSON object (via `dataclasses.asdict` + `json.dumps(ensure_ascii=False)`).

**Raises:** none explicitly (propagates whatever `json.dumps` would raise for non-serialisable content, which cannot occur here since all fields are `str`/`int`).

#### `write_lexicon(entries: list[Entry], path: Path) -> int`

Write entries as JSON Lines, longest term first. Longest-first is a convenience for a human reading
the file; `relevant_entries` re-sorts on read, so callers do not depend on this order.

**Args:**
- `entries` (`list[Entry]`): the entries to write.
- `path` (`Path`): the destination file.

**Returns:** `int` — the number of entries written (`len(entries)`).

**Raises:** none explicitly (propagates any `OSError` from creating the parent directory or writing the file).

**Side effects:** creates `path.parent` if missing; overwrites `path`.

#### `read_lexicon(path: Path) -> list[Entry]`

Load established terminology, or `[]` if the file does not exist — a missing lexicon is a
legitimate, common state (terminology accumulates over a project's life, it is not something every
task starts with). A lexicon that *exists* and is malformed is still an error: silently treating a
corrupt file as empty would drop terminology the caller believes is in force.

**Args:**
- `path` (`Path`): the lexicon file (JSON Lines).

**Returns:** `list[Entry]` — every entry in the file, in file order; `[]` if `path` does not exist.

**Raises:**
- `LexiconError`: `path` exists but is not a regular file; a line is not valid JSON; a line's JSON object is missing `term` or `rendering`; or a line's fields do not match `Entry`'s constructor (`TypeError`).

#### `relevant_entries(source_text: str, entries: list[Entry], limit: int = 24) -> list[Entry]`

Lexicon entries whose term literally occurs in `source_text`, longest first, capped at `limit`.
Longest-first so a compound term wins over its constituents. Substring matching is a deliberate
over-match (an inflected language's lemma will not equal its declined form, so a whole-word
requirement would miss most real occurrences) — the asymmetry between over- and under-matching is
what settles it: showing the model an unneeded term is safer than silently dropping one it needed.

**Args:**
- `source_text` (`str`): the text to search for term occurrences.
- `entries` (`list[Entry]`): the candidate entries.
- `limit` (`int`, default `24`): maximum number of entries returned.

**Returns:** `list[Entry]` — matching entries, longest term first, truncated to `limit`.

**Raises:** none.

## ragkit.core.placeholders

The placeholder convention shared across the framework: a source of records may replace parts of a
payload the model must not touch (engine expressions, conditionals, escapes, inline markup, a
column reference) with `[[0]]`, `[[1]]`, and so on, then restore them afterwards. The producer only
has to preserve the *set* of placeholders — it never learns what any one contained, which is what
lets one engine serve inputs whose syntaxes have nothing in common. The mechanical `placeholders`
validator (in the harness layer) enforces that the set is neither changed nor renumbered.

#### `PLACEHOLDER`

A compiled regular expression, `re.compile(r"\[\[(\d+)\]\]")`, matching one placeholder token and
capturing its numeric index. The single pattern every reader and writer of the placeholder
convention must share, so a producer and its validator can never disagree on the syntax.

#### `placeholder_indices(text: str) -> list[int]`

Every placeholder index referenced by `text`, in order of appearance.

**Args:**
- `text` (`str`): text that may contain `[[N]]` placeholders.

**Returns:** `list[int]` — the integer inside each `[[N]]` match, in the order the matches occur (duplicates preserved, e.g. `[[0]] ... [[0]]` yields `[0, 0]`).

**Raises:** none.

## ragkit.core.ports

The seams: the `Protocol` every replaceable component implements, and the value types they
exchange. A layer depends on a port here, never on a concrete driver; the concrete driver is
resolved through a matching `ragkit.core.registry.Registry` and checked against the port with
`isinstance`. Every port is `runtime_checkable` so that check works whether the port is defined by
methods, attributes, or both. Signatures are kept deliberately small and hard to misuse.

A shared score convention runs through every retrieval port: **every score a port returns is
"higher is better," normalised toward `[0, 1]`.** A backend whose native scale is inverted (e.g.
SQLite's `bm25()`, negative and lower-is-better) or unbounded (a cross-encoder logit) must convert
*inside* the driver, so a sign convention can never leak into the fuser and silently invert a
ranking.

### Message

One chat turn — the unit a model client and a backend operate on. Frozen, `slots`-based dataclass.

**Attributes:**
- `role` (`str`): the chat role (`"system"`, `"user"`, `"assistant"`, ...).
- `content` (`str`): the turn's text.

### SamplingParams

Per-request decode settings for one chat completion — the knobs an OpenAI-compatible server
accepts *per call* (temperature, nucleus/top-k/min-p truncation, penalties, seed, stop). Owned by
the persona issuing the request, because the right setting is a property of the role (a producer
may want warmth, a reviewer wants determinism), not the model. Settings a server can only apply at
*launch* (context size, GPU offload, KV-cache type) live in `models.toml` instead. `max_tokens` is
deliberately not a field here — it is a separately computed output budget. Frozen, `slots`-based
dataclass; validates its own ranges in `__post_init__` so an invalid instance cannot be
constructed.

**Attributes:**
- `temperature` (`float`, default `0.2`): decode temperature; must be `>= 0`.
- `top_p` (`float | None`, default `None`): nucleus sampling threshold; when set, must be in `[0.0, 1.0]`.
- `top_k` (`int | None`, default `None`): top-k truncation; when set, must be `>= 0` (`0` disables it).
- `min_p` (`float | None`, default `None`): min-p truncation; when set, must be in `[0.0, 1.0]`.
- `seed` (`int | None`, default `None`): decode seed, for reproducibility when the server honours it.
- `presence_penalty` (`float | None`, default `None`): when set, must be in `[-2.0, 2.0]`.
- `frequency_penalty` (`float | None`, default `None`): when set, must be in `[-2.0, 2.0]`.
- `repeat_penalty` (`float | None`, default `None`): when set, must be `> 0` (`1.0` is no penalty).
- `stop` (`tuple[str, ...]`, default `()`): stop sequences.

Only fields that are set are sent to the server (see `payload`), so the default request carries
just a temperature and a strict-OpenAI endpoint is never handed a llama.cpp-only knob it would
reject.

#### `payload(self) -> dict[str, Any]`

**Args:** none.

**Returns:** `dict[str, Any]` — the request-body fragment: `"temperature"` always, plus every other
field that is not `None` (and `"stop"` as a `list` when non-empty). An unset knob is omitted rather
than sent as a default.

**Raises:** none.

### StructuredRequest

How to ask a specific model for schema-conforming JSON: the (possibly rewritten) messages and the
`response_format` fragment to put in the request body. A backend that must describe the schema in
the prompt returns rewritten messages; one that constrains decoding returns the messages
unchanged. Frozen, `slots`-based dataclass.

**Attributes:**
- `messages` (`tuple[Message, ...]`): the messages to send (possibly with an appended instruction).
- `response_format` (`dict[str, Any]`): the `response_format` body fragment.

### Document

A source document after extraction, before chunking. Frozen, `slots`-based dataclass.

**Attributes:**
- `doc_id` (`str`): the document's identifier.
- `text` (`str`): the extracted text.
- `meta` (`Mapping[str, Any]`, default `{}`): extraction provenance, free-form.

### Chunk

One indexable unit of a document: the text, an id, and its provenance in `meta` (`document_id`,
`version_id`, `ordinal`, `section_path`, offsets, ...). Frozen, `slots`-based dataclass.

**Attributes:**
- `chunk_id` (`str`): the chunk's identifier.
- `text` (`str`): the chunk's text.
- `meta` (`Mapping[str, Any]`, default `{}`): provenance, free-form.

### Retrieved

A chunk a query matched, with its relevance score (higher is better, toward `[0, 1]`). Frozen,
`slots`-based dataclass.

**Attributes:**
- `chunk_id` (`str`): the matched chunk's identifier.
- `text` (`str`): the chunk's text.
- `score` (`float`): relevance, higher is better, normalised toward `[0, 1]`.
- `meta` (`Mapping[str, Any]`, default `{}`): the chunk's provenance.

### FilterOp

Comparisons a metadata filter predicate can express. A `str`-valued `Enum` — the member's value
*is* the member, so it serialises as itself.

**Members:**
- `EQ = "eq"`, `NE = "ne"`, `LT = "lt"`, `LE = "le"`, `GT = "gt"`, `GE = "ge"`, `IN = "in"`.

### Predicate

One metadata comparison: `field <op> value`. Frozen, `slots`-based dataclass. A `Filter` (the type
alias `tuple[Predicate, ...]`) is a conjunction (AND) of predicates — the framework's own small,
backend-neutral filter language; each store driver compiles it to its own dialect, and a driver
that cannot honour a predicate refuses it at that boundary rather than at query time. Deliberately
not raw backend filter dicts, which would couple every caller to one driver.

**Attributes:**
- `field` (`str`): the metadata key being compared.
- `op` (`FilterOp`): the comparison.
- `value` (`Any`): the value to compare against (a `list`/sequence for `FilterOp.IN`).

### Source (Protocol)

The upstream boundary toward the input: yields the records a run will work on. `runtime_checkable`.
No concrete implementation of this port exists as a shared class in `ragkit.core`/`ragkit.llm`; a
recipe typically produces `Record`s from its own fetch script rather than through a named driver.

#### `records(self) -> Iterator[Record]`

**Args:** none.

**Returns:** `Iterator[Record]` — the records to process.

**Raises:** implementation-defined (the port itself declares none).

### Sink (Protocol)

The downstream boundary toward the output: writes produced records back to their home.
`runtime_checkable`. Implemented by `ragkit.store.pairings.sink.PairingSink`.

#### `write(self, records: Iterable[Record]) -> None`

**Args:**
- `records` (`Iterable[Record]`): the records to write back.

**Returns:** `None`.

**Raises:** implementation-defined.

### Extractor (Protocol)

Turns a raw source (a file's bytes, a row set) into documents ready to chunk. `runtime_checkable`.
Implemented by `ragkit.ingest.extract.TextExtractor`, `JsonlExtractor`, `HtmlExtractor`, and
`MarkdownExtractor`.

#### `extract(self, source: bytes | str, *, meta: Mapping[str, Any] = {}) -> Iterator[Document]`

**Args:**
- `source` (`bytes | str`): the raw input to extract from.
- `meta` (`Mapping[str, Any]`, default `{}`): extraction-time metadata merged into each yielded `Document`.

**Returns:** `Iterator[Document]` — the extracted documents.

**Raises:** implementation-defined.

### Chunker (Protocol)

Splits a document into indexable chunks, preserving provenance in each chunk's meta.
`runtime_checkable`. Implemented by `ragkit.ingest.chunk.FixedChunker`, `SentenceChunker`, and
`StructureChunker`.

#### `chunk(self, document: Document) -> Iterator[Chunk]`

**Args:**
- `document` (`Document`): the document to split.

**Returns:** `Iterator[Chunk]` — the resulting chunks.

**Raises:** implementation-defined.

### Embedder (Protocol)

Encodes texts into dense vectors; rows are returned in input order. `runtime_checkable`.
Implemented by `ragkit.retrieve.embedding.EmbeddingClient`.

#### `embed(self, texts: Sequence[str]) -> list[Sequence[float]]`

**Args:**
- `texts` (`Sequence[str]`): the texts to encode.

**Returns:** `list[Sequence[float]]` — one vector per input text, same order.

**Raises:** implementation-defined.

### VectorIndex (Protocol)

A dense-vector store. `search` returns `(chunk_id, score)` best-first, score higher-is-better
toward `[0, 1]`. `runtime_checkable`. Implemented by `ragkit.store.vector.lancedb.LanceVectorIndex`
and `ragkit.store.vector.qdrant.QdrantVectorIndex`.

#### `upsert(self, ids: Sequence[str], vectors: Sequence[Sequence[float]], metas: Sequence[Mapping[str, Any]]) -> None`

**Args:**
- `ids` (`Sequence[str]`): chunk ids, one per row.
- `vectors` (`Sequence[Sequence[float]]`): the dense vectors, one per id.
- `metas` (`Sequence[Mapping[str, Any]]`): per-row metadata, one per id.

**Returns:** `None`.

**Raises:** implementation-defined.

#### `search(self, vector: Sequence[float], *, k: int, where: Filter = ()) -> list[tuple[str, float]]`

**Args:**
- `vector` (`Sequence[float]`): the query vector.
- `k` (`int`): number of results to return.
- `where` (`Filter`, default `()`): a conjunction of `Predicate`s to restrict the search.

**Returns:** `list[tuple[str, float]]` — `(chunk_id, score)`, best-first.

**Raises:** implementation-defined (a driver that cannot honour a `where` predicate is expected to refuse at this boundary).

#### `delete(self, ids: Sequence[str]) -> None`

**Args:**
- `ids` (`Sequence[str]`): chunk ids to remove.

**Returns:** `None`.

**Raises:** implementation-defined.

#### `count(self) -> int`

**Args:** none.

**Returns:** `int` — the number of indexed vectors.

**Raises:** implementation-defined.

#### `reconcile(self, chunk_ids: Iterable[str]) -> set[str]`

Reconcile the index against the authoritative set of chunk ids: drops orphan vectors (indexed but
no longer in `chunk_ids`) and returns the ids that are missing a vector (in `chunk_ids` but not
indexed) for the caller to re-embed or refuse. The index cannot re-embed itself (it has neither
the text nor the embedder), so it reports the gap rather than hiding it. A driver that co-locates
vectors with the rows (so drift is impossible) may return an empty set without doing anything.

**Args:**
- `chunk_ids` (`Iterable[str]`): the authoritative set of chunk ids that should have a vector.

**Returns:** `set[str]` — ids in `chunk_ids` with no indexed vector.

**Raises:** implementation-defined.

**Side effects:** deletes orphaned vectors from the index.

### SearchIndex (Protocol)

The read side of a keyword search index: maps a query to `(chunk_id, score)` best-first,
higher-is-better, `score` in `[0, 1)`. `runtime_checkable`. This is the *narrow* interface a
`Retriever` built over a search index actually depends on (never the write side) — a
`PairingStore` satisfies it. `ragkit.store.lexical.bm25.bm25_to_relevance` is the shared map from a
raw, higher-is-better magnitude into that range.

#### `search(self, query: str, *, k: int) -> list[tuple[str, float]]`

**Args:**
- `query` (`str`): the search query.
- `k` (`int`): number of results to return.

**Returns:** `list[tuple[str, float]]` — `(chunk_id, score)`, best-first.

**Raises:** implementation-defined.

### Pairing

One reference example held in a `PairingStore`: an input, the target it pairs with (empty for a
lexical-only reference entry), and the context it was produced with — the `(source, context,
target)` triple the reference memory stores and a write-back step produces. Frozen, `slots`-based
dataclass.

**Attributes:**
- `chunk_id` (`str`): the pairing's identifier.
- `source` (`str`): the input side of the pairing.
- `target` (`str`, default `""`): the paired output; empty for a lexical-only reference entry.
- `context` (`str`, default `""`): the context the pairing was produced with.
- `meta` (`Mapping[str, Any]`, default `{}`): free-form metadata.
- `verified` (`bool`, default `False`): whether this was machine-produced and accepted (write-back provenance).
- `created_at` (`float`, default `0.0`): when it was produced (write-back provenance); an imported entry leaves this at the default.

### PairingStore (Protocol)

The reference-memory store: rows and a keyword search index co-located in one durable store, so a
hit and its display text can never drift apart the way two separately-written stores can.
`runtime_checkable`. Implemented by `ragkit.store.pairings.sqlite.SqlitePairings` and
`ragkit.store.pairings.duckdb.DuckDBPairings`. `search` satisfies `SearchIndex`.

#### `add(self, pairings: Iterable[Pairing]) -> int`

Add pairings in one transaction (their search entries included).

**Args:**
- `pairings` (`Iterable[Pairing]`): the pairings to add.

**Returns:** `int` — the number of rows actually added; a pairing whose `chunk_id` already exists is left untouched (not overwritten), so re-adding the same write-back result twice is idempotent.

**Raises:** implementation-defined.

**Side effects:** persists rows and their search-index entries in one transaction.

#### `search(self, query: str, *, k: int) -> list[tuple[str, float]]`

**Args:**
- `query` (`str`): the search query.
- `k` (`int`): number of results.

**Returns:** `list[tuple[str, float]]` — `(chunk_id, score)`, best-first.

**Raises:** implementation-defined.

#### `document(self, chunk_id: str) -> tuple[str, Mapping[str, Any]] | None`

**Args:**
- `chunk_id` (`str`): the pairing to resolve.

**Returns:** `tuple[str, Mapping[str, Any]] | None` — the display text and metadata for a hit, or `None` if `chunk_id` is absent.

**Raises:** implementation-defined.

#### `get(self, chunk_id: str) -> Pairing | None`

The full pairing for `chunk_id` (source, target, context, meta, verification, provenance), or
`None` if absent. Unlike `document`, which flattens a pairing to display text for a retriever,
this is the write-back / inspection path that needs the whole row.

**Args:**
- `chunk_id` (`str`): the pairing to fetch.

**Returns:** `Pairing | None`.

**Raises:** implementation-defined.

#### `all_ids(self) -> Iterator[str]`

Every chunk id currently stored, for a caller reconciling a separate vector index against this
store's authoritative rows (see `VectorIndex.reconcile`).

**Args:** none.

**Returns:** `Iterator[str]`.

**Raises:** implementation-defined.

#### `count(self) -> int`

**Args:** none.

**Returns:** `int` — the number of stored pairings.

**Raises:** implementation-defined.

### LexiconStore (Protocol)

DB-native established terminology, replacing a JSONL lexicon file as the accumulating source of
truth for a project's terminology. `entries()` returns exactly what
`ragkit.core.lexicon.relevant_entries` and the harness's `LexiconBlock` already consume from a
JSONL-sourced `list[Entry]`, so either source works with the same downstream code.
`runtime_checkable`. Implemented by `ragkit.store.lexicon.sqlite.SqliteLexicon`.

#### `entries(self) -> list[Entry]`

**Args:** none.

**Returns:** `list[Entry]` — every stored entry.

**Raises:** implementation-defined.

#### `add(self, entries: Iterable[Entry]) -> int`

Add or update entries, keyed on `(term, category)`.

**Args:**
- `entries` (`Iterable[Entry]`): entries to add or update.

**Returns:** `int` — the number of rows actually inserted; an update to an existing term's rendering does not count as added.

**Raises:** implementation-defined.

**Side effects:** persists new/updated entries.

### SqlStore (Protocol)

A relational store. `runtime_checkable`. Implemented by `ragkit.store.sql.sqlite.SqliteStore` and
`ragkit.store.sql.duckdb.DuckDBStore`.

**Attributes:**
- `read_only` (`bool`): marks a binding the framework must not write through; a write attempt on one is refused at the port, before the database.

#### `query(self, sql: str, params: Sequence[Any] = ()) -> list[Mapping[str, Any]]`

Runs a read.

**Args:**
- `sql` (`str`): the SQL statement.
- `params` (`Sequence[Any]`, default `()`): bound parameters.

**Returns:** `list[Mapping[str, Any]]` — one mapping per result row.

**Raises:** implementation-defined.

#### `execute(self, sql: str, params: Sequence[Any] = ()) -> None`

Runs a write; expected to raise when `read_only` is `True`.

**Args:**
- `sql` (`str`): the SQL statement.
- `params` (`Sequence[Any]`, default `()`): bound parameters.

**Returns:** `None`.

**Raises:** implementation-defined — a conforming driver raises on a `read_only` binding.

### SchemaIntrospector (Protocol)

Reads a database's schema for the NL->SQL feature, without importing a store driver.
`runtime_checkable`. Implemented by `ragkit.store.sql.sqlite.SqliteIntrospector` and
`ragkit.store.sql.duckdb.DuckDBIntrospector`.

#### `schema(self) -> Mapping[str, Sequence[tuple[str, str]]]`

**Args:** none.

**Returns:** `Mapping[str, Sequence[tuple[str, str]]]` — table name to its ordered column
`(name, type)` pairs.

**Raises:** implementation-defined.

### RetrievedRef

A JSON-safe projection of a `Retrieved` hit, for persistence in a `RunResult`: `(chunk_id, text,
score)` only, no `meta` — meta need not be JSON-safe, and is reference data the pairing store
already holds, keyed by `chunk_id`, so there is nothing to duplicate into the run store. Frozen,
`slots`-based dataclass.

**Attributes:**
- `chunk_id` (`str`): the matched chunk's identifier.
- `text` (`str`): the chunk's text.
- `score` (`float`): relevance score.

### RunResult

One completed attempt at a record, as a `RunStore` persists it. `record` already carries its
verdict (the status/output/notes an `Outcome` applies to it); the fields here are what the legacy
JSONL journal could not hold: the structured violations (never flattened to strings), and the
context that actually produced the output. `reviews` is a tuple of plain, already-JSON-safe
mappings rather than the harness's own `Review` type, so this port need not import the harness
layer. Frozen, `slots`-based dataclass.

**Attributes:**
- `record` (`Record`): the finished record (output, status, notes, meta).
- `context_passage` (`str`, default `""`): the context actually used to produce the output.
- `retrieved` (`tuple[RetrievedRef, ...]`, default `()`): the retrieval hits behind the output.
- `reviews` (`tuple[Mapping[str, Any], ...]`, default `()`): plain, JSON-safe review records.
- `violations` (`tuple[Violation, ...]`, default `()`): structured rule violations found.
- `rounds` (`int`, default `0`): number of production/review rounds taken.
- `error` (`str | None`, default `None`): a fatal error message, when the attempt did not complete normally.

### RunStore (Protocol)

The framework's own run-state store: the record catalogue and the append-only result history,
replacing the JSONL catalogue+journal pair. A result write is one transaction, so a torn/partial
row is impossible. `runtime_checkable`. Implemented by
`ragkit.store.run.sqlite.SqliteRunStore`.

#### `add_records(self, records: Iterable[Record]) -> int`

**Args:**
- `records` (`Iterable[Record]`): records to add to the catalogue.

**Returns:** `int` — the number actually added; idempotent on an already-present `record_id`, so re-running an import is safe.

**Raises:** implementation-defined.

**Side effects:** persists new catalogue rows.

#### `append_result(self, result: RunResult) -> None`

**Args:**
- `result` (`RunResult`): one completed attempt.

**Returns:** `None`. Persisted in a single transaction; never overwrites an earlier attempt at the same record — each call adds a new result, and `results`/`completed_ids` resolve to the latest by write order.

**Raises:** implementation-defined.

**Side effects:** appends a new result row.

#### `completed_ids(self) -> set[str]`

**Args:** none.

**Returns:** `set[str]` — ids of every record with at least one result; what a resumed run must skip.

**Raises:** implementation-defined.

#### `pending(self) -> Iterator[Record]`

**Args:** none.

**Returns:** `Iterator[Record]` — records with no result yet, in the store's stable catalogue order (by provenance: `rel_path` then `line_no`); a streamed cursor, never materialising the whole catalogue in memory.

**Raises:** implementation-defined.

#### `results(self) -> Iterator[RunResult]`

**Args:** none.

**Returns:** `Iterator[RunResult]` — the latest result for every record that has one.

**Raises:** implementation-defined.

#### `latest_records(self) -> Iterator[Record]`

Like `results`, but yields the finished `Record` alone rather than the full `RunResult` — for a
caller that only needs what a record produced, not the retrieval/review/violation detail behind
how it got there.

**Args:** none.

**Returns:** `Iterator[Record]`.

**Raises:** implementation-defined.

#### `count_records(self) -> int`

**Args:** none.

**Returns:** `int` — the number of records in the catalogue.

**Raises:** implementation-defined.

### Retriever (Protocol)

Retrieves the chunks most relevant to a query, best-first, each with score `>= min_score`.
Required to be deterministic and safe to call concurrently — one retriever is shared by every
worker in a run. `runtime_checkable`. Implemented by `ragkit.retrieve.retrievers.LexicalRetriever`,
`DenseRetriever`, and `ragkit.retrieve.hybrid.HybridRetriever`.

#### `retrieve(self, query: str, *, k: int, min_score: float = 0.0) -> tuple[Retrieved, ...]`

**Args:**
- `query` (`str`): the query text.
- `k` (`int`): number of results to return.
- `min_score` (`float`, default `0.0`): minimum score a result must meet.

**Returns:** `tuple[Retrieved, ...]` — best-first, each with `score >= min_score`.

**Raises:** implementation-defined.

### Reranker (Protocol)

Re-scores a candidate shortlist against a query. `runtime_checkable`. Implemented by
`ragkit.retrieve.rerank.RerankClient`.

#### `rerank(self, query: str, documents: Sequence[str]) -> list[tuple[int, float]]`

**Args:**
- `query` (`str`): the query text.
- `documents` (`Sequence[str]`): candidate documents to score.

**Returns:** `list[tuple[int, float]]` — `(index, score)` into `documents`, best-first; `index` is a permutation of `range(len(documents))`.

**Raises:** implementation-defined.

### ContextBlock (Protocol)

One prompt section. Returns the rendered text, or `None` when it has no content — the no-empty-
section rule is enforced here, so an absent block contributes nothing rather than a dangling
heading. `runtime_checkable`. Implemented by `ragkit.harness.context.blocks.LiteralBlock`,
`LexiconBlock`, `NeighboursBlock`, `EstablishedBlock`, `RetrievedBlock`, `PreviousAttemptBlock`,
`SqlRowsBlock`, `SchemaBlock`, and `ReadingsBlock`.

#### `render(self, record: Record, context: Mapping[str, Any]) -> str | None`

**Args:**
- `record` (`Record`): the record being processed.
- `context` (`Mapping[str, Any]`): shared run context available to every block.

**Returns:** `str | None` — the rendered section text, or `None` for no content.

**Raises:** implementation-defined.

### Validator (Protocol)

A mechanical (code-decidable) check over a produced output. Returns the violations it finds, empty
when the output passes. Must abstain (return no *blocking* violation) rather than guess when the
input gives it no positive evidence to judge on. `runtime_checkable`. Pluggable validators
conforming to this port are task-specific components supplied to
`ragkit.harness.validators.ValidatorPipeline` (which orchestrates them but does not itself
implement this port — its own method is `check`, not `validate`).

#### `validate(self, record: Record, output: str, context: Mapping[str, Any]) -> list[Violation]`

**Args:**
- `record` (`Record`): the record being checked.
- `output` (`str`): the produced output.
- `context` (`Mapping[str, Any]`): shared validation context (ruleset, required terms, column budget, ...).

**Returns:** `list[Violation]` — violations found, `[]` if none.

**Raises:** implementation-defined.

### OutputSchema (Protocol)

Describes the structured output a task produces: its JSON schema (what the model is constrained or
asked to return) and how to pull the output string out of a parsed reply. `runtime_checkable`.
Implemented by `ragkit.harness.schemas.JsonFieldSchema` and `FormSchema`.

**Attributes:**
- `name` (`str`): the schema's name (used as the `response_format`'s `json_schema.name` in a strict-grammar request).

#### `json_schema(self) -> dict[str, Any]`

**Args:** none.

**Returns:** `dict[str, Any]` — the JSON Schema describing the expected output shape.

**Raises:** implementation-defined.

#### `extract(self, reply: Mapping[str, Any]) -> str`

**Args:**
- `reply` (`Mapping[str, Any]`): the parsed JSON reply.

**Returns:** `str` — the output string pulled out of `reply`.

**Raises:** implementation-defined.

### Backend (Protocol)

Request-shaping for one model family: turns messages + a JSON schema into the `(messages,
response_format)` a specific server will honour (strict grammar, or a prompt-described shape).
`runtime_checkable`. Implemented by `ragkit.llm.backends.SchemaBackend` and `JsonObjectBackend`
(see `ragkit.llm`).

**Attributes:**
- `name` (`str`): the backend's registered name.

#### `structured_request(self, messages: Sequence[Message], schema: Mapping[str, Any]) -> StructuredRequest`

**Args:**
- `messages` (`Sequence[Message]`): the conversation so far.
- `schema` (`Mapping[str, Any]`): the JSON Schema the reply must conform to.

**Returns:** `StructuredRequest` — the (possibly rewritten) messages and `response_format` fragment.

**Raises:** implementation-defined.

### Provider (Protocol)

A transport to one endpoint kind (llama.cpp router, ollama, any OpenAI-compatible server). Routes
a chat completion to a named model and returns its text. `runtime_checkable`. No concrete
implementation of this exact port exists in `ragkit.core`/`ragkit.llm` today — `ragkit.llm.client.
LlmClient` plays the equivalent role for one endpoint but exposes a richer surface (`complete`,
`complete_json`, retry, usage stats) rather than this minimal `chat` shape.

#### `chat(self, messages: Sequence[Message], *, model: str, schema: Mapping[str, Any] | None, temperature: float, max_tokens: int) -> str`

**Args:**
- `messages` (`Sequence[Message]`): the conversation so far.
- `model` (`str`): the served model id to route to.
- `schema` (`Mapping[str, Any] | None`): a JSON Schema the reply should conform to, or `None` for free text.
- `temperature` (`float`): decode temperature.
- `max_tokens` (`int`): output token budget.

**Returns:** `str` — the model's reply text.

**Raises:** implementation-defined.

## ragkit.core.records

The intermediate work item — `Record` — and its durable catalogue and journal. A record is the
framework's single source of truth for one task instance (a line to translate, a natural-language
question to turn into SQL, a form to fill), and is deliberately *provenance-carrying*: it stores
the exact span it came from, so the produced output can be put back where it belongs rather than
re-derived. Everything task-specific lives in the free-form `Record.meta` mapping, which is frozen
at construction and must be JSON-serialisable, because a catalogue round-trips through JSON Lines.
The durability machinery here (atomic catalogue writes, a torn-record-tolerant journal reader) is
what lets a run of tens of thousands of records survive a crash and resume without repeating
finished work.

### Status

Lifecycle of a record. A `str`-valued `Enum` — the string value *is* the member, so it serialises
as itself.

**Members:**
- `PENDING = "pending"`: not yet processed.
- `PRODUCED = "produced"`: mechanically sound, but the review panel still objected when the budget ran out. Injectable like a verified record, kept under this status so the items a human might want to revisit stay findable in the catalogue.
- `VERIFIED = "verified"`: passed the mechanical checks and every reviewer.
- `REJECTED = "rejected"`: failed the mechanical checks, or the model produced nothing usable. Never injected — the output may be empty or wrong, so surfacing it would put a defect in front of the consumer.
- `SKIPPED = "skipped"`: the input needs no production (already in the desired form, or empty).

#### `is_injectable(self) -> bool` (property)

**Args:** none.

**Returns:** `bool` — whether a record in this state may have its output rendered back into a sink; `True` for `VERIFIED` and `PRODUCED`.

**Raises:** none.

#### `is_done(self) -> bool` (property)

**Args:** none.

**Returns:** `bool` — whether this record needs no further work; `True` for `VERIFIED` and `SKIPPED`.

**Raises:** none.

### CatalogError

A catalogue or journal is malformed or internally inconsistent. Subclasses `LocatedError` directly
(no custom `__init__`); carries the inherited `reason`, `context`, `path`, `line_no`.

### Record

One task instance, with everything needed to produce its output and reinject it. Frozen,
`slots`-based dataclass; `__post_init__` enforces its invariants (see Raises below), including
defensively freezing `meta` into a `MappingProxyType` copy so the record's state cannot be mutated
through a reference the caller kept.

**Attributes:**
- `record_id` (`str`): the record's stable identifier (see `make_record_id`).
- `source` (`str`): the input payload the task operates on — a self-contained unit, since a fragment cannot be handled well in isolation.
- `output` (`str | None`, default `None`): what the harness produced, once it has; `None` until then, required to be non-`None` for any injectable status.
- `status` (`Status`, default `Status.PENDING`): lifecycle state.
- `rel_path` (`str`, default `""`): where this record came from, named for reinjection; opaque to this module.
- `line_no` (`int`, default `0`): line number within `rel_path`.
- `span_start` (`int`, default `0`): where the payload begins in its source, in whatever unit the source uses; only ordering is interpreted here.
- `span_end` (`int`, default `0`): where it ends; half-open, so a payload's end is its successor's start. Must be `>= span_start`.
- `notes` (`tuple[str, ...]`, default `()`): free-form annotations.
- `meta` (`Mapping[str, Any]`, default `{}` — a shared `MappingProxyType({})`): task-specific extras, frozen and JSON-serialisable; the core never interprets these.

**Raises (via `__post_init__`, i.e. at construction):**
- `CatalogError`: `span_end < span_start`; `status.is_injectable` is `True` but `output is None`; or `meta` is not a `Mapping`.

#### `with_output(self, output: str, status: Status) -> Record`

**Args:**
- `output` (`str`): the produced output text.
- `status` (`Status`): the status to set alongside it.

**Returns:** `Record` — a copy of `self` with `output`/`status` replaced (via `dataclasses.replace`); everything else unchanged.

**Raises:** `CatalogError` (via the copy's `__post_init__`) if the new `status`/`output` combination violates the injectable-implies-output invariant.

#### `to_json(self) -> str`

**Args:** none.

**Returns:** `str` — the record as a single-line JSON object. Built field by field (not via `dataclasses.asdict`, which would deep-copy every field and cannot copy the read-only `MappingProxyType` guarding `meta`).

**Raises:** none explicitly.

#### `from_json(cls, text: str, *, path: Path | None = None, line_no: int | None = None) -> Record`

**Args:**
- `text` (`str`): one JSON Lines record.
- `path` (`Path | None`, default `None`): the source file, for error messages.
- `line_no` (`int | None`, default `None`): the line within `path`, for error messages.

**Returns:** `Record` — parsed and validated.

**Raises:**
- `CatalogError`: `text` is not valid JSON; the parsed value is not a JSON object; required fields `record_id`/`source` are missing; `meta` is present but not an object; `notes` is present but not an array; `line_no`/`span_start`/`span_end` is present but not a plain integer (explicitly checked here because `Record.__post_init__` would otherwise raise a bare `TypeError` that this method's own `except ValueError` would not catch); or the field values otherwise fail `Record`'s own construction (`Status(...)` raising `ValueError` for an unknown status, or any other `ValueError`/`TypeError` from `__post_init__`).

#### `make_record_id(rel_path: str, source: str, ordinal: int) -> str`

Deterministic id that survives the record moving within its source. Keyed on `(source name,
payload, occurrence index)` rather than position, so re-running extraction with different settings
keeps the ids of every payload that came out the same, and their outputs survive with them. The
occurrence index is what keeps a payload repeated verbatim from collapsing into one id.

**Args:**
- `rel_path` (`str`): the source file's name/path.
- `source` (`str`): the payload text.
- `ordinal` (`int`): the occurrence index of this payload within its source.

**Returns:** `str` — the first 16 hex characters of `sha1(f"{rel_path}\0{source}\0{ordinal}")`.

**Raises:** none.

#### `fsync_directory(path: Path) -> None`

Flush a directory entry, so a rename into it survives power loss. Renaming is atomic, but
atomicity is not durability: without this the new name can be lost while the old one is already
gone. Not every filesystem supports opening a directory; where unsupported, the rename is durable
anyway, so a refusal is silently ignored (not every side effect here is essential).

**Args:**
- `path` (`Path`): the directory to fsync.

**Returns:** `None`.

**Raises:** none — `OSError` opening or fsyncing the directory descriptor is caught and swallowed.

**Side effects:** opens and fsyncs the directory; closes the descriptor in a `finally`.

#### `write_catalog(records: Iterable[Record], path: Path) -> int`

Write records as JSON Lines, atomically. Writes a temporary sibling (`path` + `.partial`) and
renames it over the target, so a failure part-way through cannot leave a half-written catalogue
where a good one used to be; the data is fsynced before the rename.

**Args:**
- `records` (`Iterable[Record]`): the records to write.
- `path` (`Path`): the destination catalogue file.

**Returns:** `int` — the number of records written.

**Raises:** propagates whatever the source iterable or the filesystem raises (`BaseException`); on any such failure the partial temporary file is removed first, then the exception re-raised.

**Side effects:** creates `path.parent` if missing; writes and fsyncs a `.partial` temp file, then atomically renames it over `path` and fsyncs the parent directory (via `fsync_directory`); removes the temp file on failure.

#### `read_catalog(path: Path) -> Iterator[Record]`

Stream records from a JSON Lines catalogue, validating each line.

**Args:**
- `path` (`Path`): the catalogue file.

**Returns:** `Iterator[Record]` — one per non-blank line, in file order.

**Raises:**
- `CatalogError`: `path` is not a regular file, or any line fails `Record.from_json`.

#### `read_journal(journal: Path) -> Iterator[Record]`

Yield every result recorded in `journal`, in write order. A malformed *final* line without a
trailing newline is tolerated and dropped — the expected shape of a journal left behind by a
process killed mid-write, and the record it describes simply stays pending. A malformed line
anywhere else means the journal is corrupt and is not silently skipped, since that would discard a
completed result while still reporting success.

**Args:**
- `journal` (`Path`): the journal file; a missing file yields nothing.

**Returns:** `Iterator[Record]` — one per non-blank line, in write order. Holds the whole journal in memory (bounded by the catalogue size).

**Raises:**
- `CatalogError`: a non-final line (or a final line ending in a newline) fails `Record.from_json`.

#### `merge_journal(catalog: Path, journal: Path, output: Path) -> tuple[int, int]`

Fold journalled results into the catalogue. Later journal entries win, so re-running a record
supersedes its earlier result.

**Args:**
- `catalog` (`Path`): the existing catalogue file.
- `journal` (`Path`): the journal of results to merge in.
- `output` (`Path`): where to write the merged catalogue.

**Returns:** `tuple[int, int]` — `(records_written, records_updated)`.

**Raises:**
- `CatalogError`: `journal` does not exist; or the journal holds a result whose `record_id` has no matching record in `catalog` (refused rather than dropped, since silently discarding a completed result while reporting success is exactly the failure `read_journal` guards against, and usually means the journal and catalogue have drifted apart).

**Side effects:** writes `output` via `write_catalog` (atomic write + fsync).

#### `import_jsonl(store: RunStore, path: Path) -> int`

Import a JSON Lines catalogue into a `ragkit.core.ports.RunStore`'s record catalogue — the bridge
from a fetch script's unchanged JSONL output to a DB-native run.

**Args:**
- `store` (`RunStore`): the destination store.
- `path` (`Path`): the JSONL catalogue to import.

**Returns:** `int` — the number of records actually added (idempotent on an already-present `record_id`, per `RunStore.add_records`).

**Raises:** propagates `CatalogError` from `read_catalog` and whatever `store.add_records` raises.

**Side effects:** adds records to `store`.

#### `export_jsonl(store: RunStore, path: Path) -> int`

Export a `RunStore`'s results as a `journal.jsonl`-compatible file, atomically (reuses
`write_catalog`'s temp-then-rename write) — so a script written against the legacy journal format
keeps working unchanged over a DB-native run.

**Args:**
- `store` (`RunStore`): the source store.
- `path` (`Path`): the destination file.

**Returns:** `int` — the number of results written.

**Raises:** propagates whatever `write_catalog` raises.

**Side effects:** writes `path` atomically.

## ragkit.core.registry

The one extension mechanism: a typed component registry, one per port. Everything swappable in the
framework (a store driver, a retriever, a chunker, a model backend, a context block, a validator)
is a *component* resolved through a `Registry`, selected in config by name, dotted import path, or
published entry point, and constructed from its own validated options. A module-level registry is
global mutable state, which the project otherwise forbids — this is the deliberate, bounded
exception (the codec-registry pattern), earned by two rules: a registry holds only component types
and their option schemas, never per-run state; and resolution is a pure function of `(registry
contents, spec, options)` — a run reads the registry, never writes it.

#### `P`

A `TypeVar` — the generic parameter of `Registry[P]`, bound to whichever `Protocol` a given
registry instance is parameterised over (e.g. `Registry[VectorIndex]`).

### RegistryError

A component could not be registered, found, or constructed. Subclasses `RagkitError` directly (no
custom `__init__`); carries the inherited `reason`/`context`.

### Component (Protocol)

The convention a registrable component follows — describes only *how the registry builds it*, not
the port Protocol the component implements (which describes what it *does*). `runtime_checkable`.
Every concrete driver registered with a `Registry` (e.g. `LanceVectorIndex`, `SqliteRunStore`)
implicitly follows this convention by defining `CONFIG_KEYS` and, if it takes options, `from_config`.

**Attributes:**
- `CONFIG_KEYS` (`frozenset[str]`): the set of option keys the component accepts; the registry rejects any other key before construction.

#### `from_config(cls, options: Mapping[str, Any]) -> Any`

**Args:**
- `options` (`Mapping[str, Any]`): validated options (already checked against `CONFIG_KEYS`).

**Returns:** `Any` — a constructed instance of the component.

**Raises:** implementation-defined (a `ValueError`/`TypeError` raised for an invalid option value is caught by `Registry._build` and re-raised as `ConfigError`).

### Registry (Generic[P])

A named collection of components that all satisfy one `Protocol`. Construct one per port
(`VectorIndex`, `Retriever`, `Backend`, ...) via `Registry(kind: str, protocol: type[P], *,
entry_point_group: str | None = None, entry_point_loader: EntryPointLoader | None = None)`; the
constructor raises `RegistryError` immediately if `protocol` is not decorated
`@runtime_checkable`, since conformance is verified with `isinstance` at construction time.

#### `register(self, name: str, component: type, *, aliases: Iterable[str] = ()) -> type`

Register `component` under `name` and any `aliases`. Returns it, so this can decorate a class.

**Args:**
- `name` (`str`): the canonical name to register under.
- `component` (`type`): the class to register.
- `aliases` (`Iterable[str]`, default `()`): additional names that also resolve to `component`.

**Returns:** `type` — `component`, unchanged (so `register` can be used as a class decorator).

**Raises:**
- `RegistryError`: `component` is not a class; `component` structurally fails the port (checked via `issubclass` against the `runtime_checkable` protocol, when that check is possible); or `name`/an alias is already registered to a *different* component (re-registering the *same* component under the same name is a no-op, so importing a module twice is harmless).

**Side effects:** mutates `self._entries`.

#### `create(self, spec: str, options: Mapping[str, Any] | None = None, *, path: Path | None = None) -> P`

Resolve `spec` to a component and build it from `options`. `spec` is resolved in this precedence:
an explicit dotted path (`"pkg.mod:Attr"`, recognised by the colon), then a registered built-in
name, then — if this registry was given an entry-point group — a published entry point. The
resolved component's own `CONFIG_KEYS` gate `options` (unknown keys refused), `from_config` builds
it, then the result is checked against the port both for having every required member and for
each of those being callable with the port's keyword-only parameters. This is a strong structural
check, not a total one — parameter types, return values, and behaviour are not verified.

**Args:**
- `spec` (`str`): a registered name/alias, a `"module:attr"` dotted path, or an entry-point name.
- `options` (`Mapping[str, Any] | None`, default `None`): the component's construction options; treated as `{}` when `None`.
- `path` (`Path | None`, default `None`): the config file `options` came from, for error messages.

**Returns:** `P` — the constructed component, verified to satisfy the registry's protocol.

**Raises:**
- `RegistryError`: `spec` matches no dotted path, built-in name, or entry point; the dotted path is malformed, its module cannot be imported, or its attribute is missing or not a class; the resolved component's `CONFIG_KEYS` is not a `set`/`frozenset`; the resolved component defines no `from_config` but was given non-empty `options`; the built instance does not satisfy the protocol (`isinstance` check); or it has the protocol's members but with an incompatible keyword-only signature.
- `ConfigError`: `options` contains a key outside the component's `CONFIG_KEYS`; or `from_config` itself raises `ConfigError`, or raises `ValueError`/`TypeError` (wrapped into `ConfigError` naming the component).

#### `available(self) -> list[str]`

**Args:** none.

**Returns:** `list[str]` — the distinct registered built-in canonical names, sorted.

**Raises:** none.

## ragkit.core.rules

The shared vocabulary of a rule violation: `Severity` and `Violation`. A `Validator` inspects a
produced output and returns zero or more `Violation`s; whether a violation *blocks* acceptance is
carried by its `Severity`, so the harness can treat "must be fixed" and "worth noting" differently
without re-deciding per rule. Lives in the core because both the validators (which produce
violations) and the harness (which acts on them) depend on the same shape.

### Severity

A `str`-valued `Enum`.

**Members:**
- `ERROR = "error"`: the output must be rejected and re-produced (or, past the budget, handled per the rule's exhaustion policy).
- `WARNING = "warning"`: recorded for review; does not block acceptance.

### Violation

One rule violation: which rule, how severe, and a message the model can act on. Frozen,
`slots`-based dataclass.

**Attributes:**
- `rule_id` (`str`): identifies which rule was violated.
- `severity` (`Severity`): how severe.
- `message` (`str`): a message the model (or a human) can act on.

#### `blocking(self) -> bool` (property)

**Args:** none.

**Returns:** `bool` — `True` when `severity is Severity.ERROR`.

**Raises:** none.

#### `blocking(violations: Sequence[Violation]) -> list[Violation]`

Just the blocking (error-severity) violations, preserving order. Lives beside `Violation` and is
imported by `ragkit.harness.validators`, which used to carry a byte-identical second copy — two
definitions of "which violations stop a run" being one more than the rule can survive.

**Args:**
- `violations` (`Sequence[Violation]`): candidate violations.

**Returns:** `list[Violation]` — the subset with `.blocking is True`, in order.

**Raises:** none.

## ragkit.core.width

How wide a piece of text is when a fixed-cell renderer draws it. In the core because a length
constraint is meaningless unless every side measures it the same way — a component that knows a
field fits twenty columns and a producer that counts characters will disagree the moment CJK is
involved, invisibly, until text overflows in the running program. Measured in half-width columns
via `unicodedata.east_asian_width`, which covers the whole of Unicode (including the supplementary-
plane CJK extensions a hand-rolled range list would miss). The model is deliberately context-free:
each character costs the same wherever it appears, so width is additive over concatenation.

#### `display_columns(text: str) -> int`

Width of `text` in half-width display columns. Marks, format characters, and composing Hangul jamo
add nothing (they render into a neighbouring cell, or nothing at all); East-Asian wide and full-
width characters count two; everything else, including East-Asian *ambiguous*, counts one — the
conservative reading for a non-CJK terminal. Because every character's cost is fixed, the total is
additive: `display_columns(a + b) == display_columns(a) + display_columns(b)` for all strings, so
callers may measure a payload in pieces and add up, tracking a remaining budget incrementally.

**Known, deliberate over-count:** a sequence joined by ZWJ (e.g. the family emoji, three code
points plus two joiners) renders as one grapheme but is measured per code point (6 columns here
against ~2 as drawn). Not fixed because grapheme-cluster segmentation is unavailable in the
standard library, and every available approximation either under-counts without limit (breaking
the budget's purpose) or breaks additivity for a partial answer. The model stays context-free and
honestly over-counts rather than degrading silently on text that merely looks similar.

**Args:**
- `text` (`str`): the text to measure.

**Returns:** `int` — total half-width columns, `sum` of each character's cost (`0`, `1`, or `2`).

**Raises:** none.
