"""The seams: the ``Protocol`` every replaceable component implements, and the value types
they exchange.

These are the contracts that make the framework decoupled. A layer depends on a port here,
never on a concrete driver; the concrete driver is resolved through the matching
:class:`~ragkit.core.registry.Registry` and checked against the port with ``isinstance``. Each
port is ``runtime_checkable`` so that check works whether it is defined by methods, attributes,
or both.

Signatures are kept deliberately small and hard to misuse (``CLAUDE.md``: small, stable,
hard-to-misuse public APIs). Where a later milestone implements a port it may add optional
parameters, but the shape here is the stable core each driver must honour.

A shared score convention runs through the retrieval ports: **every score a port returns is
"higher is better", normalised toward ``[0, 1]``.** A backend whose native scale is inverted
(SQLite's ``bm25()`` is negative, lower-is-better) or unbounded (a cross-encoder logit) is the
driver's problem to convert *inside* the driver, so a sign convention can never leak into the
fuser and silently invert a ranking.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from .records import Record
from .rules import Violation

# --- shared value types -------------------------------------------------------

_EMPTY: Mapping[str, Any] = {}


@dataclass(frozen=True, slots=True)
class Message:
    """One chat turn — the unit a model client and a backend operate on."""

    role: str
    content: str


def _in_range(value: float | None, low: float, high: float, *, what: str) -> None:
    if value is not None and not (low <= value <= high):
        raise ValueError(f"{what} must be in [{low}, {high}], got {value}")


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Per-request decode settings for one chat completion — the knobs an OpenAI-compatible server
    accepts *per call* (temperature, nucleus/top-k/min-p truncation, penalties, seed, stop).

    These are owned by the **persona** that issues the request, because the right setting is a
    property of the role, not the model: a producer may want a little warmth, a reviewer wants
    determinism. Settings that a server can only apply at *launch* (context size, GPU offload,
    KV-cache type) are not here — they live in ``models.toml`` and drive the serve script.

    ``max_tokens`` is deliberately *not* a field: it is a computed budget (scaled to the input for
    the producer, a ceiling for a reviewer) passed separately, so the size budget and the decode
    style stay separate concerns. Only fields that are set are sent, so the default request carries
    just a temperature and a strict-OpenAI endpoint is never handed a llama.cpp-only knob it would
    reject.
    """

    temperature: float = 0.2
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repeat_penalty: float | None = None
    stop: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        _in_range(self.top_p, 0.0, 1.0, what="top_p")
        _in_range(self.min_p, 0.0, 1.0, what="min_p")
        _in_range(self.presence_penalty, -2.0, 2.0, what="presence_penalty")
        _in_range(self.frequency_penalty, -2.0, 2.0, what="frequency_penalty")
        if self.top_k is not None and self.top_k < 0:
            raise ValueError(f"top_k must be >= 0 (0 disables it), got {self.top_k}")
        if self.repeat_penalty is not None and self.repeat_penalty <= 0:
            raise ValueError(f"repeat_penalty must be > 0 (1.0 is no penalty), got "
                             f"{self.repeat_penalty}")

    def payload(self) -> dict[str, Any]:
        """The request-body fragment: temperature always, plus every optional knob that is set.
        An unset knob is omitted rather than sent as a default, so a server never has to interpret
        a value the persona did not choose."""
        body: dict[str, Any] = {"temperature": self.temperature}
        for key in ("top_p", "top_k", "min_p", "seed", "presence_penalty", "frequency_penalty",
                    "repeat_penalty"):
            value = getattr(self, key)
            if value is not None:
                body[key] = value
        if self.stop:
            body["stop"] = list(self.stop)
        return body


@dataclass(frozen=True, slots=True)
class StructuredRequest:
    """How to ask a specific model for schema-conforming JSON: the (possibly rewritten)
    messages and the ``response_format`` fragment to put in the request body. A backend that
    must describe the schema in the prompt returns rewritten messages; one that constrains
    decoding returns the messages unchanged."""

    messages: tuple[Message, ...]
    response_format: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Document:
    """A source document after extraction, before chunking."""

    doc_id: str
    text: str
    meta: Mapping[str, Any] = field(default_factory=lambda: _EMPTY)


@dataclass(frozen=True, slots=True)
class Chunk:
    """One indexable unit of a document: the text, an id, and its provenance in ``meta``
    (``document_id``, ``version_id``, ``ordinal``, ``section_path``, offsets, ...)."""

    chunk_id: str
    text: str
    meta: Mapping[str, Any] = field(default_factory=lambda: _EMPTY)


@dataclass(frozen=True, slots=True)
class Retrieved:
    """A chunk a query matched, with its relevance score (higher is better, toward ``[0, 1]``)."""

    chunk_id: str
    text: str
    score: float
    meta: Mapping[str, Any] = field(default_factory=lambda: _EMPTY)


class FilterOp(str, Enum):
    """Comparisons a metadata filter predicate can express."""

    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    IN = "in"


@dataclass(frozen=True, slots=True)
class Predicate:
    """One metadata comparison: ``field <op> value``."""

    field: str
    op: FilterOp
    value: Any


# A filter is a conjunction (AND) of predicates -- the framework's own small, backend-neutral
# filter language. Each store driver compiles it to its own dialect; a driver that cannot honour
# a predicate refuses it at that boundary rather than at query time. Deliberately not raw backend
# filter dicts, which would couple every caller to one driver.
Filter = tuple[Predicate, ...]


# --- ingestion ----------------------------------------------------------------


@runtime_checkable
class Source(Protocol):
    """The upstream boundary toward the input: yields the records a run will work on."""

    def records(self) -> Iterator[Record]: ...


@runtime_checkable
class Sink(Protocol):
    """The downstream boundary toward the output: writes produced records back to their home."""

    def write(self, records: Iterable[Record]) -> None: ...


@runtime_checkable
class Extractor(Protocol):
    """Turns a raw source (a file's bytes, a row set) into documents ready to chunk."""

    def extract(self, source: bytes | str, *, meta: Mapping[str, Any] = _EMPTY
                ) -> Iterator[Document]: ...


@runtime_checkable
class Chunker(Protocol):
    """Splits a document into indexable chunks, preserving provenance in each chunk's meta."""

    def chunk(self, document: Document) -> Iterator[Chunk]: ...


@runtime_checkable
class Embedder(Protocol):
    """Encodes texts into dense vectors. Rows are returned in input order."""

    def embed(self, texts: Sequence[str]) -> list[Sequence[float]]: ...


# --- storage ------------------------------------------------------------------


@runtime_checkable
class VectorIndex(Protocol):
    """A dense-vector store. ``search`` returns ``(chunk_id, score)`` best-first, score
    higher-is-better toward ``[0, 1]``."""

    def upsert(self, ids: Sequence[str], vectors: Sequence[Sequence[float]],
               metas: Sequence[Mapping[str, Any]]) -> None: ...

    def search(self, vector: Sequence[float], *, k: int,
               where: Filter = ()) -> list[tuple[str, float]]: ...

    def delete(self, ids: Sequence[str]) -> None: ...

    def count(self) -> int: ...

    def reconcile(self, chunk_ids: Iterable[str]) -> set[str]:
        """Reconcile the index against the authoritative set of chunk ids: **drop orphan
        vectors** (indexed ids no longer in ``chunk_ids``) and **return the ids that are missing
        a vector** (in ``chunk_ids`` but not indexed), for the caller to re-embed or refuse. The
        index cannot re-embed itself — it has neither the text nor the embedder — so it reports
        the gap rather than hiding it. A driver that co-locates vectors with the rows (so drift
        is impossible) returns an empty set without doing anything."""
        ...


@runtime_checkable
class SearchIndex(Protocol):
    """The read side of a keyword search index: maps a query to ``(chunk_id, score)`` best-first,
    higher-is-better. This is the *narrow* interface a :class:`Retriever` built over a search index
    actually depends on (never the write side) — every :class:`LexicalIndex` satisfies it, and so
    does a :class:`PairingStore` (its rows and search index are co-located, but a retriever never
    indexes or deletes through it directly)."""

    def search(self, query: str, *, k: int) -> list[tuple[str, float]]: ...


@runtime_checkable
class LexicalIndex(SearchIndex, Protocol):
    """A keyword/BM25 index. ``search`` returns ``(chunk_id, score)`` best-first, score
    higher-is-better (a driver over an inverted BM25 converts the native scale itself)."""

    def index(self, chunk_id: str, text: str) -> None: ...

    def delete(self, chunk_id: str) -> None: ...


@runtime_checkable
class DocumentStore(Protocol):
    """The relational home for chunk rows — the single store every retrieval path resolves a hit
    through, so a corpus of any size lives in the database rather than a RAM map. A search index
    (BM25, ANN) returns ids; this turns an id back into its display text + metadata. ``count`` is
    the corpus size, used to skip re-ingesting an already-built store."""

    def add_documents(self, rows: Iterable[tuple[str, str, Mapping[str, Any]]]) -> None: ...

    def document(self, chunk_id: str) -> tuple[str, Mapping[str, Any]] | None: ...

    def count(self) -> int: ...


@dataclass(frozen=True, slots=True)
class Pairing:
    """One reference example held in a :class:`PairingStore`: an input, the target it pairs with
    (empty for a lexical-only reference entry), and the context it was produced with. This is the
    ``(source, context, target)`` triple the reference memory stores and a write-back step
    produces. ``verified``/``created_at`` are write-back provenance (was this machine-produced and
    accepted, and when); an imported reference entry leaves them at their defaults."""

    chunk_id: str
    source: str
    target: str = ""
    context: str = ""
    meta: Mapping[str, Any] = field(default_factory=lambda: _EMPTY)
    verified: bool = False
    created_at: float = 0.0


@runtime_checkable
class PairingStore(Protocol):
    """The reference-memory store: rows and a keyword search index co-located in one durable store,
    so a hit and its display text can never drift apart the way two separately-written stores can.

    ``search`` and ``document`` deliberately share :class:`LexicalIndex`'s and
    :class:`DocumentStore`'s exact signatures: a driver that implements this port also *is* a valid
    ``LexicalIndex`` and ``DocumentStore``, so the existing lexical/dense/hybrid retriever stack
    runs over a pairing store unchanged, with no parallel retrieval code path to keep in sync.
    """

    def add(self, pairings: Iterable[Pairing]) -> int:
        """Add pairings in one transaction (their search entries included); returns the number of
        rows actually added (a pairing whose ``chunk_id`` already exists is left untouched, not
        overwritten, so re-adding the same write-back result twice is idempotent)."""
        ...

    def search(self, query: str, *, k: int) -> list[tuple[str, float]]: ...

    def document(self, chunk_id: str) -> tuple[str, Mapping[str, Any]] | None: ...

    def get(self, chunk_id: str) -> Pairing | None:
        """The full pairing for ``chunk_id`` (source, target, context, meta, verification,
        provenance), or ``None`` if absent. Unlike ``document``, which flattens a pairing to display
        text for a retriever, this is the write-back / inspection path that needs the whole row."""
        ...

    def all_ids(self) -> Iterator[str]:
        """Every chunk id currently stored, for a caller reconciling a separate vector index
        against this store's authoritative rows (:meth:`VectorIndex.reconcile`)."""
        ...

    def count(self) -> int: ...


@runtime_checkable
class SqlStore(Protocol):
    """A relational store. ``read_only`` marks a binding the framework must not write through;
    a write attempt on one is refused at the port, before the database. ``query`` runs a
    read; ``execute`` a write (and raises on a ``read_only`` binding)."""

    read_only: bool

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[Mapping[str, Any]]: ...

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None: ...


@runtime_checkable
class SchemaIntrospector(Protocol):
    """Reads a database's schema for the NL->SQL feature, without importing a store driver.
    Returns a mapping of table name to its ordered column ``(name, type)`` pairs."""

    def schema(self) -> Mapping[str, Sequence[tuple[str, str]]]: ...


@dataclass(frozen=True, slots=True)
class RetrievedRef:
    """A JSON-safe projection of a :class:`Retrieved` hit, for persistence in a :class:`RunResult`:
    ``(chunk_id, text, score)`` only, no ``meta`` — meta need not be JSON-safe, and is reference
    data the pairing store already holds, keyed by ``chunk_id``, so there is nothing to duplicate
    into the run store."""

    chunk_id: str
    text: str
    score: float


@dataclass(frozen=True, slots=True)
class RunResult:
    """One completed attempt at a record, as a :class:`RunStore` persists it.

    ``record`` already carries its verdict — the status/output/notes an ``Outcome`` applies to it
    (see ``harness.agents.Outcome.applied_to``) — while the fields here are what the legacy JSONL
    journal could not hold: the structured violations (never flattened to strings), and the context
    that actually produced the output (see ``harness.capture``). ``reviews`` is a tuple of plain,
    already-JSON-safe mappings rather than the harness's own ``Review`` type, so this port need not
    import the harness layer (a lower layer never depends on one above it).
    """

    record: Record
    context_passage: str = ""
    retrieved: tuple[RetrievedRef, ...] = ()
    reviews: tuple[Mapping[str, Any], ...] = ()
    violations: tuple[Violation, ...] = ()
    rounds: int = 0
    error: str | None = None


@runtime_checkable
class RunStore(Protocol):
    """The framework's own run-state store: the record catalogue and the append-only result
    history, replacing the JSONL catalogue+journal pair (see docs/storage-overhaul-plan.md). A
    result write is one transaction, so a torn/partial row is impossible — the durability the JSONL
    journal approximated with a per-line fsync and a reader tolerant of a torn *final* line only."""

    def add_records(self, records: Iterable[Record]) -> int:
        """Add records to the catalogue (idempotent on an already-present ``record_id``, so
        re-running an import is safe). Returns the number actually added."""
        ...

    def append_result(self, result: RunResult) -> None:
        """Persist one completed attempt in a single transaction. Never overwrites an earlier
        attempt at the same record — each call adds a new result, and :meth:`results` /
        :meth:`completed_ids` resolve to the latest by write order."""
        ...

    def completed_ids(self) -> set[str]:
        """Ids of every record with at least one result — what a resumed run must skip."""
        ...

    def pending(self) -> Iterator[Record]:
        """Records with no result yet, in the store's stable catalogue order (by provenance:
        ``rel_path`` then ``line_no``) — a streamed cursor, never materialising the whole
        catalogue in memory."""
        ...

    def results(self) -> Iterator[RunResult]:
        """The latest result for every record that has one."""
        ...

    def count_records(self) -> int: ...


# --- retrieval ----------------------------------------------------------------


@runtime_checkable
class Retriever(Protocol):
    """Retrieves the chunks most relevant to a query, best-first, each with score >=
    ``min_score``. Required to be deterministic and safe to call concurrently — one retriever
    is shared by every worker in a run."""

    def retrieve(self, query: str, *, k: int, min_score: float = 0.0) -> tuple[Retrieved, ...]: ...


@runtime_checkable
class Reranker(Protocol):
    """Re-scores a candidate shortlist against a query. Returns ``(index, score)`` into the
    given documents, best-first; ``index`` is a permutation of ``range(len(documents))``."""

    def rerank(self, query: str, documents: Sequence[str]) -> list[tuple[int, float]]: ...


# --- harness ------------------------------------------------------------------


@runtime_checkable
class ContextBlock(Protocol):
    """One prompt section. Returns the rendered text, or ``None`` when it has no content — the
    no-empty-section rule is enforced here, so an absent block contributes nothing rather than
    a dangling heading."""

    def render(self, record: Record, context: Mapping[str, Any]) -> str | None: ...


@runtime_checkable
class Validator(Protocol):
    """A mechanical (code-decidable) check over a produced output. Returns the violations it
    finds, empty when the output passes. Must abstain (return no *blocking* violation) rather
    than guess when the input gives it no positive evidence to judge on."""

    def validate(self, record: Record, output: str,
                 context: Mapping[str, Any]) -> list[Violation]: ...


@runtime_checkable
class OutputSchema(Protocol):
    """Describes the structured output a task produces: its JSON schema (what the model is
    constrained or asked to return) and how to pull the output string out of a parsed reply."""

    name: str

    def json_schema(self) -> dict[str, Any]: ...

    def extract(self, reply: Mapping[str, Any]) -> str: ...


# --- model layer --------------------------------------------------------------


@runtime_checkable
class Backend(Protocol):
    """Request-shaping for one model family: turns messages + a JSON schema into the
    ``(messages, response_format)`` a specific server will honour (strict grammar, or a
    prompt-described shape)."""

    name: str

    def structured_request(self, messages: Sequence[Message],
                           schema: Mapping[str, Any]) -> StructuredRequest: ...


@runtime_checkable
class Provider(Protocol):
    """A transport to one endpoint kind (llama.cpp router, ollama, any OpenAI-compatible
    server). Routes a chat completion to a named model and returns its text."""

    def chat(self, messages: Sequence[Message], *, model: str, schema: Mapping[str, Any] | None,
             temperature: float, max_tokens: int) -> str: ...
