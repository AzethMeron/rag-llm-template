"""Built-in context blocks, and the registry a config selects them through.

A context block renders one prompt section from the record and the run's shared services, or
returns ``None`` when it has no content — the no-empty-section rule lives here, so an absent block
contributes nothing rather than a dangling heading. The ordered list of blocks a run assembles is
config (see :mod:`ragkit.harness.context.assembler`); a bespoke block is named by dotted path and
resolved through :data:`CONTEXT_BLOCKS` exactly like a built-in.

The shared services a block may read from the ``context`` mapping (all optional):

* ``lexicon`` — the established terminology (``list[Entry]``);
* ``memory`` — outputs already produced this run (an :class:`~ragkit.harness.memory.OutputMemory`);
* ``retriever`` — a reference :class:`~ragkit.core.ports.Retriever` for worked examples;
* ``sql_store`` — a read-only :class:`~ragkit.core.ports.SqlStore` for the form-autofill task;
* ``previous_attempt`` — on revision, the attempt being fixed (``.target``, ``.issues``);
* ``stand_in`` — a readable replacement for a placeholder shown in a *neighbour* line;
* ``capture`` — the run's :class:`~ragkit.harness.capture.Capture` sink, if wired; a block that
  already retrieves (``retrieved``) reports its hits there rather than the caller re-retrieving.

A block that is configured but whose required service is absent **raises** rather than silently
rendering nothing — a missing wiring is a misconfiguration, not an empty section.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from ragkit.core.errors import RagkitError
from ragkit.core.lexicon import Entry, relevant_entries
from ragkit.core.ports import ContextBlock, Retriever, SchemaIntrospector, SqlStore
from ragkit.core.records import Record
from ragkit.core.registry import Registry

from ..capture import capture_retrieved

_PLACEHOLDER = re.compile(r"\[\[\d+\]\]")


class ContextBlockError(RagkitError):
    """A context block is misconfigured, or a service it needs is not wired in."""


CONTEXT_BLOCKS: Registry[ContextBlock] = Registry(
    "context block", ContextBlock,  # type: ignore[type-abstract]
    entry_point_group="ragkit.context_blocks")
"""Registry for context-block components. The port parameterises the registry (a Protocol, which
mypy flags as abstract); the ignore is scoped to exactly that argument."""


def _mask(text: str, stand_in: str) -> str:
    """Replace engine placeholders in a *neighbour* line with a readable stand-in: a placeholder
    there refers to that line's own lookup, not this record's, so shown raw it teaches the model to
    emit one on a line that has none."""
    return _PLACEHOLDER.sub(stand_in, text)


def _section(heading: str, body: str) -> str:
    return f"{heading}\n{body}" if heading else body


def _meta_strings(record: Record, key: str, limit: int, *, from_end: bool) -> tuple[str, ...]:
    raw = record.meta.get(key, ())
    if not isinstance(raw, (list, tuple)) or limit <= 0:
        return ()
    items = tuple(str(x) for x in raw)
    return items[-limit:] if from_end else items[:limit]


class LiteralBlock:
    """A fixed block of text (a grounding instruction, a format reminder). Always present."""

    CONFIG_KEYS = frozenset({"text", "heading"})

    def __init__(self, text: str, heading: str = "") -> None:
        if not text.strip():
            raise ContextBlockError("a literal context block needs non-empty 'text'")
        self._text = text
        self._heading = heading

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> LiteralBlock:
        return cls(text=str(options.get("text", "")), heading=str(options.get("heading", "")))

    def render(self, record: Record,  # noqa: ARG002  -- required by the ContextBlock port
               context: Mapping[str, Any]) -> str | None:  # noqa: ARG002
        return _section(self._heading, self._text)


class LexiconBlock:
    """The established terms whose term occurs in the input, as "use these renderings"."""

    _DEFAULT_HEADING = "Established terminology (use these renderings):"
    CONFIG_KEYS = frozenset({"heading", "limit"})

    def __init__(self, heading: str = _DEFAULT_HEADING, limit: int = 12) -> None:
        self._heading = heading
        self._limit = limit

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> LexiconBlock:
        return cls(heading=str(options.get("heading", cls._DEFAULT_HEADING)),
                   limit=int(options.get("limit", 12)))

    def render(self, record: Record, context: Mapping[str, Any]) -> str | None:
        lexicon: Sequence[Entry] = context.get("lexicon", ())
        hits = relevant_entries(record.source, list(lexicon), limit=self._limit)
        if not hits:
            return None
        body = "\n".join(f"  {e.term} -> {e.rendering}"
                         + (f"  [{e.category}]" if e.category else "") for e in hits)
        return _section(self._heading, body)


class NeighboursBlock:
    """The surrounding lines (``context_before``/``context_after`` in the record's meta) as one
    continuous passage, masked. A record's line is often a fragment split across boundaries, and a
    model reads continuous prose the way it read the source."""

    _DEFAULT_HEADING = ("Surrounding context (do not translate; the line to act on is given again "
                        "below):")
    CONFIG_KEYS = frozenset({"before", "after", "heading"})

    def __init__(self, before: int = 3, after: int = 2, heading: str = _DEFAULT_HEADING) -> None:
        if before < 0 or after < 0:
            raise ContextBlockError("neighbours block: before/after must be >= 0")
        self._before = before
        self._after = after
        self._heading = heading

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> NeighboursBlock:
        return cls(before=int(options.get("before", 3)), after=int(options.get("after", 2)),
                   heading=str(options.get("heading", cls._DEFAULT_HEADING)))

    def render(self, record: Record, context: Mapping[str, Any]) -> str | None:
        stand_in = str(context.get("stand_in", "they"))
        before = _meta_strings(record, "context_before", self._before, from_end=True)
        after = _meta_strings(record, "context_after", self._after, from_end=False)
        if not before and not after:
            return None
        lines = [_mask(line, stand_in) for line in before]
        lines.append(record.source)
        lines.extend(_mask(line, stand_in) for line in after)
        return _section(self._heading, "\n".join(lines))


class EstablishedBlock:
    """Neighbour lines that already have a produced output, paired with it — so the model sees how
    a neighbour was actually rendered, keeping terminology and pronouns continuous."""

    _DEFAULT_HEADING = "Already-produced neighbouring lines:"
    CONFIG_KEYS = frozenset({"before", "after", "heading"})

    def __init__(self, before: int = 3, after: int = 2, heading: str = _DEFAULT_HEADING) -> None:
        self._before = before
        self._after = after
        self._heading = heading

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> EstablishedBlock:
        return cls(before=int(options.get("before", 3)), after=int(options.get("after", 2)),
                   heading=str(options.get("heading", cls._DEFAULT_HEADING)))

    def render(self, record: Record, context: Mapping[str, Any]) -> str | None:
        memory = context.get("memory")
        if memory is None:
            return None
        stand_in = str(context.get("stand_in", "they"))
        neighbours = (_meta_strings(record, "context_before", self._before, from_end=True)
                      + _meta_strings(record, "context_after", self._after, from_end=False))
        rendered = [f"  {_mask(line, stand_in)} -> {_mask(output, stand_in)}"
                    for line in neighbours if (output := memory.get(line)) is not None]
        if not rendered:
            return None
        return _section(self._heading, "\n".join(rendered))


class RetrievedBlock:
    """Reference examples retrieved for this record, as worked examples to align with. Requires a
    ``retriever`` in the context; configured without one is a misconfiguration."""

    _DEFAULT_HEADING = ("Reference examples of similar inputs (match their choices where they "
                        "apply):")
    CONFIG_KEYS = frozenset({"heading", "k", "min_score"})

    def __init__(self, heading: str = _DEFAULT_HEADING, k: int = 3, min_score: float = 0.3) -> None:
        if k < 1:
            raise ContextBlockError("retrieved block: k must be >= 1")
        if not 0.0 <= min_score <= 1.0:
            raise ContextBlockError("retrieved block: min_score must be in [0, 1]")
        self._heading = heading
        self._k = k
        self._min_score = min_score

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> RetrievedBlock:
        return cls(heading=str(options.get("heading", cls._DEFAULT_HEADING)),
                   k=int(options.get("k", 3)), min_score=float(options.get("min_score", 0.3)))

    def render(self, record: Record, context: Mapping[str, Any]) -> str | None:
        retriever = context.get("retriever")
        if retriever is None:
            raise ContextBlockError(
                "a 'retrieved' context block is configured but no retriever was wired into the "
                "run; supply one or remove the block")
        if not isinstance(retriever, Retriever):
            raise ContextBlockError("the wired 'retriever' does not satisfy the Retriever port")
        hits = retriever.retrieve(record.source, k=self._k, min_score=self._min_score)
        capture_retrieved(context, hits)
        if not hits:
            return None
        return _section(self._heading, "\n".join(f"  {hit.text}" for hit in hits))


class PreviousAttemptBlock:
    """On revision, the exact attempt being fixed and the objections against it — so the model
    revises rather than re-produces blind. Absent on the first pass."""

    CONFIG_KEYS = frozenset({"heading"})

    def __init__(self, heading: str = "Your previous attempt:") -> None:
        self._heading = heading

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> PreviousAttemptBlock:
        return cls(heading=str(options.get("heading", "Your previous attempt:")))

    def render(self, record: Record,  # noqa: ARG002  -- required by the ContextBlock port
               context: Mapping[str, Any]) -> str | None:
        attempt = context.get("previous_attempt")
        if attempt is None:
            return None
        # The "previous attempt" can carry an empty target -- a first-round content/truncation
        # error produces an Attempt with target="" -- so only show the target section when there is
        # a target, never a dangling heading (the no-empty-section rule).
        parts = [_section(self._heading, attempt.target)] if attempt.target.strip() else []
        if attempt.issues:
            parts.append("It was sent back for these reasons. Fix exactly these and change nothing "
                         "else that already works:\n"
                         + "\n".join(f"- {issue}" for issue in attempt.issues))
        suggestions = getattr(attempt, "suggestions", ())
        if suggestions:
            parts.append("A reviewer proposed this rewrite; adopt whatever is right about it, but "
                         "you own the final output:\n"
                         + "\n".join(f"- {s}" for s in suggestions))
        return "\n\n".join(parts)


class SqlRowsBlock:
    """Rows retrieved from an external read-only database for this record — the form-autofill
    task's historical-record context. Requires a ``sql_store`` in the context."""

    CONFIG_KEYS = frozenset({"heading", "query", "param_keys", "limit"})

    def __init__(self, query: str, heading: str = "Relevant historical records:",
                 param_keys: Sequence[str] = (), limit: int = 20) -> None:
        if not query.strip():
            raise ContextBlockError("sql_rows block needs a non-empty 'query'")
        if limit < 1:
            raise ContextBlockError("sql_rows block: limit must be >= 1")
        self._query = query
        self._heading = heading
        self._param_keys = tuple(param_keys)
        self._limit = limit

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> SqlRowsBlock:
        return cls(query=str(options.get("query", "")),
                   heading=str(options.get("heading", "Relevant historical records:")),
                   param_keys=tuple(options.get("param_keys", ())),
                   limit=int(options.get("limit", 20)))

    def render(self, record: Record, context: Mapping[str, Any]) -> str | None:
        store = context.get("sql_store")
        if store is None:
            raise ContextBlockError(
                "a 'sql_rows' context block is configured but no sql_store was wired into the run")
        if not isinstance(store, SqlStore):
            raise ContextBlockError("the wired 'sql_store' does not satisfy the SqlStore port")
        params = [record.meta.get(key) for key in self._param_keys]
        rows = store.query(self._query, params)[:self._limit]
        if not rows:
            return None
        body = "\n".join("  " + ", ".join(f"{k}={v!r}" for k, v in row.items()) for row in rows)
        return _section(self._heading, body)


class SchemaBlock:
    """The introspected schema of the external database, rendered as ``CREATE TABLE``-like text — so
    the NL->SQL producer sees what tables and columns exist. Requires a ``introspector`` in the
    context."""

    CONFIG_KEYS = frozenset({"heading"})

    def __init__(self, heading: str = "Database schema (tables and their columns):") -> None:
        self._heading = heading

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> SchemaBlock:
        return cls(heading=str(options.get("heading",
                                           "Database schema (tables and their columns):")))

    def render(self, record: Record,  # noqa: ARG002  -- required by the ContextBlock port
               context: Mapping[str, Any]) -> str | None:
        introspector = context.get("introspector")
        if introspector is None:
            raise ContextBlockError(
                "a 'schema' context block is configured but no introspector was wired into the run")
        if not isinstance(introspector, SchemaIntrospector):
            raise ContextBlockError("the wired 'introspector' does not satisfy the port")
        schema = introspector.schema()
        if not schema:
            return None
        lines = [f"  {table}(" + ", ".join(f"{name} {ctype}".strip() for name, ctype in columns)
                 + ")" for table, columns in schema.items()]
        return _section(self._heading, "\n".join(lines))


class ReadingsBlock:
    """Named fields of *this record's* request, rendered into the prompt — the sensor telemetry and
    fault codes a decision task is given (``record.meta``). Unlike ``sql_rows`` (rows fetched from a
    database) this reads only the record itself, so it needs no wired service. ``keys`` selects
    which meta fields to show, in that order; absent keys are skipped, and a block with nothing to
    show renders nothing (the no-empty-section rule)."""

    CONFIG_KEYS = frozenset({"heading", "keys"})

    def __init__(self, keys: Sequence[str] = (),
                 heading: str = "Reported readings and codes:") -> None:
        self._keys = tuple(keys)
        self._heading = heading

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> ReadingsBlock:
        return cls(keys=tuple(options.get("keys", ())),
                   heading=str(options.get("heading", "Reported readings and codes:")))

    def render(self, record: Record,
               context: Mapping[str, Any]) -> str | None:  # noqa: ARG002  -- reads the record only
        keys = self._keys or tuple(record.meta.keys())
        pairs = [(key, record.meta[key]) for key in keys if key in record.meta]
        if not pairs:
            return None
        body = "\n".join(f"  {key}: {_render_value(value)}" for key, value in pairs)
        return _section(self._heading, body)


def _render_value(value: Any) -> str:
    """A compact, readable rendering of a meta value for the prompt: a list becomes a comma-joined
    line, a mapping its ``key=value`` pairs, a scalar its string form."""
    if isinstance(value, Mapping):
        return ", ".join(f"{k}={v}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def register_builtins() -> None:
    """Register the built-in blocks. Idempotent, so importing this module more than once is
    harmless; the registry refuses a genuine name collision."""
    for name, block in (("literal", LiteralBlock), ("lexicon", LexiconBlock),
                        ("neighbours", NeighboursBlock), ("established", EstablishedBlock),
                        ("retrieved", RetrievedBlock),
                        ("previous_attempt", PreviousAttemptBlock), ("sql_rows", SqlRowsBlock),
                        ("schema", SchemaBlock), ("readings", ReadingsBlock)):
        CONTEXT_BLOCKS.register(name, block)


register_builtins()
