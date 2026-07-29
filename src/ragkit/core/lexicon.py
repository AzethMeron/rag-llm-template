"""Domain terminology, as a term-to-rendering mapping retrieved by literal occurrence.

The generalisation of a translation glossary: a body of established terms and the exact form
each should take in the output — a house translation of a name, a canonical spelling, a
required SQL identifier for a business concept, a fixed label for a form field. What a source
*derives* one from differs completely between tasks; what a producer needs from it never does.

Retrieval is deliberately simple and deterministic (:func:`relevant_entries`): the entries
whose term literally occurs in the input, longest first, capped. A component decides what to
do with them — usually inject them into the prompt as "use these renderings", and, for the
ones that must appear verbatim, enforce their presence with a mechanical validator. The
``category`` field is free-form; the core never interprets it, so a recipe can tag entries
however its own validators and prompt blocks require.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .errors import RagkitError


class LexiconError(RagkitError):
    """A lexicon source could not be read or parsed."""

    def __init__(self, reason: str, *, path: Path | None = None,
                 line_no: int | None = None) -> None:
        location = f"{path}:{line_no}" if path and line_no else (str(path) if path else None)
        super().__init__(reason, location=location)
        self.path = path
        self.line_no = line_no


@dataclass(frozen=True, slots=True)
class Entry:
    """One term and its established rendering.

    ``category`` is free-form and task-defined; the core never reads it, so a recipe uses it
    to route entries to its own validators or prompt sections (for example, marking the ones
    that must appear verbatim in the output). ``entity_id`` is a caller-side handle back to
    whatever the entry came from; the framework never interprets it.
    """

    term: str
    rendering: str
    category: str = ""
    entity_id: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def write_lexicon(entries: list[Entry], path: Path) -> int:
    """Write entries as JSON Lines, longest term first. Returns the count written.

    Longest-first is a convenience for a human reading the file; :func:`relevant_entries`
    re-sorts on read, so callers do not depend on the order here.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in sorted(entries, key=lambda e: (-len(e.term), e.term)):
            handle.write(entry.to_json())
            handle.write("\n")
    return len(entries)


def read_lexicon(path: Path) -> list[Entry]:
    """Load established terminology, or none if the file does not exist.

    A missing lexicon means "no established terms", which is a legitimate and common state:
    terminology is something a project accumulates, not something every task arrives with.
    Raising here would force every caller without one to fabricate an empty file — a
    workaround standing in for a supported case.

    A lexicon that exists and is malformed is still an error: that is a corrupt file, not an
    absent one, and silently treating it as empty would drop terminology the caller believes
    is in force.
    """
    if not path.exists():
        return []
    if not path.is_file():
        raise LexiconError("lexicon path is not a regular file", path=path)
    entries: list[Entry] = []
    for line_no, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LexiconError(f"invalid JSON: {exc}", path=path, line_no=line_no) from exc
        if not isinstance(record, dict) or "term" not in record or "rendering" not in record:
            raise LexiconError(
                "a lexicon entry needs at least 'term' and 'rendering'",
                path=path, line_no=line_no)
        try:
            entries.append(Entry(**record))
        except TypeError as exc:
            raise LexiconError(f"{exc}", path=path, line_no=line_no) from exc
    return entries


def relevant_entries(source_text: str, entries: list[Entry], limit: int = 24) -> list[Entry]:
    """Lexicon entries whose term literally occurs in ``source_text``, longest first, capped.

    Longest-first so that a compound term wins over its constituents.

    Substring matching is a deliberate over-match. In an inflected language a lemma does not
    equal its declined form, so requiring a whole-word hit would miss most real occurrences.
    The asymmetry is what settles it: over-matching shows the model a term it did not need,
    while under-matching silently drops one it did.
    """
    hits = [entry for entry in entries if entry.term and entry.term in source_text]
    hits.sort(key=lambda e: -len(e.term))
    return hits[:limit]
