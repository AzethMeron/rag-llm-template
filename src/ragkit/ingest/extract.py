"""Extractors: turn a raw source into :class:`~ragkit.core.ports.Document` objects ready to chunk.

Shipped are the light, dependency-free extractors the three recipes need — plain text, JSON Lines,
SQL rows, and structure-aware HTML/Markdown (via the standard library's ``html.parser``, no binary
parsers). PDF/DOCX/email extraction is documented as a plugin: register a component under the
``EXTRACTOR`` port and select it by dotted path, no framework change.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from html.parser import HTMLParser
from typing import Any

from ragkit.core.errors import RagkitError
from ragkit.core.ports import Document, Extractor
from ragkit.core.registry import Registry

_EMPTY: Mapping[str, Any] = {}

EXTRACTORS: Registry[Extractor] = Registry(
    "extractor", Extractor,  # type: ignore[type-abstract]
    entry_point_group="ragkit.extractors")


class ExtractError(RagkitError):
    """A source could not be extracted."""


def _as_text(source: bytes | str) -> str:
    return source.decode("utf-8") if isinstance(source, bytes) else source


class TextExtractor:
    """The whole source as one plain-text document."""

    CONFIG_KEYS = frozenset({"doc_id"})

    def __init__(self, doc_id: str = "text") -> None:
        self._doc_id = doc_id

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> TextExtractor:
        return cls(doc_id=str(options.get("doc_id", "text")))

    def extract(self, source: bytes | str, *,
                meta: Mapping[str, Any] = _EMPTY) -> Iterator[Document]:
        yield Document(doc_id=str(meta.get("doc_id", self._doc_id)), text=_as_text(source),
                       meta=dict(meta))


class JsonlExtractor:
    """One document per JSON Lines record, taking the text from a named field."""

    CONFIG_KEYS = frozenset({"field", "id_field"})

    def __init__(self, field: str = "text", id_field: str = "") -> None:
        self._field = field
        self._id_field = id_field

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> JsonlExtractor:
        return cls(field=str(options.get("field", "text")),
                   id_field=str(options.get("id_field", "")))

    def extract(self, source: bytes | str, *,
                meta: Mapping[str, Any] = _EMPTY) -> Iterator[Document]:  # noqa: ARG002
        import json
        for line_no, line in enumerate(_as_text(source).splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExtractError(f"line {line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict) or self._field not in record:
                raise ExtractError(f"line {line_no}: record has no {self._field!r} field")
            doc_id = str(record[self._id_field]) if self._id_field else f"line-{line_no}"
            yield Document(doc_id=doc_id, text=str(record[self._field]),
                           meta={k: v for k, v in record.items() if k != self._field})


class _HtmlText(HTMLParser):
    """Collect visible text and the heading trail, dropping script/style content."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.headings: list[str] = []
        self._skip = 0
        self._in_heading = False

    def handle_starttag(self, tag: str, attrs: object) -> None:  # noqa: ARG002
        if tag in ("script", "style"):
            self._skip += 1
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._in_heading = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._in_heading = False

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        text = data.strip()
        if not text:
            return
        self.parts.append(text)
        if self._in_heading:
            self.headings.append(text)


class HtmlExtractor:
    """Strip HTML to text (via stdlib ``html.parser``), recording the heading trail in metadata."""

    CONFIG_KEYS = frozenset({"doc_id"})

    def __init__(self, doc_id: str = "html") -> None:
        self._doc_id = doc_id

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> HtmlExtractor:
        return cls(doc_id=str(options.get("doc_id", "html")))

    def extract(self, source: bytes | str, *,
                meta: Mapping[str, Any] = _EMPTY) -> Iterator[Document]:
        parser = _HtmlText()
        parser.feed(_as_text(source))
        combined = dict(meta)
        if parser.headings:
            combined["headings"] = parser.headings
        yield Document(doc_id=str(meta.get("doc_id", self._doc_id)),
                       text="\n\n".join(parser.parts), meta=combined)


class MarkdownExtractor:
    """Split Markdown into a document per top-level ATX (``#``/``##``) section, carrying the
    heading in ``section_path`` — so a structure-aware chunker keeps sections intact."""

    CONFIG_KEYS = frozenset({"doc_id"})

    def __init__(self, doc_id: str = "md") -> None:
        self._doc_id = doc_id

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> MarkdownExtractor:
        return cls(doc_id=str(options.get("doc_id", "md")))

    def extract(self, source: bytes | str, *,
                meta: Mapping[str, Any] = _EMPTY) -> Iterator[Document]:
        base = str(meta.get("doc_id", self._doc_id))
        sections: list[tuple[str, list[str]]] = [("", [])]
        for line in _as_text(source).splitlines():
            if line.startswith("#"):
                heading = line.lstrip("#").strip()
                sections.append((heading, []))
            else:
                sections[-1][1].append(line)
        emitted = 0
        for heading, body_lines in sections:
            body = "\n".join(body_lines).strip()
            if not body and not heading:
                continue
            # body is already stripped and non-empty-or-heading-present here, so text is non-empty.
            text = f"{heading}\n\n{body}" if heading and body else (heading or body)
            combined = dict(meta)
            if heading:
                combined["section_path"] = heading
            yield Document(doc_id=f"{base}-{emitted}", text=text, meta=combined)
            emitted += 1


EXTRACTORS.register("text", TextExtractor)
EXTRACTORS.register("jsonl", JsonlExtractor)
EXTRACTORS.register("html", HtmlExtractor)
EXTRACTORS.register("markdown", MarkdownExtractor)
