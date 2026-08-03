"""Extractors: turn a raw source into :class:`~ragkit.core.ports.Document` objects ready to chunk.

Shipped are the light, dependency-free extractors the recipes need — plain text, JSON Lines, and
structure-aware HTML/Markdown (via the standard library's ``html.parser``, no binary parsers).
PDF/DOCX/email extraction is documented as a plugin: register a component under the ``EXTRACTOR``
port and select it by dotted path, no framework change.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from html.parser import HTMLParser
from typing import Any

from ragkit.core.config import read_string
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
        return cls(doc_id=read_string(dict(options), "doc_id", "text", label="text extractor"))

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
        opts = dict(options)
        return cls(field=read_string(opts, "field", "text", label="jsonl extractor"),
                   id_field=read_string(opts, "id_field", "", label="jsonl extractor"))

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
            text = record[self._field]
            # A document needs real text: a null/non-string field used to be str()'d into "None"
            # (or "42", "['a']") and indexed as content. Refuse it, naming the line.
            if not isinstance(text, str):
                raise ExtractError(f"line {line_no}: {self._field!r} must be a string, "
                                   f"got {type(text).__name__}")
            if self._id_field:
                if self._id_field not in record:
                    # A structured error like every other failure in this module -- this used to
                    # escape as a bare KeyError naming only the missing key.
                    raise ExtractError(f"line {line_no}: record has no id field "
                                       f"{self._id_field!r}")
                if record[self._id_field] is None:
                    raise ExtractError(f"line {line_no}: id field {self._id_field!r} is null")
                doc_id = str(record[self._id_field])
            else:
                doc_id = f"line-{line_no}"
            yield Document(doc_id=doc_id, text=text,
                           meta={k: v for k, v in record.items() if k != self._field})


_HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

_BLOCK_TAGS = frozenset({
    *_HEADINGS, "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt",
    "fieldset", "figcaption", "figure", "footer", "form", "header", "hr", "li", "main", "nav",
    "ol", "p", "pre", "section", "table", "td", "th", "tr", "ul",
})
"""Tags that end a paragraph. Everything else (``b``, ``em``, ``a``, ``span``, ``code``, ...) is
inline and must *not*, or ``<p>The <b>quick</b> brown fox</p>`` becomes three paragraphs."""


class _HtmlText(HTMLParser):
    """Collect visible text and the heading trail, dropping script/style content.

    Text is accumulated into the current paragraph and flushed only at a block-level boundary.
    Every text node used to be stripped and emitted separately, then joined with ``\\n\\n``, so
    inline markup shredded a sentence into one "paragraph" per fragment *and* lost the spaces
    between them (``The``, ``quick``, ``brown fox``) — which then defeated the structure chunker
    downstream, since it splits on exactly those blank lines.
    """

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.headings: list[str] = []
        self._skip = 0
        self._in_heading = False
        self._current: list[str] = []

    def handle_starttag(self, tag: str, attrs: object) -> None:  # noqa: ARG002
        if tag in ("script", "style"):
            self._skip += 1
        if tag in _BLOCK_TAGS:
            self._flush()
        if tag in _HEADINGS:
            self._in_heading = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        if tag in _HEADINGS and self._in_heading:
            self.headings.append(" ".join(self._current).strip())
            self._in_heading = False
        if tag in _BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        text = data.strip()
        if text:
            self._current.append(text)

    def _flush(self) -> None:
        """End the current paragraph. Inline runs are joined with a single space, so the word
        boundary HTML expressed with markup survives into the text."""
        joined = " ".join(self._current).strip()
        self._current = []
        if joined:
            self.parts.append(joined)

    def close(self) -> None:
        super().close()
        self._flush()  # trailing text with no closing block tag


class HtmlExtractor:
    """Strip HTML to text (via stdlib ``html.parser``), recording the heading trail in metadata."""

    CONFIG_KEYS = frozenset({"doc_id"})

    def __init__(self, doc_id: str = "html") -> None:
        self._doc_id = doc_id

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> HtmlExtractor:
        return cls(doc_id=read_string(dict(options), "doc_id", "html", label="html extractor"))

    def extract(self, source: bytes | str, *,
                meta: Mapping[str, Any] = _EMPTY) -> Iterator[Document]:
        parser = _HtmlText()
        parser.feed(_as_text(source))
        parser.close()  # flushes any trailing paragraph that no closing tag ended
        combined = dict(meta)
        if parser.headings:
            combined["headings"] = parser.headings
        yield Document(doc_id=str(meta.get("doc_id", self._doc_id)),
                       text="\n\n".join(parser.parts), meta=combined)


class MarkdownExtractor:
    """Split Markdown into a document per ATX (``#``-prefixed) section, carrying the heading in
    ``section_path`` — so a structure-aware chunker keeps sections intact.

    **Any** ``#``-leading line starts a new section, at any depth, including one inside a fenced
    code block. The docstring used to say "top-level (``#``/``##``)", which described neither.
    Splitting at every depth is the useful behaviour for retrieval — a deep subsection is still a
    self-contained passage — so the behaviour stands and the description is corrected; a corpus
    with ``#`` comments in fenced code needs a different extractor.
    """

    CONFIG_KEYS = frozenset({"doc_id"})

    def __init__(self, doc_id: str = "md") -> None:
        self._doc_id = doc_id

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> MarkdownExtractor:
        return cls(doc_id=read_string(dict(options), "doc_id", "md", label="markdown extractor"))

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
