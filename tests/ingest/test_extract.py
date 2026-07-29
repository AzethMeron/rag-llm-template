"""The extractors."""
from __future__ import annotations

import pytest

from ragkit.ingest.extract import (
    EXTRACTORS,
    ExtractError,
    HtmlExtractor,
    JsonlExtractor,
    MarkdownExtractor,
    TextExtractor,
)


class TestText:
    def test_whole_source(self) -> None:
        [doc] = list(TextExtractor().extract("hello world"))
        assert doc.text == "hello world" and doc.doc_id == "text"

    def test_bytes_input(self) -> None:
        [doc] = list(TextExtractor().extract(b"bytes here"))
        assert doc.text == "bytes here"

    def test_doc_id_from_meta(self) -> None:
        [doc] = list(TextExtractor().extract("x", meta={"doc_id": "custom"}))
        assert doc.doc_id == "custom"


class TestJsonl:
    def test_one_document_per_record(self) -> None:
        source = '{"text": "first", "lang": "en"}\n{"text": "second"}\n'
        docs = list(JsonlExtractor().extract(source))
        assert [d.text for d in docs] == ["first", "second"]
        assert docs[0].meta["lang"] == "en"

    def test_id_field(self) -> None:
        [doc] = list(JsonlExtractor(id_field="k").extract('{"text": "t", "k": "id7"}'))
        assert doc.doc_id == "id7"

    def test_blank_lines_skipped(self) -> None:
        assert len(list(JsonlExtractor().extract('\n{"text": "x"}\n\n'))) == 1

    def test_invalid_json(self) -> None:
        with pytest.raises(ExtractError, match="invalid JSON"):
            list(JsonlExtractor().extract("{not json"))

    def test_missing_field(self) -> None:
        with pytest.raises(ExtractError, match="no 'text' field"):
            list(JsonlExtractor().extract('{"other": 1}'))


class TestHtml:
    def test_strips_tags_and_captures_headings(self) -> None:
        html = "<h1>Title</h1><p>Body <b>text</b></p><script>ignore()</script>"
        [doc] = list(HtmlExtractor().extract(html))
        assert "Title" in doc.text and "Body" in doc.text and "ignore" not in doc.text
        assert doc.meta["headings"] == ["Title"]


class TestMarkdown:
    def test_splits_by_heading(self) -> None:
        md = "# First\n\ncontent one\n\n## Second\n\ncontent two"
        docs = list(MarkdownExtractor().extract(md))
        assert len(docs) == 2
        assert docs[0].meta["section_path"] == "First"
        assert "content two" in docs[1].text

    def test_preamble_without_heading(self) -> None:
        docs = list(MarkdownExtractor().extract("just text, no headings"))
        assert len(docs) == 1 and "just text" in docs[0].text


def test_builtins_registered() -> None:
    assert set(EXTRACTORS.available()) >= {"text", "jsonl", "html", "markdown"}
    assert EXTRACTORS.create("text", {"doc_id": "d"}) is not None
