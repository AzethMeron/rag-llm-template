"""Context blocks, the budgeting assembler, and the config loader."""
from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.core.lexicon import Entry
from ragkit.core.ports import Retrieved
from ragkit.core.records import Record
from ragkit.harness.agents import Attempt
from ragkit.harness.context import (
    ContextAssembler,
    ContextBlockError,
    EstablishedBlock,
    LexiconBlock,
    LiteralBlock,
    NeighboursBlock,
    PreviousAttemptBlock,
    RetrievedBlock,
    SqlRowsBlock,
    load_context,
)
from ragkit.harness.context.assembler import _PlacedBlock
from ragkit.harness.memory import OutputMemory


def _rec(source: str = "current line", **meta: object) -> Record:
    return Record(record_id="1", source=source, meta=meta)


class _StubRetriever:
    def __init__(self, hits: list[Retrieved]) -> None:
        self._hits = hits

    def retrieve(self, query: str, *, k: int, min_score: float = 0.0) -> tuple[Retrieved, ...]:
        return tuple(self._hits[:k])


class _StubStore:
    read_only = True

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def query(self, sql: str, params: object = ()) -> list[dict]:
        return self._rows

    def execute(self, sql: str, params: object = ()) -> None:  # pragma: no cover - unused
        raise AssertionError("read-only")


class TestBlocks:
    def test_literal(self) -> None:
        assert LiteralBlock("Do it.", heading="Note:").render(_rec(), {}) == "Note:\nDo it."

    def test_literal_needs_text(self) -> None:
        with pytest.raises(ContextBlockError, match="non-empty 'text'"):
            LiteralBlock("   ")

    def test_lexicon_renders_matching_terms(self) -> None:
        block = LexiconBlock()
        out = block.render(_rec("the cat sat"), {"lexicon": [Entry(term="cat", rendering="kot")]})
        assert out is not None and "cat -> kot" in out

    def test_lexicon_empty_when_no_match(self) -> None:
        lexicon = [Entry(term="cat", rendering="kot")]
        assert LexiconBlock().render(_rec("dog"), {"lexicon": lexicon}) is None

    def test_neighbours_as_continuous_passage(self) -> None:
        record = _rec("middle", context_before=["a", "b"], context_after=["c"])
        out = NeighboursBlock(before=3, after=3, heading="Context:").render(record, {})
        assert out is not None and "a\nb\nmiddle\nc" in out

    def test_neighbours_masks_placeholders(self) -> None:
        record = _rec("m", context_before=["[[0]] said"])
        out = NeighboursBlock().render(record, {"stand_in": "NAME"})
        assert out is not None and "NAME said" in out and "[[0]]" not in out

    def test_neighbours_empty_without_context(self) -> None:
        assert NeighboursBlock().render(_rec(), {}) is None

    def test_established_pairs_with_memory(self) -> None:
        memory = OutputMemory({"prev line": "PREV"})
        record = _rec("m", context_before=["prev line"])
        out = EstablishedBlock().render(record, {"memory": memory})
        assert out is not None and "prev line -> PREV" in out

    def test_established_none_without_memory(self) -> None:
        assert EstablishedBlock().render(_rec("m", context_before=["x"]), {}) is None

    def test_retrieved_renders_hits(self) -> None:
        retriever = _StubRetriever([Retrieved("c1", "an example", 0.9)])
        out = RetrievedBlock().render(_rec(), {"retriever": retriever})
        assert out is not None and "an example" in out

    def test_retrieved_without_retriever_raises(self) -> None:
        with pytest.raises(ContextBlockError, match="no retriever was wired"):
            RetrievedBlock().render(_rec(), {})

    def test_previous_attempt_only_on_revision(self) -> None:
        assert PreviousAttemptBlock().render(_rec(), {}) is None
        attempt = Attempt(target="draft", issues=("too long",), suggestions=("shorten",))
        out = PreviousAttemptBlock().render(_rec(), {"previous_attempt": attempt})
        assert out is not None and "draft" in out and "too long" in out and "shorten" in out

    def test_sql_rows_renders_rows(self) -> None:
        store = _StubStore([{"name": "Ann", "total": 5}])
        block = SqlRowsBlock(query="select ...", param_keys=("customer_id",))
        out = block.render(_rec(customer_id=7), {"sql_store": store})
        assert out is not None and "name='Ann'" in out

    def test_sql_rows_without_store_raises(self) -> None:
        with pytest.raises(ContextBlockError, match="no sql_store was wired"):
            SqlRowsBlock(query="x").render(_rec(), {})

    def test_sql_rows_needs_a_query(self) -> None:
        with pytest.raises(ContextBlockError, match="non-empty 'query'"):
            SqlRowsBlock(query="  ")


class TestAssembler:
    def _assembler(self, **kwargs: object) -> ContextAssembler:
        blocks = [
            _PlacedBlock("literal", LiteralBlock("keep me", heading="A")),
            _PlacedBlock("neighbours", NeighboursBlock(heading="B")),
        ]
        return ContextAssembler(blocks, **kwargs)  # type: ignore[arg-type]

    def test_drops_empty_blocks(self) -> None:
        # The neighbours block has no content (no context_before/after), so it is omitted.
        out = self._assembler().assemble(_rec(), {})
        assert "keep me" in out and "B" not in out

    def test_joins_present_blocks(self) -> None:
        record = _rec("m", context_before=["x"])
        out = self._assembler().assemble(record, {})
        assert "keep me" in out and "x" in out

    def test_budget_trims_by_priority(self) -> None:
        record = _rec("m", context_before=["a very long neighbour line here"])
        # Tight budget with neighbours marked trimmable: the literal (not trimmable) survives.
        out = self._assembler(max_chars=20, trim_order=["neighbours"]).assemble(record, {})
        assert "keep me" in out and "neighbour line" not in out

    def test_untrimmable_blocks_survive_any_budget(self) -> None:
        out = self._assembler(max_chars=1).assemble(_rec(), {})
        assert "keep me" in out  # nothing in trim_order, so nothing is dropped


class TestLoadContext:
    def _write(self, tmp_path: Path, text: str) -> Path:
        path = tmp_path / "context.toml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_loads_blocks_in_order(self, tmp_path: Path) -> None:
        text = """
[[context.block]]
kind = "literal"
text = "Ground your answer."

[[context.block]]
kind = "neighbours"
before = 2

[context.budget]
max_chars = 5000
trim_order = ["neighbours"]
"""
        assembler = load_context(self._write(tmp_path, text))
        out = assembler.assemble(_rec(), {})
        assert "Ground your answer." in out

    def test_missing_kind(self, tmp_path: Path) -> None:
        from ragkit.core.config import ConfigError
        with pytest.raises(ConfigError, match="needs a 'kind'"):
            load_context(self._write(tmp_path, "[[context.block]]\ntext = 'x'\n"))

    def test_no_blocks(self, tmp_path: Path) -> None:
        from ragkit.core.config import ConfigError
        with pytest.raises(ConfigError, match=r"no \[\[context.block\]\]"):
            load_context(self._write(tmp_path, "[context.budget]\nmax_chars = 10\n"))

    def test_unknown_block_option_is_refused(self, tmp_path: Path) -> None:
        from ragkit.core.config import ConfigError
        text = '[[context.block]]\nkind = "literal"\ntext = "x"\nbogus = 1\n'
        with pytest.raises(ConfigError, match="unknown key"):
            load_context(self._write(tmp_path, text))

    def test_negative_budget_is_refused(self, tmp_path: Path) -> None:
        from ragkit.core.config import ConfigError
        text = ('[[context.block]]\nkind = "literal"\ntext = "x"\n'
                '[context.budget]\nmax_chars = -1\n')
        with pytest.raises(ConfigError, match="max_chars must be >= 0"):
            load_context(self._write(tmp_path, text))
