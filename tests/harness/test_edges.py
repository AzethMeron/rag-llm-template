"""Edge and branch coverage: repair paths, from_rules review, block config, config errors."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragkit.core.config import ConfigError
from ragkit.core.records import Record, Status
from ragkit.harness import (
    Harness,
    JsonFieldSchema,
    Panel,
    Persona,
    Progress,
    RuleSet,
    ValidatorPipeline,
    run_batch,
)
from ragkit.harness.context import ContextAssembler, ContextBlockError, LiteralBlock
from ragkit.harness.context.assembler import _PlacedBlock
from ragkit.harness.context.blocks import (
    CONTEXT_BLOCKS,
    EstablishedBlock,
    NeighboursBlock,
    RetrievedBlock,
    SqlRowsBlock,
)
from ragkit.harness.schemas import OUTPUT_SCHEMAS, FormField, FormSchema, JsonFieldSchema as JFS
from ragkit.harness.roles import load_panel
from ragkit.store.run.sqlite import SqliteRunStore

from .conftest import ACCEPT, build_harness, build_pool, ok, raw


def _rec(source: str = "hello", **meta: object) -> Record:
    return Record(record_id="1", source=source, meta=meta)


class TestRepairPaths:
    def test_envelope_recovered_then_ordinary_reply_verifies(self) -> None:
        harness = build_harness(produce=[raw('{"output": "partial'), ok({"output": "good"})],
                                review=[ACCEPT], max_repairs=2)
        outcome = harness.process(_rec())
        assert outcome.status is Status.VERIFIED and outcome.output == "good"

    def test_recovery_disabled_reasks_then_succeeds(self) -> None:
        harness = build_harness(produce=[raw('{"output": "partial'), ok({"output": "good"})],
                                review=[ACCEPT], max_repairs=2, repair_truncated_json=False)
        assert harness.process(_rec()).status is Status.VERIFIED

    def test_content_error_reasks_then_succeeds(self) -> None:
        harness = build_harness(produce=[raw("garbage"), ok({"output": "good"})],
                                review=[ACCEPT], max_repairs=2)
        assert harness.process(_rec()).status is Status.VERIFIED

    def test_style_directives_and_placeholder_contract_reach_the_prompt(self) -> None:
        # Both optional system-prompt sections are exercised: style directives, and the
        # placeholder contract (the input has a placeholder).
        ruleset = RuleSet(style_directives=("be concise",))
        harness = build_harness(produce=[ok({"output": "[[0]] done"})], review=[ACCEPT],
                                ruleset=ruleset)
        assert harness.process(_rec("[[0]] source")).status is Status.VERIFIED


class TestFromRulesReview:
    def test_a_from_rules_reviewer_judges_against_advisory(self) -> None:
        ruleset = RuleSet(advisory_rules=(("register", "keep it formal"),))
        panel = Panel(producer=Persona("p", "producer", "prod", instructions="do"),
                      reviewers=(Persona("c", "reviewer", "rev", from_rules=True),))
        pool = build_pool(produce=[ok({"output": "R"})], review=[ACCEPT])
        context = ContextAssembler([_PlacedBlock("literal", LiteralBlock("x"))])
        harness = Harness(pool, panel, ruleset, JsonFieldSchema(),
                          ValidatorPipeline(ruleset), context)
        assert harness.process(_rec()).status is Status.VERIFIED


class TestSchemas:
    def test_empty_field_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-empty field name"):
            JFS(field="  ")

    def test_schema_includes_description(self) -> None:
        schema = JFS(field="sql", description="a SELECT statement").json_schema()
        assert schema["properties"]["sql"]["description"] == "a SELECT statement"

    def test_extract_pulls_the_field(self) -> None:
        assert JFS(field="sql").extract({"sql": "SELECT 1"}) == "SELECT 1"

    def test_registered_and_buildable(self) -> None:
        schema = OUTPUT_SCHEMAS.create("json_field", {"field": "translation"})
        assert schema.name == "translation"


class TestFormSchema:
    def _schema(self) -> FormSchema:
        return FormSchema([FormField(name="genre", description="the musical genre"),
                           FormField(name="unit_price", type="number", required=False)])

    def test_json_schema_lists_fields_and_only_required(self) -> None:
        schema = self._schema().json_schema()
        assert set(schema["properties"]) == {"genre", "unit_price"}
        assert schema["properties"]["unit_price"]["type"] == "number"
        assert schema["required"] == ["genre"]  # unit_price is optional
        assert schema["properties"]["genre"]["description"] == "the musical genre"

    def test_extract_is_canonical_json_of_declared_fields_only(self) -> None:
        out = self._schema().extract({"genre": "Rock", "unit_price": 0.99, "extra": "ignored"})
        assert json.loads(out) == {"genre": "Rock", "unit_price": 0.99}
        assert out == '{"genre": "Rock", "unit_price": 0.99}'  # sorted keys, stable

    def test_missing_field_becomes_null_not_a_crash(self) -> None:
        out = self._schema().extract({"genre": "Jazz"})
        assert json.loads(out) == {"genre": "Jazz", "unit_price": None}

    def test_empty_fields_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one field"):
            FormSchema([])

    def test_duplicate_field_refused(self) -> None:
        with pytest.raises(ValueError, match="duplicate field"):
            FormSchema([FormField(name="a"), FormField(name="a")])

    def test_bad_field_type_refused(self) -> None:
        with pytest.raises(ValueError, match="type must be one of"):
            FormField(name="x", type="date")

    def test_empty_field_name_refused(self) -> None:
        with pytest.raises(ValueError, match="non-empty name"):
            FormField(name="  ")

    def test_from_config_builds_fields(self) -> None:
        schema = OUTPUT_SCHEMAS.create("form", {"fields": [
            {"name": "genre"}, {"name": "unit_price", "type": "number", "required": False}]})
        assert schema.name == "form"
        assert schema.json_schema()["required"] == ["genre"]

    def test_from_config_needs_fields(self) -> None:
        with pytest.raises(ConfigError, match="non-empty 'fields'"):
            OUTPUT_SCHEMAS.create("form", {})

    def test_from_config_field_needs_a_name(self) -> None:
        with pytest.raises(ConfigError, match="needs at least a 'name'"):
            OUTPUT_SCHEMAS.create("form", {"fields": [{"type": "string"}]})

    def test_from_config_refuses_a_mistyped_required(self) -> None:
        # required="false" must be refused, not silently coerced to bool("false")=True (which would
        # invert the constraint) -- from_config now reads through the strict core.config helpers.
        with pytest.raises(ConfigError, match="required"):
            OUTPUT_SCHEMAS.create("form", {"fields": [{"name": "x", "required": "false"}]})

    def test_array_field_is_a_string_array(self) -> None:
        schema = FormSchema([FormField(name="evidence", type="array")]).json_schema()
        assert schema["properties"]["evidence"] == {"type": "array", "items": {"type": "string"}}

    def test_array_field_extracts_the_list(self) -> None:
        out = FormSchema([FormField(name="tags", type="array")]).extract({"tags": ["a", "b"]})
        assert json.loads(out) == {"tags": ["a", "b"]}


class TestBlockConfig:
    def test_each_builtin_builds_via_from_config(self) -> None:
        for kind, options in (("literal", {"text": "x"}), ("lexicon", {}), ("neighbours", {}),
                             ("established", {}), ("retrieved", {}), ("previous_attempt", {}),
                             ("sql_rows", {"query": "select 1"}), ("schema", {}),
                             ("readings", {"keys": ["egt"]})):
            assert CONTEXT_BLOCKS.create(kind, options) is not None

    def test_retrieved_bad_k(self) -> None:
        with pytest.raises(ContextBlockError, match="k must be >= 1"):
            RetrievedBlock(k=0)

    def test_retrieved_bad_min_score(self) -> None:
        with pytest.raises(ContextBlockError, match="min_score must be in"):
            RetrievedBlock(min_score=2.0)

    def test_neighbours_negative(self) -> None:
        with pytest.raises(ContextBlockError, match="before/after must be >= 0"):
            NeighboursBlock(before=-1)

    def test_mistyped_option_is_refused_not_coerced(self) -> None:
        # from_config reads through the strict core.config helpers, so a mistyped option is refused
        # rather than silently coerced: true->1.0 (floor becomes exact-match), 12.5->12 (truncated),
        # a bare string->char-tuple (every key absent, block renders nothing).
        with pytest.raises(ConfigError, match="min_score"):
            CONTEXT_BLOCKS.create("retrieved", {"min_score": True})
        with pytest.raises(ConfigError, match="limit"):
            CONTEXT_BLOCKS.create("lexicon", {"limit": 12.5})
        with pytest.raises(ConfigError, match="param_keys"):
            CONTEXT_BLOCKS.create("sql_rows", {"query": "select 1", "param_keys": "customer_id"})

    def test_sql_rows_bad_limit(self) -> None:
        with pytest.raises(ContextBlockError, match="limit must be >= 1"):
            SqlRowsBlock(query="x", limit=0)

    def test_retrieved_wired_wrong_type(self) -> None:
        with pytest.raises(ContextBlockError, match="does not satisfy the Retriever"):
            RetrievedBlock().render(_rec(), {"retriever": "not a retriever"})

    def test_sql_rows_wired_wrong_type(self) -> None:
        with pytest.raises(ContextBlockError, match="does not satisfy the SqlStore"):
            SqlRowsBlock(query="x").render(_rec(), {"sql_store": object()})

    def test_sql_rows_missing_param_key_is_refused_not_bound_null(self) -> None:
        # A declared param_key absent from record.meta would bind SQL NULL (`= NULL` matches
        # nothing), silently rendering no rows -- indistinguishable from a genuine no-match. Refuse.
        class Store:
            read_only = True

            def query(self, sql: str, params: object = ()) -> list[dict]:  # pragma: no cover
                raise AssertionError("must not query with a missing bind")

            def execute(self, sql: str, params: object = ()) -> None:  # pragma: no cover
                raise AssertionError
        block = SqlRowsBlock(query="select * from t where id = ?", param_keys=("id",))
        with pytest.raises(ContextBlockError, match="missing param_key"):
            block.render(_rec(), {"sql_store": Store()})  # _rec() has no 'id' in meta

    def test_established_negative_window_is_refused(self) -> None:
        # Same guard its sibling NeighboursBlock has; a negative window would otherwise silently
        # drop the established context via the `limit <= 0` path.
        with pytest.raises(ContextBlockError, match="before/after must be >= 0"):
            EstablishedBlock(before=-1)


class TestAssemblerTrimBranches:
    def test_mixed_trimmable_and_untrimmable(self) -> None:
        blocks = [_PlacedBlock("literal", LiteralBlock("KEEP", heading="")),
                  _PlacedBlock("neighbours", NeighboursBlock(heading="N"))]
        record = _rec("m", context_before=["a long trimmable neighbour line"])
        assembler = ContextAssembler(blocks, max_chars=10, trim_order=["neighbours"])
        out = assembler.assemble(record, {})
        assert "KEEP" in out and "neighbour" not in out  # untrimmable literal survives


class TestConfigErrors:
    def _panel(self, tmp_path: Path, text: str) -> Path:
        path = tmp_path / "p.toml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_invalid_leniency_in_config(self, tmp_path: Path) -> None:
        text = ('[leniency]\nwindow = 0\n'
                '[[persona]]\nid="p"\nkind="producer"\nmodel="m"\ninstructions="x"\n'
                '[[persona]]\nid="r"\nkind="reviewer"\nmodel="m"\ninstructions="y"\n')
        with pytest.raises(ConfigError, match="window"):
            load_panel(self._panel(tmp_path, text))

    def test_invalid_limits_in_config(self, tmp_path: Path) -> None:
        text = ('[limits]\nproduce_tokens_floor = 5000\nproduce_tokens_ceiling = 10\n'
                '[[persona]]\nid="p"\nkind="producer"\nmodel="m"\ninstructions="x"\n'
                '[[persona]]\nid="r"\nkind="reviewer"\nmodel="m"\ninstructions="y"\n')
        with pytest.raises(ConfigError, match="exceeds"):
            load_panel(self._panel(tmp_path, text))

    def test_bad_revision_value(self, tmp_path: Path) -> None:
        text = ('[revision]\nmax_revisions = -1\n'
                '[[persona]]\nid="p"\nkind="producer"\nmodel="m"\ninstructions="x"\n'
                '[[persona]]\nid="r"\nkind="reviewer"\nmodel="m"\ninstructions="y"\n')
        with pytest.raises(ConfigError, match="max_revisions"):
            load_panel(self._panel(tmp_path, text))


class TestMoreBranches:
    def test_objection_without_issues_is_synthesised(self) -> None:
        harness = build_harness(
            produce=[ok({"output": "a"}), ok({"output": "b"})],
            review=[ok({"acceptable": False, "issues": []}), ACCEPT])
        outcome = harness.process(_rec())
        assert outcome.status is Status.VERIFIED and outcome.rounds == 2

    def test_forbidden_pattern_that_does_not_match_is_clean(self) -> None:
        ruleset = RuleSet(forbidden_patterns=(("^here is", "preamble"),))
        harness = build_harness(produce=[ok({"output": "a clean output"})], review=[ACCEPT],
                                ruleset=ruleset)
        assert harness.process(_rec()).status is Status.VERIFIED

    def test_retrieved_empty_when_no_hits(self) -> None:
        from ragkit.core.ports import Retrieved

        class Empty:
            def retrieve(self, query: str, *, k: int,
                         min_score: float = 0.0) -> tuple[Retrieved, ...]:
                return ()

        assert RetrievedBlock().render(_rec(), {"retriever": Empty()}) is None

    def test_sql_rows_empty_when_no_rows(self) -> None:
        class Empty:
            read_only = True

            def query(self, sql: str, params: object = ()) -> list[dict]:
                return []

            def execute(self, sql: str, params: object = ()) -> None:  # pragma: no cover
                raise AssertionError

        assert SqlRowsBlock(query="x").render(_rec(), {"sql_store": Empty()}) is None

    def test_established_none_when_no_neighbour_is_in_memory(self) -> None:
        from ragkit.harness.context.blocks import EstablishedBlock
        from ragkit.harness.memory import OutputMemory
        record = _rec("m", context_before=["unseen line"])
        assert EstablishedBlock().render(record, {"memory": OutputMemory()}) is None

    def test_assembler_length_of_all_empty(self) -> None:
        # A block that renders nothing, under an active budget: the fit loop measures an empty
        # result (length 0) and stops.
        class NoneBlock:
            def render(self, record: Record, context: object) -> str | None:
                return None

        assembler = ContextAssembler([_PlacedBlock("x", NoneBlock())], max_chars=100)  # type: ignore[list-item]
        assert assembler.assemble(_rec(), {}) == ""


class TestPreviousAttemptNoIssues:
    def test_renders_bare_attempt_without_issues_or_suggestions(self) -> None:
        from ragkit.harness.agents import Attempt
        from ragkit.harness.context.blocks import PreviousAttemptBlock
        attempt = Attempt(target="just the draft", issues=(), suggestions=())
        out = PreviousAttemptBlock().render(_rec(), {"previous_attempt": attempt})
        assert out is not None and "just the draft" in out
        assert "sent back" not in out and "rewrite" not in out


class TestAssemblerPriorityTie:
    def test_lower_priority_trimmable_block_is_not_chosen_over_higher(self) -> None:
        # Two trimmable kinds of different priority under a tight budget: the higher-priority
        # 'a' is trimmed first, exercising the "not a better victim" branch for 'b'.
        blocks = [_PlacedBlock("a", LiteralBlock("AAAAAAAAAA", heading="")),
                  _PlacedBlock("b", LiteralBlock("BBBBBBBBBB", heading=""))]
        assembler = ContextAssembler(blocks, max_chars=12, trim_order=["a", "b"])
        out = assembler.assemble(_rec(), {})
        # 'a' (highest trim priority) goes first; 'b' remains within budget.
        assert "AAAA" not in out and "BBBB" in out


class TestMoreConfigErrors:
    def test_non_string_instructions(self, tmp_path: Path) -> None:
        text = ('[[persona]]\nid="p"\nkind="producer"\nmodel="m"\ninstructions=5\n'
                '[[persona]]\nid="r"\nkind="reviewer"\nmodel="m"\ninstructions="y"\n')
        path = tmp_path / "p.toml"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ConfigError, match="instructions must be a string"):
            load_panel(path)

    def test_zero_max_tokens(self, tmp_path: Path) -> None:
        text = ('[[persona]]\nid="p"\nkind="producer"\nmodel="m"\ninstructions="x"\n'
                '[[persona]]\nid="r"\nkind="reviewer"\nmodel="m"\ninstructions="y"\n'
                'max_tokens = 0\n')
        path = tmp_path / "p.toml"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ConfigError, match="max_tokens"):
            load_panel(path)


class TestRunnerExtras:
    def test_elapsed_uses_the_clock(self) -> None:
        ticks = iter([100.0, 175.0])
        progress = Progress(total=1, started_at=next(ticks), _clock=lambda: next(ticks))
        assert progress.elapsed == 75.0

    def test_signal_handlers_install_and_restore(self) -> None:
        # Runs on the main thread, so the real _Interruptible installs and restores handlers.
        harness = build_harness(produce=[ok({"output": "R"})], review=[ACCEPT])
        records = [Record(record_id="1", source="s")]
        store = SqliteRunStore()
        store.add_records(records)
        progress = run_batch(harness, records, store, install_signal_handlers=True)
        assert progress.done == 1

    def test_interrupt_handler_sets_the_stop_flag(self) -> None:
        from ragkit.harness.runner import _Interruptible
        interrupt = _Interruptible()
        assert not interrupt.stop
        interrupt._handle()  # what a SIGINT would call
        assert interrupt.stop

    def test_fewer_records_than_concurrency(self) -> None:
        # One record, four workers: the initial submit loop runs out of work and breaks early.
        harness = build_harness(produce=[ok({"output": "R"})], review=[ACCEPT])
        records = [Record(record_id="1", source="s")]
        store = SqliteRunStore()
        store.add_records(records)
        progress = run_batch(harness, records, store, concurrency=4,
                             install_signal_handlers=False)
        assert progress.done == 1
