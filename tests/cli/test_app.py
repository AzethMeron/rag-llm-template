"""Config-driven assembly, and the checks it makes before any server is contacted."""
from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.core.config import ConfigError
from ragkit.core.records import Record, Status, read_journal, write_catalog
from ragkit.harness import pending_records, run_batch
from ragkit.llm.pool import ModelPoolError
from ragkit.cli.app import assemble

from .conftest import scripted_factory, write_config


class TestAssembleAndRun:
    def test_end_to_end_run(self, tmp_path: Path) -> None:
        config = write_config(tmp_path / "cfg")
        assembled = assemble(config, client_factory=scripted_factory())
        catalog, journal = tmp_path / "c.jsonl", tmp_path / "j.jsonl"
        write_catalog([Record(record_id="1", source="hello")], catalog)
        progress = run_batch(assembled.harness, pending_records(catalog, journal), journal,
                             install_signal_handlers=False)
        assert progress.verified == 1
        [result] = list(read_journal(journal))
        assert result.output == "RESULT" and result.status is Status.VERIFIED

    def test_reference_retriever_is_wired(self, tmp_path: Path) -> None:
        recipe = ('[task]\noutput_schema = "json_field"\n'
                  '[reference]\nfile = "ref.jsonl"\n')
        config = write_config(tmp_path / "cfg", recipe=recipe,
                              reference=[{"source": "the cat", "target": "kot"}])
        assembled = assemble(config, client_factory=scripted_factory())
        from ragkit.retrieve.retrievers import LexicalRetriever
        assert isinstance(assembled.retriever, LexicalRetriever)
        hits = assembled.retriever.retrieve("cat", k=1)
        assert hits and hits[0].text == "the cat -> kot"

    def test_substitutions_fill_persona_instructions(self, tmp_path: Path) -> None:
        personas = ('[[persona]]\nid="p"\nkind="producer"\nmodel="prod"\n'
                    'instructions="translate to {target_language}"\n'
                    '[[persona]]\nid="r"\nkind="reviewer"\nmodel="rev"\ninstructions="review"\n')
        config = write_config(tmp_path / "cfg", personas=personas)
        assembled = assemble(config, substitutions={"target_language": "Polish"},
                             client_factory=scripted_factory())
        assert "Polish" in assembled.harness.panel.producer.instructions


class TestChecksBeforeServer:
    def test_thrash_guard_refuses_too_many_models(self, tmp_path: Path) -> None:
        models = ('[endpoint.local]\nresident_max = 1\n'
                  '[model.prod]\nendpoint="local"\nmodel_id="a"\n'
                  '[model.rev]\nendpoint="local"\nmodel_id="b"\n')
        config = write_config(tmp_path / "cfg", models=models)
        with pytest.raises(ModelPoolError, match="would hold 2 distinct models"):
            assemble(config, client_factory=scripted_factory())

    def test_from_rules_reviewer_without_advisory_is_refused(self, tmp_path: Path) -> None:
        personas = ('[[persona]]\nid="p"\nkind="producer"\nmodel="prod"\ninstructions="do"\n'
                    '[[persona]]\nid="c"\nkind="reviewer"\nmodel="rev"\nfrom_rules=true\n')
        config = write_config(tmp_path / "cfg", personas=personas)
        with pytest.raises(ValueError, match="no \\[\\[advisory\\]\\]"):
            assemble(config, client_factory=scripted_factory())


class TestConfigErrors:
    def test_missing_recipe(self, tmp_path: Path) -> None:
        config = write_config(tmp_path / "cfg")
        (config / "recipe.toml").unlink()
        with pytest.raises(ConfigError, match="recipe file not found"):
            assemble(config, client_factory=scripted_factory())

    def test_validator_without_kind(self, tmp_path: Path) -> None:
        recipe = '[task]\noutput_schema="json_field"\n[[validator]]\noptions = 1\n'
        config = write_config(tmp_path / "cfg", recipe=recipe)
        with pytest.raises(ConfigError, match="needs a 'kind'"):
            assemble(config, client_factory=scripted_factory())

    def test_missing_reference_file(self, tmp_path: Path) -> None:
        recipe = '[task]\noutput_schema="json_field"\n[reference]\nfile="absent.jsonl"\n'
        config = write_config(tmp_path / "cfg", recipe=recipe)
        with pytest.raises(ConfigError, match="reference corpus not found"):
            assemble(config, client_factory=scripted_factory())

    def test_dense_reference_needs_wiring(self, tmp_path: Path) -> None:
        recipe = ('[task]\noutput_schema="json_field"\n'
                  '[reference]\nfile="ref.jsonl"\nretriever="dense"\n')
        config = write_config(tmp_path / "cfg", recipe=recipe,
                              reference=[{"source": "a", "target": "b"}])
        with pytest.raises(ConfigError, match="needs an embedding endpoint"):
            assemble(config, client_factory=scripted_factory())

    def test_unknown_recipe_section(self, tmp_path: Path) -> None:
        config = write_config(tmp_path / "cfg", recipe='[bogus]\nx = 1\n')
        with pytest.raises(ConfigError, match="unknown key"):
            assemble(config, client_factory=scripted_factory())


class TestReferenceCorpus:
    def _recipe(self, **extra: str) -> str:
        opts = "".join(f'{k} = "{v}"\n' for k, v in extra.items())
        return f'[task]\noutput_schema="json_field"\n[reference]\nfile="ref.jsonl"\n{opts}'

    def test_blank_lines_and_missing_index_field_are_skipped(self, tmp_path: Path) -> None:
        config = write_config(tmp_path / "cfg", recipe=self._recipe(),
                              reference=[{"source": "the cat", "target": "kot"}, {"target": "x"}])
        # append a blank line to the reference file
        ref = config / "ref.jsonl"
        ref.write_text(ref.read_text() + "\n\n", encoding="utf-8")
        assembled = assemble(config, client_factory=scripted_factory())
        assert assembled.retriever is not None
        assert len(assembled.retriever.retrieve("cat", k=5)) == 1  # only the one with a source

    def test_invalid_json_in_reference_is_reported(self, tmp_path: Path) -> None:
        config = write_config(tmp_path / "cfg", recipe=self._recipe(),
                              reference=[{"source": "a", "target": "b"}])
        ref = config / "ref.jsonl"
        ref.write_text(ref.read_text() + "\n{not json", encoding="utf-8")
        with pytest.raises(ConfigError, match="invalid JSON in reference"):
            assemble(config, client_factory=scripted_factory())

    def test_display_field_and_fallbacks(self, tmp_path: Path) -> None:
        config = write_config(
            tmp_path / "cfg", recipe=self._recipe(index_field="q"),
            reference=[{"q": "question one", "text": "shown text"}])
        assembled = assemble(config, client_factory=scripted_factory())
        assert assembled.retriever is not None
        hits = assembled.retriever.retrieve("question", k=1)
        assert hits[0].text == "shown text"  # fell back to the 'text' field


class TestExternalStore:
    def test_read_only_sql_store_is_exposed(self, tmp_path: Path) -> None:
        db = tmp_path / "ext.db"
        # Build a tiny external database first.
        from ragkit.store.sql.sqlite import SqliteStore
        SqliteStore(str(db),
                    schema_sql="CREATE TABLE t(a INTEGER); INSERT INTO t VALUES (1);").close()
        storage = f'[sql]\ndriver="sqlite"\npath="{db.as_posix()}"\nread_only=true\n'
        config = write_config(tmp_path / "cfg", storage=storage)
        assembled = assemble(config, client_factory=scripted_factory())
        assert assembled.harness.sql_store is not None
        assert assembled.harness.sql_store.query("SELECT a FROM t")[0]["a"] == 1
