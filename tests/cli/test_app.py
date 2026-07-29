"""Config-driven assembly, and the checks it makes before any server is contacted."""
from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.core.config import ConfigError
from ragkit.core.records import Record, Status, read_journal, write_catalog
from ragkit.harness import pending_records, run_batch
from ragkit.llm.pool import ModelPoolError
from ragkit.cli.app import assemble

from .conftest import RETRIEVAL_MODELS, retrieval_factory, scripted_factory, write_config

_REF = [{"source": "the cat sat", "target": "kot"}, {"source": "a dog ran", "target": "pies"}]
_VECTOR_STORAGE = '[vector]\ndriver = "lancedb"\npath = "v.lance"\ndim = 3\n'
_RECIPE_WITH_REF = ('[task]\noutput_schema = "json_field"\n[task.output_schema_options]\n'
                    'field = "translation"\n[reference]\nfile = "ref.jsonl"\n')


class StandaloneRetriever:
    """A corpus-free custom Retriever with its own 'backend' — selectable by dotted path through
    the RETRIEVERS registry, built from config alone (no reference corpus of ours)."""

    CONFIG_KEYS = frozenset({"tag"})

    def __init__(self, tag: str = "") -> None:
        self._tag = tag

    @classmethod
    def from_config(cls, options: dict) -> StandaloneRetriever:
        return cls(tag=str(options.get("tag", "")))

    def retrieve(self, query: str, *, k: int, min_score: float = 0.0) -> tuple:
        from ragkit.core.ports import Retrieved
        return (Retrieved("s", f"{self._tag}:{query}", 1.0),)[:k]


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

    def test_on_disk_reference_index_is_built_once_and_reused(self, tmp_path: Path) -> None:
        # With an on-disk document store the corpus is streamed in on the first assemble and the
        # persisted rows are reused on the next — no re-ingest (the build-once path keys on the
        # document store's count()).
        config = write_config(tmp_path / "cfg", recipe=_RECIPE_WITH_REF,
                              reference=[{"source": "the cat sat", "target": "kot"}],
                              storage='[lexical]\ndriver = "fts5"\npath = "lex.db"\n'
                                      '[documents]\ndriver = "sqlite"\npath = "rows.db"\n')
        first = assemble(config, client_factory=scripted_factory())
        assert first.retriever.retrieve("cat", k=1)[0].text == "the cat sat -> kot"
        # Overwrite the source corpus; a reuse (no re-ingest) still serves the original passage.
        (config / "ref.jsonl").write_text('{"source": "a dog ran", "target": "pies"}\n',
                                          encoding="utf-8")
        second = assemble(config, client_factory=scripted_factory())
        assert second.retriever.retrieve("cat", k=1)[0].text == "the cat sat -> kot"

    def test_injected_retriever_overrides_the_config(self, tmp_path: Path) -> None:
        # The replace-without-editing-our-code seam for a corpus-stateful Retriever: a caller
        # passes one to assemble(), and it is used instead of whatever [reference] would build.
        from ragkit.core.ports import Retrieved

        class MyRetriever:
            def retrieve(self, query: str, *, k: int, min_score: float = 0.0) -> tuple:
                return (Retrieved("x", f"custom:{query}", 1.0),)

        recipe = '[task]\noutput_schema = "json_field"\n[reference]\nfile = "ref.jsonl"\n'
        config = write_config(tmp_path / "cfg", recipe=recipe,
                              reference=[{"source": "the cat", "target": "kot"}])
        mine = MyRetriever()
        assembled = assemble(config, client_factory=scripted_factory(), retriever=mine)
        assert assembled.retriever is mine  # injection wins over the config-built lexical one

    def test_custom_retriever_by_dotted_path_is_resolved_through_the_registry(
            self, tmp_path: Path) -> None:
        # A corpus-free custom Retriever (its own backend) is selected by dotted path in config and
        # resolved through the RETRIEVERS registry -- no reference file, no code change of ours.
        recipe = ('[task]\noutput_schema = "json_field"\n'
                  f'[reference]\nretriever = "{__name__}:StandaloneRetriever"\n'
                  '[reference.options]\ntag = "hi"\n')
        config = write_config(tmp_path / "cfg", recipe=recipe)
        assembled = assemble(config, client_factory=scripted_factory())
        assert isinstance(assembled.retriever, StandaloneRetriever)
        assert assembled.retriever.retrieve("q", k=1)[0].text == "hi:q"

    def test_substitutions_fill_persona_instructions(self, tmp_path: Path) -> None:
        personas = ('[[persona]]\nid="p"\nkind="producer"\nmodel="prod"\n'
                    'instructions="translate to {target_language}"\n'
                    '[[persona]]\nid="r"\nkind="reviewer"\nmodel="rev"\ninstructions="review"\n')
        config = write_config(tmp_path / "cfg", personas=personas)
        assembled = assemble(config, substitutions={"target_language": "Polish"},
                             client_factory=scripted_factory())
        assert "Polish" in assembled.harness.panel.producer.instructions


class TestRetrievalToml:
    def test_lexical_stack_from_config(self, tmp_path: Path) -> None:
        config = write_config(tmp_path / "cfg", recipe=_RECIPE_WITH_REF, reference=_REF,
                              retrieval='[retrieval]\nkind = "lexical"\n[retrieval.lexical]\n'
                                        'min_score = 0.1\n')
        assembled = assemble(config, client_factory=scripted_factory())
        from ragkit.retrieve.retrievers import LexicalRetriever
        assert isinstance(assembled.retriever, LexicalRetriever)

    def test_hybrid_stack_from_config_assembles_and_retrieves(self, tmp_path: Path) -> None:
        retrieval = ('[retrieval]\nkind = "hybrid"\ncandidate_pool = 10\n'
                     '[retrieval.dense]\nmodel = "embedder"\n'
                     '[retrieval.rerank]\nenabled = true\nmodel = "reranker"\n')
        config = write_config(tmp_path / "cfg", models=RETRIEVAL_MODELS, recipe=_RECIPE_WITH_REF,
                              reference=_REF, storage=_VECTOR_STORAGE, retrieval=retrieval)
        assembled = assemble(config, client_factory=retrieval_factory())
        from ragkit.retrieve.hybrid import HybridRetriever
        assert isinstance(assembled.retriever, HybridRetriever)
        hits = assembled.retriever.retrieve("cat", k=2, min_score=0.0)
        assert any("cat" in hit.text for hit in hits)  # the stack actually returns the cat doc

    def test_dense_stack_from_config(self, tmp_path: Path) -> None:
        config = write_config(tmp_path / "cfg", models=RETRIEVAL_MODELS, recipe=_RECIPE_WITH_REF,
                              reference=_REF, storage=_VECTOR_STORAGE,
                              retrieval='[retrieval]\nkind = "dense"\n[retrieval.dense]\n'
                                        'model = "embedder"\n')
        assembled = assemble(config, client_factory=retrieval_factory())
        from ragkit.retrieve.retrievers import DenseRetriever
        assert isinstance(assembled.retriever, DenseRetriever)

    def test_dense_without_a_vector_store_is_refused(self, tmp_path: Path) -> None:
        config = write_config(tmp_path / "cfg", models=RETRIEVAL_MODELS, recipe=_RECIPE_WITH_REF,
                              reference=_REF,  # no storage.toml -> no [vector]
                              retrieval='[retrieval]\nkind = "dense"\n[retrieval.dense]\n'
                                        'model = "embedder"\n')
        with pytest.raises(ConfigError, match=r"needs a \[vector\] store"):
            assemble(config, client_factory=retrieval_factory())

    def test_wrong_kind_of_model_is_refused(self, tmp_path: Path) -> None:
        config = write_config(tmp_path / "cfg", models=RETRIEVAL_MODELS, recipe=_RECIPE_WITH_REF,
                              reference=_REF, storage=_VECTOR_STORAGE,
                              retrieval='[retrieval]\nkind = "dense"\n[retrieval.dense]\n'
                                        'model = "prod"\n')  # prod is a chat model
        with pytest.raises(ConfigError, match="not an embedding model"):
            assemble(config, client_factory=retrieval_factory())

    def test_retrieval_without_a_reference_file_is_refused(self, tmp_path: Path) -> None:
        config = write_config(tmp_path / "cfg",  # recipe has no [reference].file
                              retrieval='[retrieval]\nkind = "lexical"\n')
        with pytest.raises(ConfigError, match=r"no \[reference\]\.file"):
            assemble(config, client_factory=scripted_factory())

    def test_missing_reference_corpus_file_is_refused(self, tmp_path: Path) -> None:
        # [reference].file is set but the file is absent (reference=None writes no ref.jsonl).
        config = write_config(tmp_path / "cfg", recipe=_RECIPE_WITH_REF,
                              retrieval='[retrieval]\nkind = "lexical"\n')
        with pytest.raises(ConfigError, match="reference corpus not found"):
            assemble(config, client_factory=scripted_factory())

    def test_wrong_kind_of_rerank_model_is_refused(self, tmp_path: Path) -> None:
        config = write_config(tmp_path / "cfg", models=RETRIEVAL_MODELS, recipe=_RECIPE_WITH_REF,
                              reference=_REF, storage=_VECTOR_STORAGE,
                              retrieval='[retrieval]\nkind = "hybrid"\n[retrieval.dense]\n'
                                        'model = "embedder"\n[retrieval.rerank]\nenabled = true\n'
                                        'model = "prod"\n')  # prod is a chat model
        with pytest.raises(ConfigError, match="not a rerank model"):
            assemble(config, client_factory=retrieval_factory())


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
