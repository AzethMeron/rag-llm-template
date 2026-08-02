"""The CLI entry point: subcommand parsing, substitutions, the no-work path, structured error
exit, and the import/run/export split over a run store."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from ragkit.cli.main import _configure_logging, _substitutions, build_parser, main
from ragkit.core.records import Record, Status, read_journal, write_catalog
from ragkit.store.run.sqlite import SqliteRunStore

from .conftest import scripted_factory, write_config


class TestParser:
    def test_requires_a_subcommand(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_run_requires_config(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["run"])

    def test_run_defaults(self) -> None:
        args = build_parser().parse_args(["run", "--config", "cfg"])
        assert args.concurrency == 2 and args.config == Path("cfg")
        assert args.run_db == Path("work/run.db")

    def test_import_defaults(self) -> None:
        args = build_parser().parse_args(["import"])
        assert args.catalog == Path("work/records.jsonl")
        assert args.run_db == Path("work/run.db")

    def test_export_defaults(self) -> None:
        args = build_parser().parse_args(["export"])
        assert args.journal == Path("work/journal.jsonl")
        assert args.run_db == Path("work/run.db")

    def test_writeback_defaults(self) -> None:
        args = build_parser().parse_args(["writeback"])
        assert args.run_db == Path("work/run.db")
        assert args.pairings_db == Path("work/pairings.db")
        assert args.pairings_driver == "sqlite"
        assert args.vector_path is None


class TestSubstitutions:
    def test_parses_pairs(self) -> None:
        assert _substitutions(["a=1", "b=two"]) == {"a": "1", "b": "two"}

    def test_none(self) -> None:
        assert _substitutions(None) == {}

    def test_missing_equals_is_refused(self) -> None:
        with pytest.raises(SystemExit, match="key=value"):
            _substitutions(["bogus"])


class TestImport:
    def test_imports_a_catalogue(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        catalog = tmp_path / "c.jsonl"
        write_catalog([Record(record_id="1", source="hello")], catalog)
        run_db = tmp_path / "run.db"
        code = main(["import", "--catalog", str(catalog), "--run-db", str(run_db),
                     "--no-log-file"])
        assert code == 0
        assert "1 record(s) added" in capsys.readouterr().out
        assert SqliteRunStore(str(run_db)).count_records() == 1

    def test_reimporting_is_idempotent(self, tmp_path: Path,
                                       capsys: pytest.CaptureFixture[str]) -> None:
        catalog = tmp_path / "c.jsonl"
        write_catalog([Record(record_id="1", source="hello")], catalog)
        run_db = tmp_path / "run.db"
        main(["import", "--catalog", str(catalog), "--run-db", str(run_db), "--no-log-file"])
        capsys.readouterr()
        code = main(["import", "--catalog", str(catalog), "--run-db", str(run_db),
                     "--no-log-file"])
        assert code == 0
        assert "0 record(s) added" in capsys.readouterr().out
        assert SqliteRunStore(str(run_db)).count_records() == 1


class TestExport:
    def test_exports_results_as_a_journal(self, tmp_path: Path,
                                          capsys: pytest.CaptureFixture[str]) -> None:
        from ragkit.core.ports import RunResult

        run_db = tmp_path / "run.db"
        store = SqliteRunStore(str(run_db))
        store.add_records([Record(record_id="1", source="hello")])
        store.append_result(RunResult(
            record=Record(record_id="1", source="hello", status=Status.VERIFIED, output="HI")))
        journal = tmp_path / "j.jsonl"
        code = main(["export", "--run-db", str(run_db), "--journal", str(journal),
                     "--no-log-file"])
        assert code == 0
        assert "1 result(s) exported" in capsys.readouterr().out
        [record] = list(read_journal(journal))
        assert record.output == "HI" and record.status is Status.VERIFIED


class TestWriteback:
    def test_writes_verified_results_as_pairings(self, tmp_path: Path,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
        from ragkit.core.ports import RunResult
        from ragkit.store.pairings.sqlite import SqlitePairings

        run_db = tmp_path / "run.db"
        run_store = SqliteRunStore(str(run_db))
        run_store.add_records([Record(record_id="1", source="q1")])
        run_store.append_result(RunResult(
            record=Record(record_id="1", source="q1", status=Status.VERIFIED, output="a1"),
            context_passage="ctx"))

        pairings_db = tmp_path / "pairings.db"
        code = main(["writeback", "--run-db", str(run_db), "--pairings-db", str(pairings_db),
                     "--no-log-file"])
        assert code == 0
        assert "1 pairing(s) written back" in capsys.readouterr().out

        store = SqlitePairings(str(pairings_db))
        assert store.count() == 1
        [chunk_id] = list(store.all_ids())
        pairing = store.get(chunk_id)
        assert pairing is not None
        assert pairing.source == "q1" and pairing.target == "a1" and pairing.context == "ctx"

    def test_a_subsequent_run_retrieves_the_written_back_pairing(self, tmp_path: Path) -> None:
        # The accumulating-memory use case: writeback's whole point is that a LATER run's retriever
        # -- built the same way assemble() builds one, over the same [pairings] path -- finds what
        # an earlier run produced and verified.
        from ragkit.core.ports import RunResult
        from ragkit.ingest.reference import PairingRetrievers
        from ragkit.store.pairings.sqlite import SqlitePairings

        run_db = tmp_path / "run.db"
        run_store = SqliteRunStore(str(run_db))
        run_store.add_records([Record(record_id="1", source="What is the capital of Poland?")])
        run_store.append_result(RunResult(
            record=Record(record_id="1", source="What is the capital of Poland?",
                         status=Status.VERIFIED, output="Warsaw"),
            context_passage="ctx"))

        pairings_db = tmp_path / "pairings.db"
        code = main(["writeback", "--run-db", str(run_db), "--pairings-db", str(pairings_db),
                     "--no-log-file"])
        assert code == 0

        retrievers = PairingRetrievers(SqlitePairings(str(pairings_db)))
        hits = retrievers.lexical_retriever().retrieve("capital of Poland", k=1)
        assert hits and "Warsaw" in hits[0].text

    def test_vector_path_without_embedding_url_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="needs --embedding-url"):
            main(["writeback", "--run-db", str(tmp_path / "run.db"),
                 "--pairings-db", str(tmp_path / "p.db"),
                 "--vector-path", str(tmp_path / "v.lance"), "--no-log-file"])

    def test_vector_path_without_a_dim_is_the_drivers_own_structured_error(
            self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        # --vector-dim is optional at the CLI layer; a real vector driver still needs a positive
        # dim, so omitting it surfaces the driver's own structured error rather than a silent
        # default.
        code = main(["writeback", "--run-db", str(tmp_path / "run.db"),
                     "--pairings-db", str(tmp_path / "p.db"),
                     "--vector-path", str(tmp_path / "v.lance"),
                     "--embedding-url", "http://x/v1", "--no-log-file"])
        assert code == 1 and "positive dim" in capsys.readouterr().err

    def test_reconciles_the_vector_index_when_configured(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import json

        import httpx

        import ragkit.cli.main as main_module
        from ragkit.core.ports import RunResult
        from ragkit.retrieve.embedding import EmbeddingClient
        from ragkit.store.pairings.sqlite import SqlitePairings
        from ragkit.store.vector.lancedb import LanceVectorIndex

        run_db = tmp_path / "run.db"
        run_store = SqliteRunStore(str(run_db))
        run_store.add_records([Record(record_id="1", source="cat text")])
        run_store.append_result(RunResult(
            record=Record(record_id="1", source="cat text", status=Status.VERIFIED, output="a1")))

        def handler(request: httpx.Request) -> httpx.Response:
            inputs = json.loads(request.content)["input"]
            return httpx.Response(200, json={
                "data": [{"index": i, "embedding": [1.0, 0.0]} for i, _ in enumerate(inputs)]})

        def fake_embedding_client(*, base_url: str, model: str) -> EmbeddingClient:
            return EmbeddingClient(base_url=base_url, model=model,
                                   client=httpx.Client(transport=httpx.MockTransport(handler)))

        monkeypatch.setattr(main_module, "EmbeddingClient", fake_embedding_client)

        pairings_db = tmp_path / "pairings.db"
        vector_path = tmp_path / "v.lance"
        code = main(["writeback", "--run-db", str(run_db), "--pairings-db", str(pairings_db),
                     "--vector-path", str(vector_path), "--vector-dim", "2",
                     "--embedding-url", "http://x/v1", "--no-log-file"])
        assert code == 0

        vector = LanceVectorIndex(str(vector_path), dim=2)
        assert vector.count() == 1
        pairings = SqlitePairings(str(pairings_db))
        [chunk_id] = list(pairings.all_ids())
        assert vector.search([1.0, 0.0], k=1)[0][0] == chunk_id


class TestRun:
    def test_no_pending_records_exits_zero(self, tmp_path: Path,
                                           capsys: pytest.CaptureFixture[str]) -> None:
        # A valid config with a fresh (empty) run store: the pool is built but never contacted.
        config = write_config(tmp_path / "cfg")
        code = main(["run", "--config", str(config), "--run-db", str(tmp_path / "run.db"),
                     "--no-log-file"])
        assert code == 0 and "nothing to do" in capsys.readouterr().out

    def test_bad_config_exits_one(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        empty = tmp_path / "empty-config"
        empty.mkdir()
        code = main(["run", "--config", str(empty), "--no-log-file"])
        assert code == 1 and "ragkit:" in capsys.readouterr().err

    def test_bad_substitution_exits(self, tmp_path: Path) -> None:
        config = write_config(tmp_path / "cfg")
        with pytest.raises(SystemExit):
            main(["run", "--config", str(config), "--set", "novalue", "--no-log-file"])

    def test_full_run_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                           capsys: pytest.CaptureFixture[str]) -> None:
        # Drive the whole cmd_run path with a scripted pool by intercepting assemble.
        import ragkit.cli.main as main_module
        from ragkit.cli import app

        config = write_config(tmp_path / "cfg")
        run_db = tmp_path / "run.db"
        store = SqliteRunStore(str(run_db))
        store.add_records([Record(record_id="1", source="hello")])

        def patched_assemble(config_dir: Path, **_kwargs: object) -> app.Assembled:
            return app.assemble(config_dir, client_factory=scripted_factory())

        monkeypatch.setattr(main_module, "assemble", patched_assemble)
        code = main(["run", "--config", str(config), "--run-db", str(run_db),
                     "--concurrency", "1", "--limit", "5", "--no-log-file"])
        assert code == 0
        [result] = list(store.results())
        assert result.record.output == "RESULT"
        assert "records queued" in capsys.readouterr().out

    def test_seeds_memory_from_existing_results_when_the_recipe_uses_it(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # A recipe with use_memory=true, and a run store that already holds a verified result: the
        # seeded harness.memory must carry it before the run starts (reproducible resume, not a
        # RAM-only memory that forgets what an earlier run already produced).
        import ragkit.cli.main as main_module
        from ragkit.cli import app
        from ragkit.core.ports import RunResult

        recipe_with_memory = """
[task]
output_schema = "json_field"
input_label = "Line:"
use_memory = true
[task.output_schema_options]
field = "translation"
"""
        config = write_config(tmp_path / "cfg", recipe=recipe_with_memory)
        run_db = tmp_path / "run.db"
        store = SqliteRunStore(str(run_db))
        store.add_records([Record(record_id="1", source="hello"),
                           Record(record_id="prev", source="prior line")])
        store.append_result(RunResult(
            record=Record(record_id="prev", source="prior line", output="PRIOR",
                          status=Status.VERIFIED)))

        captured: dict[str, object] = {}

        def patched_assemble(config_dir: Path, **_kwargs: object) -> app.Assembled:
            assembled = app.assemble(config_dir, client_factory=scripted_factory())
            captured["harness"] = assembled.harness
            return assembled

        monkeypatch.setattr(main_module, "assemble", patched_assemble)
        code = main(["run", "--config", str(config), "--run-db", str(run_db),
                     "--concurrency", "1", "--no-log-file"])
        assert code == 0
        harness = captured["harness"]
        assert harness.memory is not None and harness.memory.get("prior line") == "PRIOR"


class TestLogging:
    def test_configures_a_file_handler(self, tmp_path: Path) -> None:
        log = tmp_path / "run.log"
        _configure_logging(log)
        logging.getLogger("ragkit").warning("hello")
        assert log.exists() and "hello" in log.read_text(encoding="utf-8")

    def test_console_only(self) -> None:
        _configure_logging(None)
        handlers = logging.getLogger("ragkit").handlers
        assert len(handlers) == 1  # stderr only
