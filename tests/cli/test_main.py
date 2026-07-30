"""The CLI entry point: parsing, substitutions, the no-work path, and structured error exit."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from ragkit.cli.main import _configure_logging, _substitutions, build_parser, main

from .conftest import scripted_factory, write_config


class TestParser:
    def test_requires_config(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_defaults(self) -> None:
        args = build_parser().parse_args(["--config", "cfg"])
        assert args.concurrency == 2 and args.config == Path("cfg")


class TestSubstitutions:
    def test_parses_pairs(self) -> None:
        assert _substitutions(["a=1", "b=two"]) == {"a": "1", "b": "two"}

    def test_none(self) -> None:
        assert _substitutions(None) == {}

    def test_missing_equals_is_refused(self) -> None:
        with pytest.raises(SystemExit, match="key=value"):
            _substitutions(["bogus"])


class TestMain:
    def test_no_pending_records_exits_zero(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
                                           ) -> None:
        # A valid config with an empty catalogue: the pool is built but never contacted.
        config = write_config(tmp_path / "cfg")
        catalog = tmp_path / "empty.jsonl"
        catalog.write_text("", encoding="utf-8")
        code = main(["--config", str(config), "--catalog", str(catalog),
                     "--journal", str(tmp_path / "j.jsonl"), "--no-log-file"])
        assert code == 0 and "nothing to do" in capsys.readouterr().out

    def test_bad_config_exits_one(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        empty = tmp_path / "empty-config"
        empty.mkdir()
        code = main(["--config", str(empty), "--no-log-file"])
        assert code == 1 and "ragkit:" in capsys.readouterr().err

    def test_bad_substitution_exits(self, tmp_path: Path) -> None:
        config = write_config(tmp_path / "cfg")
        with pytest.raises(SystemExit):
            main(["--config", str(config), "--set", "novalue", "--no-log-file"])

    def test_full_run_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                           capsys: pytest.CaptureFixture[str]) -> None:
        # Drive the whole cmd_run path with a scripted pool by intercepting assemble.
        import ragkit.cli.main as main_module
        from ragkit.cli import app
        from ragkit.core.records import Record, read_journal, write_catalog

        config = write_config(tmp_path / "cfg")
        catalog, journal = tmp_path / "c.jsonl", tmp_path / "j.jsonl"
        write_catalog([Record(record_id="1", source="hello")], catalog)

        def patched_assemble(config_dir: Path, **_kwargs: object) -> app.Assembled:
            return app.assemble(config_dir, client_factory=scripted_factory())

        monkeypatch.setattr(main_module, "assemble", patched_assemble)
        code = main(["--config", str(config), "--catalog", str(catalog),
                     "--journal", str(journal), "--concurrency", "1", "--limit", "5",
                     "--no-log-file"])
        assert code == 0
        assert [r.output for r in read_journal(journal)] == ["RESULT"]
        assert "records queued" in capsys.readouterr().out

    def test_seeds_memory_from_an_existing_journal_when_the_recipe_uses_it(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # A recipe with use_memory=true, and a journal that already holds a verified result: the
        # seeded harness.memory must carry it before the run starts (reproducible resume, not a
        # RAM-only memory that forgets what an earlier run already produced).
        import ragkit.cli.main as main_module
        from ragkit.cli import app
        from ragkit.core.records import Record, Status, write_catalog

        recipe_with_memory = """
[task]
output_schema = "json_field"
input_label = "Line:"
use_memory = true
[task.output_schema_options]
field = "translation"
"""
        config = write_config(tmp_path / "cfg", recipe=recipe_with_memory)
        catalog, journal = tmp_path / "c.jsonl", tmp_path / "j.jsonl"
        write_catalog([Record(record_id="1", source="hello")], catalog)
        journal.write_text(
            Record(record_id="prev", source="prior line", output="PRIOR",
                  status=Status.VERIFIED).to_json() + "\n", encoding="utf-8")

        captured: dict[str, object] = {}

        def patched_assemble(config_dir: Path, **_kwargs: object) -> app.Assembled:
            assembled = app.assemble(config_dir, client_factory=scripted_factory())
            captured["harness"] = assembled.harness
            return assembled

        monkeypatch.setattr(main_module, "assemble", patched_assemble)
        code = main(["--config", str(config), "--catalog", str(catalog),
                     "--journal", str(journal), "--concurrency", "1", "--no-log-file"])
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
