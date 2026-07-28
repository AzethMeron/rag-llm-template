"""The translation recipe: its validators, and an end-to-end run through its real config."""
from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from ragkit.cli.app import assemble
from ragkit.core.records import Record, Status, read_journal, write_catalog
from ragkit.core.rules import Severity
from ragkit.harness import pending_records, run_batch

from recipes.translation.plugins.validators import (
    EchoValidator,
    ScriptValidator,
    available_scripts,
)

CONFIG = Path(__file__).resolve().parents[1] / "config"
SAMPLE = Path(__file__).resolve().parents[1] / "sample" / "reference.jsonl"


def _rec(source: str) -> Record:
    return Record(record_id="1", source=source)


class TestEchoValidator:
    def test_flags_a_verbatim_echo(self) -> None:
        v = EchoValidator().validate(_rec("The cat is here"), "The cat is here", {})
        assert v and v[0].rule_id == "untranslated" and v[0].severity is Severity.ERROR

    def test_flags_echo_ignoring_case_and_whitespace(self) -> None:
        assert EchoValidator().validate(_rec("The Cat Sat"), "the   cat sat", {})

    def test_abstains_on_a_real_translation(self) -> None:
        assert EchoValidator().validate(_rec("The cat sat"), "Kot siedział", {}) == []

    def test_abstains_on_too_short_input(self) -> None:
        # A one-word input may legitimately be the same in both languages (a name).
        assert EchoValidator().validate(_rec("Kraków"), "Kraków", {}) == []

    def test_from_config(self) -> None:
        assert EchoValidator.from_config({"min_words": 3})._min_words == 3


class TestScriptValidator:
    def test_flags_surviving_source_script(self) -> None:
        v = ScriptValidator("japanese").validate(_rec("これは猫です"), "This is 猫", {})
        assert v and v[0].rule_id == "untranslated"

    def test_abstains_when_fully_translated(self) -> None:
        assert ScriptValidator("japanese").validate(_rec("これは猫です"), "This is a cat", {}) == []

    def test_abstains_when_source_is_not_in_the_script(self) -> None:
        # An English source under a Japanese-script check: the check does not apply, so abstain.
        assert ScriptValidator("japanese").validate(_rec("hello world"), "hello 猫", {}) == []

    def test_tolerates_a_few_survivors(self) -> None:
        v = ScriptValidator("japanese", max_survivors=2)
        assert v.validate(_rec("猫が好き"), "I like 猫", {}) == []  # one survivor, tolerated

    def test_unknown_script_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown source_script"):
            ScriptValidator("klingon")

    def test_available_scripts(self) -> None:
        assert "japanese" in available_scripts() and "cyrillic" in available_scripts()


def _factory(produce: str) -> Callable[[str, float], httpx.Client]:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        schema = body.get("response_format", {}).get("json_schema", {}).get("schema", {})
        if "acceptable" in schema.get("properties", {}):
            content = json.dumps({"acceptable": True, "issues": []})
        else:
            content = produce
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {}})

    def factory(_b: str, _t: float) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler))
    return factory


def _staged_config(tmp_path: Path) -> Path:
    """Copy the recipe's real config into a tmp tree and place the sample corpus where the recipe's
    reference path (../data/reference.jsonl) points."""
    config = tmp_path / "config"
    shutil.copytree(CONFIG, config)
    (tmp_path / "data").mkdir()
    shutil.copy(SAMPLE, tmp_path / "data" / "reference.jsonl")
    return config


class TestEndToEnd:
    def test_translates_through_the_real_config(self, tmp_path: Path) -> None:
        config = _staged_config(tmp_path)
        assembled = assemble(config, substitutions={"source_language": "English",
                                                    "target_language": "Polish"},
                             client_factory=_factory(json.dumps({"translation": "Kot śpi."})))
        catalog, journal = tmp_path / "c.jsonl", tmp_path / "j.jsonl"
        write_catalog([Record(record_id="1", source="The cat is sleeping.")], catalog)
        run_batch(assembled.harness, pending_records(catalog, journal), journal,
                  install_signal_handlers=False)
        [result] = list(read_journal(journal))
        assert result.status is Status.VERIFIED and result.output == "Kot śpi."

    def test_an_untranslated_echo_is_rejected(self, tmp_path: Path) -> None:
        config = _staged_config(tmp_path)
        source = "The weather is very cold today."
        # The producer echoes the source verbatim every time: the EchoValidator blocks it, and it
        # is rejected after the repair budget rather than shipped untranslated.
        assembled = assemble(config, substitutions={"source_language": "English",
                                                    "target_language": "Polish"},
                             client_factory=_factory(json.dumps({"translation": source})))
        catalog, journal = tmp_path / "c.jsonl", tmp_path / "j.jsonl"
        write_catalog([Record(record_id="1", source=source)], catalog)
        run_batch(assembled.harness, pending_records(catalog, journal), journal,
                  install_signal_handlers=False)
        [result] = list(read_journal(journal))
        assert result.status is Status.REJECTED

    def test_reference_examples_are_retrieved(self, tmp_path: Path) -> None:
        config = _staged_config(tmp_path)
        assembled = assemble(config, substitutions={"source_language": "English",
                                                    "target_language": "Polish"},
                             client_factory=_factory("{}"))
        assert assembled.retriever is not None
        hits = assembled.retriever.retrieve("The cat is sleeping on the sofa.", k=1)
        assert hits and "Kot śpi na kanapie." in hits[0].text
