"""The translation recipe: its validators, and an end-to-end run through its real config."""
from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from ragkit.cli.app import assemble
from ragkit.core.records import Record, Status
from ragkit.core.rules import Severity
from ragkit.harness import Harness, run_batch
from ragkit.store.run.sqlite import SqliteRunStore

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
        store = SqliteRunStore(str(tmp_path / "run.db"))
        store.add_records([Record(record_id="1", source="The cat is sleeping.")])
        run_batch(assembled.harness, store.pending(), store, install_signal_handlers=False)
        [result] = list(store.results())
        assert result.record.status is Status.VERIFIED and result.record.output == "Kot śpi."

    def test_an_untranslated_echo_is_rejected(self, tmp_path: Path) -> None:
        config = _staged_config(tmp_path)
        source = "The weather is very cold today."
        # The producer echoes the source verbatim every time: the EchoValidator blocks it, and it
        # is rejected after the repair budget rather than shipped untranslated.
        assembled = assemble(config, substitutions={"source_language": "English",
                                                    "target_language": "Polish"},
                             client_factory=_factory(json.dumps({"translation": source})))
        store = SqliteRunStore(str(tmp_path / "run.db"))
        store.add_records([Record(record_id="1", source=source)])
        run_batch(assembled.harness, store.pending(), store, install_signal_handlers=False)
        [result] = list(store.results())
        assert result.record.status is Status.REJECTED

    def test_reference_examples_are_retrieved(self, tmp_path: Path) -> None:
        config = _staged_config(tmp_path)
        assembled = assemble(config, substitutions={"source_language": "English",
                                                    "target_language": "Polish"},
                             client_factory=_factory("{}"))
        assert assembled.retriever is not None
        hits = assembled.retriever.retrieve("The cat is sleeping on the sofa.", k=1)
        assert hits and "Kot śpi na kanapie." in hits[0].text


@pytest.mark.parametrize("pairings_driver", ["sqlite", "duckdb"])
class TestReferenceMemoryOnEachPairingsDriver:
    """The translation memory (config/storage.toml's [pairings]) retrieves identically whether the
    driver is sqlite or duckdb -- only a storage.toml driver edit differs, proving the swap
    property end-to-end over a real recipe (the unit-level conformance suite proves the same thing
    at the driver level; this is the recipe-level guarantee storage-overhaul-plan.md asks for)."""

    def test_translates_with_either_pairings_driver(self, tmp_path: Path,
                                                     pairings_driver: str) -> None:
        config = _staged_config(tmp_path)
        (config / "storage.toml").write_text(
            f'[pairings]\ndriver = "{pairings_driver}"\n'
            f'path = "../data/reference.pairings.{pairings_driver}"\n', encoding="utf-8")
        assembled = assemble(config, substitutions={"source_language": "English",
                                                    "target_language": "Polish"},
                             client_factory=_factory(json.dumps({"translation": "Kot śpi."})))
        assert assembled.retriever is not None
        hits = assembled.retriever.retrieve("The cat is sleeping on the sofa.", k=1)
        assert hits and "Kot śpi na kanapie." in hits[0].text

        store = SqliteRunStore(str(tmp_path / "run.db"))
        store.add_records([Record(record_id="1", source="The cat is sleeping.")])
        run_batch(assembled.harness, store.pending(), store, install_signal_handlers=False)
        [result] = list(store.results())
        assert result.record.status is Status.VERIFIED and result.record.output == "Kot śpi."


class TestFaithfulToLlmTranslator:
    """The recipe reproduces llm-translator's panel, rules, prompts and context building."""

    def _harness(self, tmp_path: Path) -> Harness:
        return assemble(_staged_config(tmp_path),
                        substitutions={"source_language": "English", "target_language": "Polish"},
                        client_factory=_factory("{}")).harness

    def test_the_full_five_reviewer_panel_in_order(self, tmp_path: Path) -> None:
        panel = self._harness(tmp_path).panel
        assert panel.producer.id == "translator"
        assert [r.id for r in panel.reviewers] == [
            "accuracy", "structure", "grammar", "fluency", "compliance"]
        assert panel.reviewers[-1].from_rules  # compliance judges the advisory rules
        assert panel.max_revisions == 2 and panel.max_repairs == 2

    def test_the_full_rule_set(self, tmp_path: Path) -> None:
        rules = self._harness(tmp_path).ruleset
        assert len(rules.forbidden_patterns) == 6   # llm-translator's six failure-mode patterns
        assert {advisory_id for advisory_id, _description in rules.advisory_rules} == {
            "register", "voice_consistency", "continuity", "proper_nouns",
            "pronouns_and_reference", "figurative_language", "localization"}
        assert rules.max_line_columns == 110

    def test_the_translator_system_prompt_matches(self, tmp_path: Path) -> None:
        harness = self._harness(tmp_path)
        prompt = harness._system_prompt(harness.panel.producer.instructions, placeholders=False)
        assert "professional translator working from English into Polish" in prompt
        assert "Who does what to whom" in prompt          # the signature llm-translator section
        assert "Style policy:" in prompt                  # the rule set's directives are appended
        assert "110 display columns" in prompt            # the line-width directive
        assert prompt.rstrip().endswith("Respond only with the requested JSON object.")

    def test_the_user_prompt_builds_the_same_context(self, tmp_path: Path) -> None:
        harness = self._harness(tmp_path)
        record = Record(record_id="1", source="The cat is sleeping on the sofa.",
                        meta={"context_before": ["Hello there."], "context_after": ["Good night."]})
        prompt = harness._user_prompt(record, None)
        # The retrieved translation-memory examples, the neighbours, and the line itself.
        assert "Reference translations of similar lines" in prompt
        assert "Kot śpi na kanapie." in prompt            # a retrieved worked example
        assert "Hello there." in prompt and "Good night." in prompt  # neighbours
        assert prompt.rstrip().endswith("Line to translate:\nThe cat is sleeping on the sofa.")

    def test_a_forbidden_preamble_is_rejected(self, tmp_path: Path) -> None:
        # "Sure! ..." is llm-translator's conversational-preamble failure mode -> blocked, REJECTED.
        config = _staged_config(tmp_path)
        assembled = assemble(config, substitutions={"source_language": "English",
                                                    "target_language": "Polish"},
                             client_factory=_factory(json.dumps({"translation": "Sure! Kot śpi."})))
        store = SqliteRunStore(str(tmp_path / "run.db"))
        store.add_records([Record(record_id="1", source="The cat sleeps.")])
        run_batch(assembled.harness, store.pending(), store, install_signal_handlers=False)
        [result] = list(store.results())
        assert result.record.status is Status.REJECTED


# --- retrieval memory over each real vector DB (lancedb, qdrant) + the default fts5 lexical -------
_VEC_MODELS = """
[endpoint.local]
base_url = "http://127.0.0.1:8080/v1"
[model.translator]
endpoint = "local"
model_id = "t"
[model.reviewer]
endpoint = "local"
model_id = "r"
[model.embedder]
endpoint = "local"
model_id = "e"
kind = "embedding"
"""


def _embed(text: str) -> list[float]:
    # A deterministic, non-zero 3-vector -- retrieval need only run, not be relevant, for the
    # end-to-end to exercise the vector DB.
    return [1.0, (len(text) % 5) / 5.0, (sum(map(ord, text)) % 7) / 7.0]


def _vector_factory() -> Callable[[str, float], httpx.Client]:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={"data": [{"index": i, "embedding": _embed(t)}
                                              for i, t in enumerate(body["input"])]})
        schema = body.get("response_format", {}).get("json_schema", {}).get("schema", {})
        content = (json.dumps({"acceptable": True, "issues": []})
                   if "acceptable" in schema.get("properties", {})
                   else json.dumps({"translation": "Kot śpi."}))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {}})

    def factory(_b: str, _t: float) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler))
    return factory


@pytest.mark.parametrize("vector_driver", ["lancedb", "qdrant"])
class TestRetrievalMemoryOnEachVectorDB:
    """The translation memory is retrieved through dense retrieval over EITHER real vector DB
    (lancedb, qdrant), assembled from retrieval.toml -- the recipe runs end-to-end on each with only
    a storage.toml driver edit. (The default sqlite pairing-store path is covered by
    TestEndToEnd.)"""

    def test_translates_with_a_vector_backed_memory(self, tmp_path: Path,
                                                    vector_driver: str) -> None:
        config = _staged_config(tmp_path)
        (config / "models.toml").write_text(_VEC_MODELS, encoding="utf-8")
        (config / "storage.toml").write_text(
            f'[vector]\ndriver = "{vector_driver}"\npath = "../data/v.{vector_driver}"\ndim = 3\n'
            f'[pairings]\ndriver = "sqlite"\npath = "../data/reference.pairings.db"\n',
            encoding="utf-8")
        (config / "retrieval.toml").write_text(
            '[retrieval]\nkind = "dense"\n[retrieval.dense]\nmodel = "embedder"\n',
            encoding="utf-8")
        assembled = assemble(config, substitutions={"source_language": "English",
                                                    "target_language": "Polish"},
                             client_factory=_vector_factory())
        from ragkit.retrieve.retrievers import DenseRetriever
        assert isinstance(assembled.retriever, DenseRetriever)
        store = SqliteRunStore(str(tmp_path / "run.db"))
        store.add_records([Record(record_id="1", source="The cat is sleeping.")])
        run_batch(assembled.harness, store.pending(), store, install_signal_handlers=False)
        [result] = list(store.results())
        assert result.record.status is Status.VERIFIED and result.record.output == "Kot śpi."


from recipes.translation import eval as tr_eval  # noqa: E402


class TestQualityEval:
    def test_exact_and_similarity(self) -> None:
        report = tr_eval.evaluate([
            ("a", "Kot śpi.", "Kot śpi."),              # exact
            ("b", "Zgasły wszystkie światła.", "Wszystkie światła zgasły."),  # valid, reordered
            ("c", None, "cokolwiek")])                  # not produced -> miss
        assert report.total == 3 and report.produced == 2
        assert report.exact_match == pytest.approx(1 / 3)
        assert 0.0 < report.mean_similarity < 1.0      # 'b' is close but not identical

    def test_empty_report(self) -> None:
        assert tr_eval.Report(()).exact_match == 0.0 and tr_eval.Report(()).mean_similarity == 0.0


class TestEvalGoldAndMain:
    def _gold(self, tmp_path: Path, text: str) -> Path:
        path = tmp_path / "gold.jsonl"
        path.write_text(text, encoding="utf-8")
        return path

    def test_load_gold(self, tmp_path: Path) -> None:
        p = self._gold(tmp_path, '{"record_id":"a","source":"x","target":"Kot śpi."}\n\n')
        assert tr_eval.load_gold(p) == {"a": "Kot śpi."}

    def test_load_gold_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(tr_eval.EvalError, match="not found"):
            tr_eval.load_gold(tmp_path / "no.jsonl")

    def test_load_gold_invalid_json(self, tmp_path: Path) -> None:
        with pytest.raises(tr_eval.EvalError, match="invalid JSON"):
            tr_eval.load_gold(self._gold(tmp_path, "{bad\n"))

    def test_load_gold_missing_field(self, tmp_path: Path) -> None:
        with pytest.raises(tr_eval.EvalError, match="needs 'record_id'"):
            tr_eval.load_gold(self._gold(tmp_path, '{"record_id":"a"}\n'))

    def test_load_gold_empty(self, tmp_path: Path) -> None:
        with pytest.raises(tr_eval.EvalError, match="empty"):
            tr_eval.load_gold(self._gold(tmp_path, "\n"))

    def test_main_success(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        journal = tmp_path / "j.jsonl"
        rec = Record(record_id="a", source="x", output="Kot śpi.", status=Status.VERIFIED)
        journal.write_text(rec.to_json() + "\n", encoding="utf-8")
        gold = self._gold(tmp_path, '{"record_id":"a","source":"x","target":"Kot śpi."}\n')
        code = tr_eval.main(["--journal", str(journal), "--gold", str(gold)])
        assert code == 0 and "exact match 1.000" in capsys.readouterr().out

    def test_main_missing_gold(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        code = tr_eval.main(["--journal", str(tmp_path / "j.jsonl"),
                             "--gold", str(tmp_path / "no.jsonl")])
        assert code == 1 and "error:" in capsys.readouterr().err
