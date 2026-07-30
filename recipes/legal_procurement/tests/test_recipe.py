"""The legal-procurement recipe: the citation-grounding validator (adversarial), the
retrieval-quality eval (Recall@20/MRR@10/NDCG@10 against gold), and an end-to-end run that answers
from a legal-passage memory and rejects a fabricated citation."""
from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from ragkit.cli.app import assemble
from ragkit.core.ports import Retrieved
from ragkit.core.records import Record, Status
from ragkit.harness import run_batch
from ragkit.store.run.sqlite import SqliteRunStore

from recipes.legal_procurement import eval as lp_eval
from recipes.legal_procurement.plugins.validators import CitationGroundingValidator

CONFIG = Path(__file__).resolve().parents[1] / "config"

QUESTION = "W jakim trybie zamawiający udziela zamówienia publicznego?"
PASSAGE1 = "Zamawiający udziela zamówienia publicznego w trybie przetargu nieograniczonego."
PASSAGE2 = "Odwołanie wnosi się do Prezesa Krajowej Izby Odwoławczej w terminie dziesięciu dni."


# --- the citation-grounding validator ------------------------------------------------------------
class _StubRetriever:
    def __init__(self, ids: list[str]) -> None:
        self._hits = [Retrieved(i, f"passage {i}", 1.0) for i in ids]

    def retrieve(self, query: str, *, k: int, min_score: float = 0.0) -> tuple[Retrieved, ...]:
        return tuple(self._hits[:k])


def _validator() -> CitationGroundingValidator:
    return CitationGroundingValidator()


def _rec() -> Record:
    return Record(record_id="q1", source=QUESTION)


def _answer(**over: object) -> str:
    base: dict[str, object] = {
        "answer": "Zamawiający udziela zamówienia w trybie przetargu nieograniczonego.",
        "citations": ["ref-1"]}
    base.update(over)
    return json.dumps(base)


def _ctx(ids: list[str] | None = None) -> dict:
    return {"retriever": _StubRetriever(["ref-1", "ref-2"] if ids is None else ids)}


class TestCitationRefusals:
    def _refused(self, output: str, ctx: dict | None = None) -> bool:
        vs = _validator().validate(_rec(), output, ctx if ctx is not None else _ctx())
        return any(v.rule_id == "ungrounded_citation" and v.blocking for v in vs)

    def test_a_fabricated_citation_is_refused(self) -> None:
        assert self._refused(_answer(citations=["ref-999"]))

    def test_empty_citations_on_a_substantive_answer_is_refused(self) -> None:
        assert self._refused(_answer(citations=[]))

    def test_citations_not_a_list_is_refused(self) -> None:
        assert self._refused(_answer(citations="ref-1"))

    def test_a_blank_answer_is_refused(self) -> None:
        assert self._refused(_answer(answer="   "))

    def test_a_missing_answer_field_is_refused(self) -> None:
        assert self._refused(json.dumps({"citations": ["ref-1"]}))

    def test_missing_retriever_blocks_rather_than_passes(self) -> None:
        # A cite-from-memory recipe with no memory wired cannot verify grounding -> block.
        assert self._refused(_answer(), ctx={})

    def test_invalid_json_is_refused(self) -> None:
        assert self._refused("{not json")

    def test_non_object_json_is_refused(self) -> None:
        assert self._refused("[1, 2, 3]")

    def test_one_good_one_fabricated_citation_is_refused(self) -> None:
        assert self._refused(_answer(citations=["ref-1", "ref-999"]))


class TestCitationAcceptance:
    def test_a_grounded_citation_passes(self) -> None:
        assert _validator().validate(_rec(), _answer(), _ctx()) == []

    def test_multiple_grounded_citations_pass(self) -> None:
        out = _answer(citations=["ref-1", "ref-2"])
        assert _validator().validate(_rec(), out, _ctx()) == []

    def test_abstention_with_no_citations_passes(self) -> None:
        out = json.dumps({"answer": "brak podstaw", "citations": []})
        assert _validator().validate(_rec(), out, _ctx()) == []

    def test_abstention_marker_inside_a_sentence_passes(self) -> None:
        # Abstention is detected by the marker anywhere in the answer, and no citations are
        # required.
        out = json.dumps({"answer": "Brak podstaw w przedstawionych przepisach.", "citations": []})
        assert _validator().validate(_rec(), out, _ctx()) == []

    def test_integer_citation_ids_are_coerced_and_checked(self) -> None:
        retriever = {"retriever": _StubRetriever(["1", "2"])}
        assert _validator().validate(_rec(), _answer(citations=[1]), retriever) == []

    def test_from_config(self) -> None:
        validator = CitationGroundingValidator.from_config({"k": 10, "abstain_marker": "brak"})
        assert validator.validate(_rec(), _answer(), _ctx()) == []

    def test_construction_guards(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            CitationGroundingValidator(k=0)
        with pytest.raises(ValueError, match="'answer_field'"):
            CitationGroundingValidator(answer_field=" ")
        with pytest.raises(ValueError, match="'citations_field'"):
            CitationGroundingValidator(citations_field="")
        with pytest.raises(ValueError, match="'abstain_marker'"):
            CitationGroundingValidator(abstain_marker="  ")


# --- the retrieval-quality eval ------------------------------------------------------------------
class _RankRetriever:
    """Returns a fixed ranking for every query, so metric arithmetic is deterministic."""

    def __init__(self, ranking: list[str]) -> None:
        self._hits = [Retrieved(i, f"passage {i}", 1.0 - n / 100) for n, i in enumerate(ranking)]

    def retrieve(self, query: str, *, k: int, min_score: float = 0.0) -> tuple[Retrieved, ...]:
        return tuple(self._hits[:k])


class TestEvaluate:
    def test_means_over_a_hit_and_a_miss(self) -> None:
        retriever = _RankRetriever(["ref-1", "ref-9"])
        queries = [("q1", "a"), ("q2", "b"), ("q3", "c")]  # q3 has no gold -> skipped
        gold = {"q1": frozenset({"ref-1"}), "q2": frozenset({"ref-2"})}
        report = lp_eval.evaluate(retriever, queries, gold, k=20)
        assert report.queries == 2
        assert report.recall_at_k == pytest.approx(0.5)  # q1 hit, q2 miss
        assert report.mrr == pytest.approx(0.5)           # q1 rank-1, q2 none
        assert report.ndcg == pytest.approx(0.5)
        for value in (report.recall_at_k, report.mrr, report.ndcg):
            assert 0.0 <= value <= 1.0

    def test_a_perfect_ranking_scores_one(self) -> None:
        retriever = _RankRetriever(["ref-1", "ref-2", "ref-3"])
        report = lp_eval.evaluate(retriever, [("q1", "a")], {"q1": frozenset({"ref-1"})}, k=20)
        assert report.recall_at_k == 1.0 and report.mrr == 1.0 and report.ndcg == 1.0

    def test_empty_report_is_zero(self) -> None:
        report = lp_eval.Report((), k=20)
        assert report.queries == 0
        assert report.recall_at_k == 0.0 and report.mrr == 0.0 and report.ndcg == 0.0

    def test_over_the_real_assembled_retriever(self, tmp_path: Path) -> None:
        config = _staged(tmp_path)
        retriever = assemble(config).retriever
        assert retriever is not None
        queries = [("q1", "W jakim trybie zamawiający udziela zamówienia?"),
                   ("q2", "Do kogo wnosi się odwołanie Krajowa Izba Odwoławcza?")]
        gold = {"q1": frozenset({"ref-1"}), "q2": frozenset({"ref-2"})}
        report = lp_eval.evaluate(retriever, queries, gold, k=20)
        assert report.queries == 2
        for value in (report.recall_at_k, report.mrr, report.ndcg):
            assert 0.0 <= value <= 1.0
        assert report.recall_at_k > 0.0  # the lexical retriever finds the gold passages


class TestLoadGold:
    def _write(self, tmp_path: Path, text: str) -> Path:
        path = tmp_path / "gold.jsonl"
        path.write_text(text, encoding="utf-8")
        return path

    def test_reads_rows(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, '{"record_id":"q1","relevant":["ref-1","ref-2"]}\n\n')
        assert lp_eval.load_gold(path) == {"q1": frozenset({"ref-1", "ref-2"})}

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(lp_eval.EvalError, match="not found"):
            lp_eval.load_gold(tmp_path / "no.jsonl")

    def test_invalid_json(self, tmp_path: Path) -> None:
        with pytest.raises(lp_eval.EvalError, match="invalid JSON"):
            lp_eval.load_gold(self._write(tmp_path, "{bad\n"))

    def test_missing_field(self, tmp_path: Path) -> None:
        with pytest.raises(lp_eval.EvalError, match="needs 'record_id'"):
            lp_eval.load_gold(self._write(tmp_path, '{"record_id":"q1"}\n'))

    def test_relevant_must_be_nonempty_list(self, tmp_path: Path) -> None:
        with pytest.raises(lp_eval.EvalError, match="non-empty list"):
            lp_eval.load_gold(self._write(tmp_path, '{"record_id":"q1","relevant":[]}\n'))

    def test_empty_file(self, tmp_path: Path) -> None:
        with pytest.raises(lp_eval.EvalError, match="empty"):
            lp_eval.load_gold(self._write(tmp_path, "\n"))


class TestLoadHeldout:
    def _write(self, tmp_path: Path, text: str) -> Path:
        path = tmp_path / "heldout.jsonl"
        path.write_text(text, encoding="utf-8")
        return path

    def test_reads_rows(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, '{"record_id":"q1","source":"Pytanie?","meta":{}}\n')
        assert lp_eval.load_heldout(path) == [("q1", "Pytanie?")]

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(lp_eval.EvalError, match="not found"):
            lp_eval.load_heldout(tmp_path / "no.jsonl")

    def test_missing_field(self, tmp_path: Path) -> None:
        with pytest.raises(lp_eval.EvalError, match="needs 'record_id'"):
            lp_eval.load_heldout(self._write(tmp_path, '{"record_id":"q1"}\n'))

    def test_invalid_json(self, tmp_path: Path) -> None:
        with pytest.raises(lp_eval.EvalError, match="invalid JSON"):
            lp_eval.load_heldout(self._write(tmp_path, "{bad\n"))

    def test_empty_file(self, tmp_path: Path) -> None:
        with pytest.raises(lp_eval.EvalError, match="empty"):
            lp_eval.load_heldout(self._write(tmp_path, "\n"))


def _eval_data(tmp_path: Path) -> tuple[Path, Path, Path]:
    config = _staged(tmp_path)
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text(
        json.dumps({"record_id": "q1", "source": "W jakim trybie zamawiający udziela zamówienia?",
                    "meta": {}}, ensure_ascii=False) + "\n"
        + json.dumps({"record_id": "q2", "source": "Do kogo wnosi się odwołanie?", "meta": {}},
                     ensure_ascii=False) + "\n", encoding="utf-8")
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps({"record_id": "q1", "relevant": ["ref-1"]}) + "\n"
        + json.dumps({"record_id": "q2", "relevant": ["ref-2"]}) + "\n", encoding="utf-8")
    return config, heldout, gold


class TestEvalMain:
    def test_success(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        config, heldout, gold = _eval_data(tmp_path)
        code = lp_eval.main(["--config", str(config), "--heldout", str(heldout),
                             "--gold", str(gold)])
        out = capsys.readouterr().out
        assert code == 0
        assert "Recall@20" in out and "MRR@10" in out and "NDCG@10" in out and "queries 2" in out

    def test_missing_gold_errors(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        config, heldout, _ = _eval_data(tmp_path)
        code = lp_eval.main(["--config", str(config), "--heldout", str(heldout),
                             "--gold", str(tmp_path / "no.jsonl")])
        assert code == 1 and "error:" in capsys.readouterr().err

    def test_bad_k_errors(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        config, heldout, gold = _eval_data(tmp_path)
        code = lp_eval.main(["--config", str(config), "--heldout", str(heldout),
                             "--gold", str(gold), "--k", "0"])
        assert code == 1 and "error:" in capsys.readouterr().err

    def test_journal_reports_grounding_rate(self, tmp_path: Path,
                                            capsys: pytest.CaptureFixture) -> None:
        config, heldout, gold = _eval_data(tmp_path)
        journal = tmp_path / "j.jsonl"
        good = Record(record_id="q1", source="W jakim trybie zamawiający udziela zamówienia?",
                      output=json.dumps({"answer": "przetarg nieograniczony",
                                         "citations": ["ref-1"]}), status=Status.VERIFIED)
        bad = Record(record_id="q2", source="Do kogo wnosi się odwołanie?",
                     output=json.dumps({"answer": "gdzieś", "citations": ["ref-999"]}),
                     status=Status.VERIFIED)
        journal.write_text(good.to_json() + "\n" + bad.to_json() + "\n", encoding="utf-8")
        code = lp_eval.main(["--config", str(config), "--heldout", str(heldout),
                             "--gold", str(gold), "--journal", str(journal)])
        out = capsys.readouterr().out
        assert code == 0 and "citation-grounding 1/2" in out


# --- end to end: answer from the legal-passage memory over each retriever ------------------------
def _factory(answer: dict) -> Callable[[str, float], httpx.Client]:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        props = body.get("response_format", {}).get("json_schema", {}).get(
            "schema", {}).get("properties", {})
        content = (json.dumps({"acceptable": True, "issues": []}) if "acceptable" in props
                   else json.dumps(answer))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {}})

    def factory(_b: str, _t: float) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler))
    return factory


def _staged(tmp_path: Path) -> Path:
    config = tmp_path / "config"
    if not config.exists():
        shutil.copytree(CONFIG, config)
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    (data / "passages.jsonl").write_text(
        json.dumps({"id": "ref-1", "text": PASSAGE1}, ensure_ascii=False) + "\n"
        + json.dumps({"id": "ref-2", "text": PASSAGE2}, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return config


def _catalog(tmp_path: Path) -> SqliteRunStore:
    store = SqliteRunStore(str(tmp_path / "run.db"))
    store.add_records([Record(record_id="q1", source=QUESTION, meta={})])
    return store


class TestEndToEnd:
    def _run(self, tmp_path: Path, answer: dict) -> Record:
        config = _staged(tmp_path)
        assembled = assemble(config, client_factory=_factory(answer))
        store = _catalog(tmp_path)
        run_batch(assembled.harness, store.pending(), store, install_signal_handlers=False)
        [result] = list(store.results())
        return result.record

    def test_a_grounded_answer_verifies(self, tmp_path: Path) -> None:
        result = self._run(tmp_path, {
            "answer": "Zamawiający udziela zamówienia w trybie przetargu nieograniczonego.",
            "citations": ["ref-1"]})
        assert result.status is Status.VERIFIED
        assert json.loads(result.output or "{}")["citations"] == ["ref-1"]

    def test_a_fabricated_citation_is_rejected(self, tmp_path: Path) -> None:
        result = self._run(tmp_path, {
            "answer": "Zamawiający udziela zamówienia w trybie negocjacji.",
            "citations": ["ref-999"]})
        assert result.status is Status.REJECTED

    def test_an_uncited_substantive_answer_is_rejected(self, tmp_path: Path) -> None:
        result = self._run(tmp_path, {"answer": "Przetarg nieograniczony.", "citations": []})
        assert result.status is Status.REJECTED

    def test_an_abstention_verifies(self, tmp_path: Path) -> None:
        result = self._run(tmp_path, {"answer": "brak podstaw", "citations": []})
        assert result.status is Status.VERIFIED

    def test_the_passage_reaches_the_prompt(self, tmp_path: Path) -> None:
        config = _staged(tmp_path)
        seen: list[str] = []

        def factory(_b: str, _t: float) -> httpx.Client:
            def handler(request: httpx.Request) -> httpx.Response:
                body = json.loads(request.content)
                seen.append(body["messages"][-1]["content"])
                props = body.get("response_format", {}).get("json_schema", {}).get(
                    "schema", {}).get("properties", {})
                content = (json.dumps({"acceptable": True, "issues": []}) if "acceptable" in props
                           else json.dumps({"answer": "przetarg nieograniczony",
                                            "citations": ["ref-1"]}))
                return httpx.Response(200, json={"choices": [
                    {"message": {"content": content}, "finish_reason": "stop"}], "usage": {}})
            return httpx.Client(transport=httpx.MockTransport(handler))

        assembled = assemble(config, client_factory=factory)
        store = _catalog(tmp_path)
        run_batch(assembled.harness, store.pending(), store, install_signal_handlers=False)
        assert "przetargu nieograniczonego" in seen[0]  # the retrieved legal passage
        assert "Relevant legal passages:" in seen[0]    # the retrieved block heading


_VEC_MODELS = """
[endpoint.local]
base_url = "http://127.0.0.1:8080/v1"
[model.author]
endpoint = "local"
model_id = "a"
[model.reviewer]
endpoint = "local"
model_id = "r"
[model.embedder]
endpoint = "local"
model_id = "e"
kind = "embedding"
"""


def _embed(text: str) -> list[float]:
    return [1.0, (len(text) % 5) / 5.0, (sum(map(ord, text)) % 7) / 7.0]


def _vector_factory(answer: dict) -> Callable[[str, float], httpx.Client]:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={"data": [{"embedding": _embed(t)}
                                                      for t in body["input"]]})
        props = body.get("response_format", {}).get("json_schema", {}).get(
            "schema", {}).get("properties", {})
        content = (json.dumps({"acceptable": True, "issues": []}) if "acceptable" in props
                   else json.dumps(answer))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {}})

    def factory(_b: str, _t: float) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler))
    return factory


@pytest.mark.parametrize("vector_driver", ["lancedb", "qdrant"])
class TestPassageMemoryOnEachVectorDB:
    """The legal-passage memory is retrieved through dense retrieval over EITHER real vector DB
    (lancedb, qdrant); a grounded answer citing a retrieved passage VERIFIES. Only a storage.toml
    driver edit differs. (The default sqlite pairing-store path is covered by TestEndToEnd.)"""

    def test_a_grounded_answer_verifies(self, tmp_path: Path, vector_driver: str) -> None:
        config = _staged(tmp_path)
        (config / "models.toml").write_text(_VEC_MODELS, encoding="utf-8")
        (config / "storage.toml").write_text(
            f'[vector]\ndriver = "{vector_driver}"\npath = "../data/v.{vector_driver}"\ndim = 3\n'
            f'[pairings]\ndriver = "sqlite"\npath = "../data/passages.pairings.db"\n',
            encoding="utf-8")
        (config / "retrieval.toml").write_text(
            '[retrieval]\nkind = "dense"\n[retrieval.dense]\nmodel = "embedder"\n',
            encoding="utf-8")
        answer = {"answer": "Zamawiający udziela zamówienia w trybie przetargu nieograniczonego.",
                  "citations": ["ref-1"]}
        assembled = assemble(config, client_factory=_vector_factory(answer))
        from ragkit.retrieve.retrievers import DenseRetriever
        assert isinstance(assembled.retriever, DenseRetriever)
        store = _catalog(tmp_path)
        run_batch(assembled.harness, store.pending(), store, install_signal_handlers=False)
        [result] = list(store.results())
        assert result.record.status is Status.VERIFIED
