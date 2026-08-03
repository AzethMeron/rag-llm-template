"""The form-autofill recipe: the field-value validator, the fill-accuracy eval, and an end-to-end
run that fills a track's genre/price from its album siblings (and rejects an illegal fill)."""
from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from ragkit.cli.app import assemble
from ragkit.core.records import Record, Status
from ragkit.core.rules import Violation
from ragkit.harness import run_batch
from ragkit.store.run.sqlite import SqliteRunStore
from ragkit.store.sql.duckdb import DuckDBStore
from ragkit.store.sql.sqlite import SqliteStore

from recipes.form_autofill import eval as fill_eval
from recipes.form_autofill.plugins.validators import FieldRule, FieldTypesValidator

CONFIG = Path(__file__).resolve().parents[1] / "config"

FIXTURE = """
CREATE TABLE Artist(ArtistId INTEGER PRIMARY KEY, Name TEXT);
CREATE TABLE Album(AlbumId INTEGER PRIMARY KEY, Title TEXT, ArtistId INTEGER);
CREATE TABLE Genre(GenreId INTEGER PRIMARY KEY, Name TEXT);
CREATE TABLE Track(TrackId INTEGER PRIMARY KEY, Name TEXT, AlbumId INTEGER, GenreId INTEGER,
                   Composer TEXT, Milliseconds INTEGER, UnitPrice REAL);
INSERT INTO Artist VALUES (1, 'AC/DC');
INSERT INTO Genre VALUES (1, 'Rock'), (2, 'Jazz');
INSERT INTO Album VALUES (1, 'High Voltage', 1);
INSERT INTO Track VALUES
  (1, 'Song A', 1, 1, 'Young', 200000, 0.99),
  (2, 'Song B', 1, 1, 'Young', 210000, 0.99),
  (3, 'Song C', 1, 1, 'Young', 220000, 0.99);
"""

GENRE_PRICE = (FieldRule(name="genre", type="string", nonempty=True),
               FieldRule(name="unit_price", type="number", min_exclusive=0.0))


def _rec() -> Record:
    return Record(record_id="track-1", source="Track 'Song A'")


def _validate(form: dict[str, object]) -> list[Violation]:
    return FieldTypesValidator(GENRE_PRICE).validate(_rec(), json.dumps(form), {})


class TestFieldValidator:
    def test_a_well_filled_form_passes(self) -> None:
        assert _validate({"genre": "Rock", "unit_price": 0.99}) == []

    def test_null_field_refused(self) -> None:
        vs = _validate({"genre": None, "unit_price": 0.99})
        assert any(v.rule_id == "form_field" and "not filled" in v.message and v.blocking
                   for v in vs)

    def test_wrong_type_refused(self) -> None:
        assert any("should be number" in v.message for v in _validate(
            {"genre": "Rock", "unit_price": "cheap"}))

    def test_blank_string_refused(self) -> None:
        assert any("is blank" in v.message for v in _validate({"genre": "  ", "unit_price": 0.99}))

    def test_nonpositive_price_refused(self) -> None:
        assert any("greater than 0" in v.message for v in _validate(
            {"genre": "Rock", "unit_price": 0}))

    def test_range_min_and_max(self) -> None:
        rule = FieldRule(name="n", type="number", minimum=1.0, maximum=10.0)
        assert any("below the minimum" in v.message for v in rule.check(0.5))
        assert any("above the maximum" in v.message for v in rule.check(11))
        assert rule.check(5) == []

    def test_enum_membership_case_insensitive(self) -> None:
        rule = FieldRule(name="g", type="string", enum=("Rock", "Jazz"))
        assert rule.check("rock") == []  # case-insensitive
        assert any("not one of the allowed" in v.message for v in rule.check("Techno"))

    def test_enum_on_a_number(self) -> None:
        rule = FieldRule(name="n", type="integer", enum=("1", "2"))
        assert rule.check(1) == []
        assert any("not one of the allowed" in v.message for v in rule.check(3))

    def test_boolean_and_integer_types(self) -> None:
        assert FieldRule(name="b", type="boolean").check(True) == []
        assert any("should be boolean" in v.message
                   for v in FieldRule(name="b", type="boolean").check("yes"))
        # A bool must not satisfy integer/number (bool is an int subclass in Python).
        assert any("should be integer" in v.message
                   for v in FieldRule(name="i", type="integer").check(True))

    def test_invalid_json_refused(self) -> None:
        vs = FieldTypesValidator(GENRE_PRICE).validate(_rec(), "{not json", {})
        assert any("not valid JSON" in v.message for v in vs)

    def test_non_object_json_refused(self) -> None:
        vs = FieldTypesValidator(GENRE_PRICE).validate(_rec(), "[1, 2]", {})
        assert any("must be a JSON object" in v.message for v in vs)

    def test_bad_field_type_in_rule_refused(self) -> None:
        with pytest.raises(ValueError, match="type must be one of"):
            FieldRule(name="x", type="date")

    def test_from_config_builds_rules(self) -> None:
        validator = FieldTypesValidator.from_config({"field": [
            {"name": "genre", "nonempty": True},
            {"name": "unit_price", "type": "number", "min_exclusive": 0}]})
        form = json.dumps({"genre": "Rock", "unit_price": 0.99})
        assert validator.validate(_rec(), form, {}) == []

    def test_from_config_needs_fields(self) -> None:
        with pytest.raises(ValueError, match="non-empty 'field'"):
            FieldTypesValidator.from_config({})

    def test_from_config_field_needs_name(self) -> None:
        with pytest.raises(ValueError, match="needs at least a 'name'"):
            FieldTypesValidator.from_config({"field": [{"type": "string"}]})

    def test_from_config_rejects_non_numeric_constraint(self) -> None:
        with pytest.raises(ValueError, match="must be a number"):
            FieldTypesValidator.from_config(
                {"field": [{"name": "n", "type": "number", "min": "x"}]})

    def test_empty_rules_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one field rule"):
            FieldTypesValidator([])


class TestFillAccuracy:
    def _gold(self) -> dict[str, dict[str, object]]:
        return {"t1": {"genre": "Rock", "unit_price": 0.99}}

    def test_exact_fill_scores_one(self) -> None:
        report = fill_eval.evaluate([("t1", json.dumps({"genre": "Rock", "unit_price": 0.99}),
                                      self._gold()["t1"])])
        assert report.both_accuracy == 1.0 and report.produced == 1

    def test_genre_case_and_whitespace_insensitive(self) -> None:
        report = fill_eval.evaluate([("t1", json.dumps({"genre": " rock ", "unit_price": 0.99}),
                                      self._gold()["t1"])])
        assert report.genre_accuracy == 1.0

    def test_price_tolerance(self) -> None:
        report = fill_eval.evaluate([("t1", json.dumps({"genre": "Rock", "unit_price": 0.9899}),
                                      self._gold()["t1"])])
        assert report.price_accuracy == 1.0

    def test_wrong_values_score_zero(self) -> None:
        report = fill_eval.evaluate([("t1", json.dumps({"genre": "Jazz", "unit_price": 1.99}),
                                      self._gold()["t1"])])
        assert report.genre_accuracy == 0.0 and report.price_accuracy == 0.0
        assert report.both_accuracy == 0.0

    def test_no_output_is_a_miss(self) -> None:
        report = fill_eval.evaluate([("t1", None, self._gold()["t1"])])
        assert report.produced == 0 and report.both_accuracy == 0.0

    def test_malformed_output_is_a_miss_not_a_crash(self) -> None:
        report = fill_eval.evaluate([("t1", "{bad", self._gold()["t1"]),
                                     ("t2", "[1,2]", self._gold()["t1"])])
        assert report.both_accuracy == 0.0

    def test_empty_report_rates_are_zero(self) -> None:
        assert fill_eval.Report(()).both_accuracy == 0.0


def _factory(genre: object, price: object) -> Callable[[str, float], httpx.Client]:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        schema = body.get("response_format", {}).get("json_schema", {}).get("schema", {})
        props = schema.get("properties", {})
        content = (json.dumps({"acceptable": True, "issues": []}) if "acceptable" in props
                   else json.dumps({"genre": genre, "unit_price": price}))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {}})

    def factory(_b: str, _t: float) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler))
    return factory


def _staged(tmp_path: Path) -> Path:
    config = tmp_path / "config"
    shutil.copytree(CONFIG, config)
    (tmp_path / "data").mkdir()
    SqliteStore(str(tmp_path / "data" / "chinook.sqlite"), schema_sql=FIXTURE).close()
    return config


def _heldout(tmp_path: Path) -> SqliteRunStore:
    store = SqliteRunStore(str(tmp_path / "run.db"))
    store.add_records([Record(record_id="track-1", source="Track 'Song A' from 'High Voltage'",
                              meta={"album_id": 1, "track_id": 1})])
    return store


class TestEndToEnd:
    def test_a_grounded_fill_verifies(self, tmp_path: Path) -> None:
        config = _staged(tmp_path)
        assembled = assemble(config, client_factory=_factory("Rock", 0.99))
        store = _heldout(tmp_path)
        run_batch(assembled.harness, store.pending(), store, install_signal_handlers=False)
        [result] = list(store.results())
        assert result.record.status is Status.VERIFIED
        assert json.loads(result.record.output or "{}") == {"genre": "Rock", "unit_price": 0.99}

    def test_the_sibling_rows_reach_the_prompt(self, tmp_path: Path) -> None:
        # The sql_rows block must query the album's other tracks (2 and 3), excluding track 1.
        config = _staged(tmp_path)
        seen: list[str] = []

        def factory(_b: str, _t: float) -> httpx.Client:
            def handler(request: httpx.Request) -> httpx.Response:
                body = json.loads(request.content)
                seen.append(body["messages"][-1]["content"])
                props = body.get("response_format", {}).get("json_schema", {}).get(
                    "schema", {}).get("properties", {})
                content = (json.dumps({"acceptable": True, "issues": []}) if "acceptable" in props
                           else json.dumps({"genre": "Rock", "unit_price": 0.99}))
                return httpx.Response(200, json={"choices": [
                    {"message": {"content": content}, "finish_reason": "stop"}], "usage": {}})
            return httpx.Client(transport=httpx.MockTransport(handler))

        assembled = assemble(config, client_factory=factory)
        store = _heldout(tmp_path)
        run_batch(assembled.harness, store.pending(), store, install_signal_handlers=False)
        producer_prompt = seen[0]
        assert "Song B" in producer_prompt and "Song C" in producer_prompt
        assert "Song A" not in producer_prompt.split("Track to fill:")[0]  # the held-out track

    def test_an_illegal_fill_is_rejected(self, tmp_path: Path) -> None:
        # A non-positive price violates the FieldTypesValidator every attempt -> REJECTED.
        config = _staged(tmp_path)
        assembled = assemble(config, client_factory=_factory("Rock", 0))
        store = _heldout(tmp_path)
        run_batch(assembled.harness, store.pending(), store, install_signal_handlers=False)
        [result] = list(store.results())
        assert result.record.status is Status.REJECTED


class TestEvalMain:
    def _setup(self, tmp_path: Path, output: str) -> tuple[Path, Path]:
        journal = tmp_path / "j.jsonl"
        rec = Record(record_id="track-1", source="x", output=output, status=Status.VERIFIED)
        journal.write_text(rec.to_json() + "\n", encoding="utf-8")
        gold = tmp_path / "gold.jsonl"
        gold.write_text(json.dumps({"record_id": "track-1", "genre": "Rock",
                                    "unit_price": 0.99}) + "\n", encoding="utf-8")
        return journal, gold

    def test_success(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        journal, gold = self._setup(tmp_path, json.dumps({"genre": "Rock", "unit_price": 0.99}))
        code = fill_eval.main(["--journal", str(journal), "--gold", str(gold)])
        out = capsys.readouterr().out
        assert code == 0 and "both 1.000" in out

    def test_missing_gold_errors(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        journal, _ = self._setup(tmp_path, json.dumps({"genre": "Rock", "unit_price": 0.99}))
        code = fill_eval.main(["--journal", str(journal), "--gold", str(tmp_path / "no.jsonl")])
        assert code == 1 and "error:" in capsys.readouterr().err


@pytest.mark.parametrize(("store_cls", "driver", "filename"),
                         [(SqliteStore, "sqlite", "chinook.sqlite"),
                          (DuckDBStore, "duckdb", "chinook.duckdb")])
class TestDatabaseSwap:
    """The recipe fills a form from historical records held in EITHER real SqlStore driver (sqlite
    and duckdb) via the sql_rows block -- a one-line storage.toml edit, no code change."""

    def _staged(self, tmp_path: Path, store_cls: type[SqliteStore] | type[DuckDBStore],
               driver: str, filename: str) -> Path:
        config = tmp_path / "config"
        shutil.copytree(CONFIG, config)
        (tmp_path / "data").mkdir()
        store_cls(str(tmp_path / "data" / filename), schema_sql=FIXTURE).close()
        (config / "storage.toml").write_text(
            f'[sql]\ndriver = "{driver}"\npath = "../data/{filename}"\nread_only = true\n',
            encoding="utf-8")
        return config

    def test_a_grounded_fill_verifies(self, tmp_path: Path,
                                      store_cls: type[SqliteStore] | type[DuckDBStore],
                                      driver: str, filename: str) -> None:
        config = self._staged(tmp_path, store_cls, driver, filename)
        assembled = assemble(config, client_factory=_factory("Rock", 0.99))
        store = _heldout(tmp_path)
        run_batch(assembled.harness, store.pending(), store, install_signal_handlers=False)
        [result] = list(store.results())
        assert result.record.status is Status.VERIFIED
        assert json.loads(result.record.output or "{}") == {"genre": "Rock", "unit_price": 0.99}
