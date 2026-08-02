"""The NL->SQL recipe: the generated-SQL safety validator (adversarial), and an end-to-end run
where a dangerous generation is rejected rather than executed."""
from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from ragkit.eval.gold import EvalError

from ragkit.cli.app import assemble
from ragkit.core.records import Record, Status
from ragkit.harness import run_batch
from ragkit.store.run.sqlite import SqliteRunStore
from ragkit.store.sql.duckdb import DuckDBStore
from ragkit.store.sql.sqlite import SqliteIntrospector, SqliteStore

from recipes.nl_to_sql import eval as sqleval
from recipes.nl_to_sql.plugins.validators import SqlSafetyValidator

CONFIG = Path(__file__).resolve().parents[1] / "config"

SCHEMA = ("CREATE TABLE singer(id INTEGER PRIMARY KEY, name TEXT, country TEXT);"
          "CREATE TABLE concert(id INTEGER PRIMARY KEY, year INT);"
          "INSERT INTO singer(id, name, country) VALUES (1, 'Ada', 'PL');")


def _db(tmp_path: Path) -> Path:
    path = tmp_path / "database.sqlite"
    if not path.exists():  # a single test may build its context more than once
        SqliteStore(str(path), schema_sql=SCHEMA).close()
    return path


def _context(tmp_path: Path) -> dict[str, object]:
    path = _db(tmp_path)
    return {"introspector": SqliteIntrospector(str(path)),
            "sql_store": SqliteStore(str(path), read_only=True)}


def _rec() -> Record:
    return Record(record_id="1", source="how many singers are there?")


class TestSafetyRefusals:
    """Every attack must produce a blocking sql_unsafe violation, never a clean pass."""

    def _refused(self, sql: str, tmp_path: Path) -> bool:
        vs = SqlSafetyValidator().validate(_rec(), sql, _context(tmp_path))
        return any(v.rule_id == "sql_unsafe" and v.blocking for v in vs)

    def test_drop_table(self, tmp_path: Path) -> None:
        assert self._refused("DROP TABLE singer", tmp_path)

    def test_delete(self, tmp_path: Path) -> None:
        assert self._refused("DELETE FROM singer WHERE id = 1", tmp_path)

    def test_update(self, tmp_path: Path) -> None:
        assert self._refused("UPDATE singer SET name = 'x'", tmp_path)

    def test_stacked_statement_injection(self, tmp_path: Path) -> None:
        assert self._refused("SELECT * FROM singer; DROP TABLE singer", tmp_path)

    def test_comment_smuggled_keyword(self, tmp_path: Path) -> None:
        # A DROP hidden after a line comment must not slip past: comments are stripped first, and
        # the remaining statement is still not a lone SELECT.
        assert self._refused("SELECT 1 -- harmless\n; DROP TABLE singer", tmp_path)

    def test_unknown_table(self, tmp_path: Path) -> None:
        assert self._refused("SELECT * FROM secret_users", tmp_path)

    def test_pragma_and_attach(self, tmp_path: Path) -> None:
        assert self._refused("SELECT 1; ATTACH DATABASE 'x' AS y", tmp_path)
        assert self._refused("PRAGMA table_info(singer)", tmp_path)

    def test_inline_forbidden_keyword_in_single_statement(self, tmp_path: Path) -> None:
        # One statement, starting with SELECT, but a DML keyword smuggled mid-query (no semicolon
        # to split on) — the keyword scan is the defence here, not the statement count.
        assert self._refused("SELECT * FROM singer WHERE 1=1 AND TRUNCATE", tmp_path)

    def test_syntactically_broken_sql(self, tmp_path: Path) -> None:
        # EXPLAIN catches an unknown column the text checks miss.
        assert self._refused("SELECT no_such_column FROM singer", tmp_path)


class TestSafetyAcceptance:
    def test_a_plain_select_passes(self, tmp_path: Path) -> None:
        assert SqlSafetyValidator().validate(_rec(), "SELECT count(*) FROM singer",
                                             _context(tmp_path)) == []

    def test_a_join_over_known_tables_passes(self, tmp_path: Path) -> None:
        sql = "SELECT s.name FROM singer AS s JOIN concert AS c ON c.year = s.id"
        assert SqlSafetyValidator().validate(_rec(), sql, _context(tmp_path)) == []

    def test_a_cte_is_not_flagged_as_an_unknown_table(self, tmp_path: Path) -> None:
        sql = "WITH recent AS (SELECT * FROM concert) SELECT count(*) FROM recent"
        assert SqlSafetyValidator().validate(_rec(), sql, _context(tmp_path)) == []

    def test_trailing_semicolon_is_accepted(self, tmp_path: Path) -> None:
        # A single terminated statement leaves an empty segment after the final ';' — that segment
        # must not be counted as a second statement.
        assert SqlSafetyValidator().validate(_rec(), "SELECT count(*) FROM singer;",
                                             _context(tmp_path)) == []

    def test_empty_output_abstains(self, tmp_path: Path) -> None:
        assert SqlSafetyValidator().validate(_rec(), "   ", _context(tmp_path)) == []

    def test_works_without_a_wired_schema(self) -> None:
        # No introspector/store: the syntactic checks still run; the table-existence and EXPLAIN
        # checks simply do not apply.
        assert SqlSafetyValidator(explain=False).validate(_rec(), "SELECT 1", {}) == []
        assert SqlSafetyValidator().validate(_rec(), "DROP TABLE t", {})  # still refused

    def test_from_config(self) -> None:
        assert SqlSafetyValidator.from_config({"explain": False})._explain is False


def _factory(sql: str) -> Callable[[str, float], httpx.Client]:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        schema = body.get("response_format", {}).get("json_schema", {}).get("schema", {})
        content = (json.dumps({"acceptable": True, "issues": []})
                   if "acceptable" in schema.get("properties", {}) else json.dumps({"sql": sql}))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {}})

    def factory(_b: str, _t: float) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler))
    return factory


def _staged_config(tmp_path: Path) -> Path:
    config = tmp_path / "config"
    shutil.copytree(CONFIG, config)
    (tmp_path / "data").mkdir()
    SqliteStore(str(tmp_path / "data" / "database.sqlite"), schema_sql=SCHEMA).close()
    return config


class TestEndToEnd:
    def test_a_safe_query_verifies(self, tmp_path: Path) -> None:
        config = _staged_config(tmp_path)
        assembled = assemble(config, client_factory=_factory("SELECT count(*) FROM singer"))
        store = SqliteRunStore(str(tmp_path / "run.db"))
        store.add_records([Record(record_id="1", source="how many singers?")])
        run_batch(assembled.harness, store.pending(), store, install_signal_handlers=False)
        [result] = list(store.results())
        assert result.record.status is Status.VERIFIED
        assert "SELECT" in (result.record.output or "")

    def test_a_destructive_generation_is_rejected_not_executed(self, tmp_path: Path) -> None:
        config = _staged_config(tmp_path)
        # The model is made to emit a DROP every time; it is rejected and never runs. The row
        # count in the database is unchanged afterward.
        assembled = assemble(config, client_factory=_factory("DROP TABLE singer"))
        store = SqliteRunStore(str(tmp_path / "run.db"))
        store.add_records([Record(record_id="1", source="delete all singers")])
        run_batch(assembled.harness, store.pending(), store, install_signal_handlers=False)
        [result] = list(store.results())
        assert result.record.status is Status.REJECTED
        # The table still exists and still has its row.
        check = SqliteStore(str(tmp_path / "data" / "database.sqlite"), read_only=True)
        assert check.query("SELECT count(*) AS n FROM singer")[0]["n"] == 1

    def test_the_external_store_is_read_only_at_the_port(self, tmp_path: Path) -> None:
        config = _staged_config(tmp_path)
        assembled = assemble(config, client_factory=_factory("SELECT 1"))
        from ragkit.store.sql.sqlite import SqlStoreError
        assert assembled.harness.sql_store is not None
        with pytest.raises(SqlStoreError, match="read_only"):
            assembled.harness.sql_store.execute("DROP TABLE singer")


EVAL_SCHEMA = (
    "CREATE TABLE singer(id INTEGER PRIMARY KEY, name TEXT, country TEXT);"
    "INSERT INTO singer(id, name, country) VALUES (1, 'Ada', 'PL'), (2, 'Bo', 'US'),"
    " (3, 'Cy', 'PL');")


def _eval_store(tmp_path: Path) -> SqliteStore:
    path = tmp_path / "eval.sqlite"
    SqliteStore(str(path), schema_sql=EVAL_SCHEMA).close()
    return SqliteStore(str(path), read_only=True)


class TestExecutionAccuracy:
    def test_identical_query_matches(self, tmp_path: Path) -> None:
        store = _eval_store(tmp_path)
        report = sqleval.evaluate([("q1", "SELECT count(*) FROM singer",
                                    "SELECT count(*) FROM singer")], store)
        assert report.accuracy == 1.0 and report.matched == 1 and report.executed == 1

    def test_equivalent_query_matches_by_result_not_string(self, tmp_path: Path) -> None:
        store = _eval_store(tmp_path)
        report = sqleval.evaluate([("q1", "SELECT count(id) FROM singer",
                                    "SELECT count(*) FROM singer")], store)
        assert report.matched == 1

    def test_row_order_ignored_without_order_by(self, tmp_path: Path) -> None:
        store = _eval_store(tmp_path)
        produced = "SELECT name FROM singer WHERE country='PL' ORDER BY id DESC"
        gold = "SELECT name FROM singer WHERE country='PL'"
        report = sqleval.evaluate([("q1", produced, gold)], store)
        # Gold has no ORDER BY, so the two are compared as a multiset: order does not matter.
        assert report.matched == 1

    def test_row_order_enforced_when_gold_orders(self, tmp_path: Path) -> None:
        store = _eval_store(tmp_path)
        report = sqleval.evaluate([("q1", "SELECT name FROM singer ORDER BY id DESC",
                                    "SELECT name FROM singer ORDER BY id ASC")], store)
        # Gold has ORDER BY, so a differently-ordered result is a miss.
        assert report.matched == 0 and report.executed == 1

    def test_wrong_result_is_a_miss(self, tmp_path: Path) -> None:
        store = _eval_store(tmp_path)
        report = sqleval.evaluate([("q1", "SELECT count(*) FROM singer WHERE country='PL'",
                                    "SELECT count(*) FROM singer")], store)
        assert report.matched == 0 and "differs" in report.outcomes[0].detail

    def test_no_produced_output_is_a_miss(self, tmp_path: Path) -> None:
        store = _eval_store(tmp_path)
        report = sqleval.evaluate([("q1", None, "SELECT count(*) FROM singer")], store)
        assert report.matched == 0 and report.outcomes[0].produced is False

    def test_unrunnable_produced_query_is_a_miss_not_a_crash(self, tmp_path: Path) -> None:
        store = _eval_store(tmp_path)
        report = sqleval.evaluate([("q1", "SELECT nope FROM singer",
                                    "SELECT count(*) FROM singer")], store)
        assert report.matched == 0 and report.outcomes[0].executed is False

    def test_bad_gold_is_a_configuration_error(self, tmp_path: Path) -> None:
        store = _eval_store(tmp_path)
        with pytest.raises(EvalError, match="gold SQL"):
            sqleval.evaluate([("q1", "SELECT 1", "SELECT bad FROM nowhere")], store)

    def test_accuracy_of_empty_report_is_zero(self) -> None:
        assert sqleval.Report(()).accuracy == 0.0


class TestEvalMain:
    def _setup(self, tmp_path: Path, produced_sql: str, status: Status) -> tuple[Path, Path, Path]:
        config = tmp_path / "config"
        config.mkdir()
        db = tmp_path / "eval.sqlite"
        SqliteStore(str(db), schema_sql=EVAL_SCHEMA).close()
        (config / "storage.toml").write_text(
            f'[sql]\ndriver = "sqlite"\npath = "{db}"\nread_only = true\n', encoding="utf-8")
        journal = tmp_path / "j.jsonl"
        rec = Record(record_id="q1", source="how many?", output=produced_sql, status=status)
        journal.write_text(rec.to_json() + "\n", encoding="utf-8")
        gold = tmp_path / "gold.jsonl"
        gold.write_text(
            json.dumps({"record_id": "q1", "sql": "SELECT count(*) FROM singer"}) + "\n",
            encoding="utf-8")
        return config, journal, gold

    @staticmethod
    def _run(config: Path, journal: Path, gold: Path) -> int:
        return sqleval.main(["--config", str(config), "--journal", str(journal),
                             "--gold", str(gold)])

    def test_end_to_end_success(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        args = self._setup(tmp_path, "SELECT count(*) FROM singer", Status.VERIFIED)
        assert self._run(*args) == 0 and "1/1 = 1.000" in capsys.readouterr().out

    def test_rejected_record_scores_as_miss(self, tmp_path: Path,
                                            capsys: pytest.CaptureFixture[str]) -> None:
        args = self._setup(tmp_path, "DROP TABLE singer", Status.REJECTED)
        assert self._run(*args) == 0 and "0/1 = 0.000" in capsys.readouterr().out

    def test_missing_gold_errors(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        config, journal, _ = self._setup(tmp_path, "SELECT 1", Status.VERIFIED)
        code = self._run(config, journal, tmp_path / "nope.jsonl")
        assert code == 1 and "error:" in capsys.readouterr().err

    def test_writable_store_is_refused(self, tmp_path: Path,
                                       capsys: pytest.CaptureFixture[str]) -> None:
        config = tmp_path / "config"
        config.mkdir()
        db = tmp_path / "eval.sqlite"
        SqliteStore(str(db), schema_sql=EVAL_SCHEMA).close()
        (config / "storage.toml").write_text(
            f'[sql]\ndriver = "sqlite"\npath = "{db}"\n', encoding="utf-8")
        gold = tmp_path / "gold.jsonl"
        gold.write_text('{"record_id": "q1", "sql": "SELECT 1"}\n', encoding="utf-8")
        code = self._run(config, tmp_path / "j.jsonl", gold)
        assert code == 1 and "read_only" in capsys.readouterr().err

    def test_no_sql_store_configured_errors(self, tmp_path: Path,
                                            capsys: pytest.CaptureFixture[str]) -> None:
        config = tmp_path / "config"
        config.mkdir()
        (config / "storage.toml").write_text(
            '[introspector]\ndriver = "sqlite"\npath = ":memory:"\n', encoding="utf-8")
        gold = tmp_path / "gold.jsonl"
        gold.write_text('{"record_id": "q1", "sql": "SELECT 1"}\n', encoding="utf-8")
        code = self._run(config, tmp_path / "j.jsonl", gold)
        assert code == 1 and "no [sql] store" in capsys.readouterr().err


_SQL_DRIVERS = [(SqliteStore, "sqlite", "database.sqlite"),
                (DuckDBStore, "duckdb", "database.duckdb")]


@pytest.mark.parametrize(("store_cls", "driver", "filename"), _SQL_DRIVERS)
class TestDatabaseSwap:
    """The recipe runs end-to-end against BOTH real SqlStore drivers (sqlite and duckdb) with only
    a storage.toml driver edit -- proving the external database is swappable without code change."""

    def _staged(self, tmp_path: Path, store_cls: type[SqliteStore] | type[DuckDBStore],
               driver: str, filename: str) -> Path:
        config = tmp_path / "config"
        shutil.copytree(CONFIG, config)
        (tmp_path / "data").mkdir()
        store_cls(str(tmp_path / "data" / filename), schema_sql=SCHEMA).close()
        (config / "storage.toml").write_text(
            f'[sql]\ndriver = "{driver}"\npath = "../data/{filename}"\nread_only = true\n'
            f'[introspector]\ndriver = "{driver}"\npath = "../data/{filename}"\n', encoding="utf-8")
        return config

    def test_a_safe_query_verifies(self, tmp_path: Path,
                                   store_cls: type[SqliteStore] | type[DuckDBStore], driver: str,
                                   filename: str) -> None:
        config = self._staged(tmp_path, store_cls, driver, filename)
        assembled = assemble(config, client_factory=_factory("SELECT count(*) FROM singer"))
        store = SqliteRunStore(str(tmp_path / "run.db"))
        store.add_records([Record(record_id="1", source="how many singers?")])
        run_batch(assembled.harness, store.pending(), store, install_signal_handlers=False)
        [result] = list(store.results())
        assert result.record.status is Status.VERIFIED
        assert "SELECT" in (result.record.output or "")

    def test_a_destructive_generation_is_rejected(
            self, tmp_path: Path, store_cls: type[SqliteStore] | type[DuckDBStore], driver: str,
            filename: str) -> None:
        config = self._staged(tmp_path, store_cls, driver, filename)
        assembled = assemble(config, client_factory=_factory("DROP TABLE singer"))
        store = SqliteRunStore(str(tmp_path / "run.db"))
        store.add_records([Record(record_id="1", source="delete all")])
        run_batch(assembled.harness, store.pending(), store, install_signal_handlers=False)
        [result] = list(store.results())
        assert result.record.status is Status.REJECTED
        # The read-only binding on the chosen engine still holds the row.
        check = store_cls(str(tmp_path / "data" / filename), read_only=True)
        assert check.query("SELECT count(*) AS n FROM singer")[0]["n"] == 1
