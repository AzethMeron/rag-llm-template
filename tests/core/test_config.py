"""The strict TOML loader primitives: unknown keys refused, bool-vs-number refused, the
``[[section]]``/``[section]`` mistake named, and range checks deliberately left to the caller."""
from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.core.config import (
    ConfigError,
    as_table,
    load_toml,
    read_bool,
    read_float,
    read_int,
    read_string,
    read_string_list,
    reject_unknown,
    tables,
)


class TestLoadToml:
    def test_reads_valid_toml(self, tmp_path: Path) -> None:
        path = tmp_path / "c.toml"
        path.write_text("[a]\nx = 1\n", encoding="utf-8")
        assert load_toml(path) == {"a": {"x": 1}}

    def test_missing_file_is_named(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_toml(tmp_path / "none.toml")

    def test_missing_file_uses_the_what_label(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="rules file not found"):
            load_toml(tmp_path / "none.toml", what="rules file")

    def test_invalid_toml_is_named(self, tmp_path: Path) -> None:
        path = tmp_path / "c.toml"
        path.write_text("= broken", encoding="utf-8")
        with pytest.raises(ConfigError, match="invalid TOML"):
            load_toml(path)


class TestAsTable:
    def test_accepts_a_dict(self) -> None:
        assert as_table({"a": 1}, label="[x]") == {"a": 1}

    def test_refuses_a_scalar(self) -> None:
        with pytest.raises(ConfigError, match=r"\[x\] must be a table"):
            as_table(5, label="[x]")


class TestRejectUnknown:
    def test_accepts_known_keys(self) -> None:
        assert reject_unknown({"a": 1}, {"a", "b"}, label="[x]") == {"a": 1}

    def test_refuses_unknown_key_naming_the_allowed_set(self) -> None:
        with pytest.raises(ConfigError, match=r"unknown key\(s\) \['c'\].*allowed keys"):
            reject_unknown({"c": 1}, {"a", "b"}, label="[x]")

    def test_accepts_a_frozenset_of_allowed_keys(self) -> None:
        assert reject_unknown({"a": 1}, frozenset({"a"}), label="[x]") == {"a": 1}

    def test_non_table_section_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="must be a table"):
            reject_unknown(5, {"a"}, label="[x]")


class TestTables:
    def test_returns_array_of_tables(self) -> None:
        data = {"reviewer": [{"id": "a"}, {"id": "b"}]}
        assert tables(data, "reviewer") == [{"id": "a"}, {"id": "b"}]

    def test_absent_key_is_empty(self) -> None:
        assert tables({}, "reviewer") == []

    def test_single_table_instead_of_array_is_named(self) -> None:
        with pytest.raises(ConfigError, match=r"write \[\[reviewer\]\], not \[reviewer\]"):
            tables({"reviewer": {"id": "a"}}, "reviewer")

    def test_non_table_entry_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="entry 0 must be a table"):
            tables({"reviewer": ["a string"]}, "reviewer")


class TestReadInt:
    def test_reads_an_int(self) -> None:
        assert read_int({"n": 5}, "n", 0, label="[x]") == 5

    def test_uses_default_when_absent(self) -> None:
        assert read_int({}, "n", 7, label="[x]") == 7

    def test_bool_is_refused_not_coerced(self) -> None:
        with pytest.raises(ConfigError, match="must be an integer"):
            read_int({"n": True}, "n", 0, label="[x]")

    def test_string_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="must be an integer"):
            read_int({"n": "5"}, "n", 0, label="[x]")


class TestReadFloat:
    def test_reads_a_float(self) -> None:
        assert read_float({"f": 0.5}, "f", 0.0, label="[x]") == 0.5

    def test_widens_an_int(self) -> None:
        value = read_float({"f": 1}, "f", 0.0, label="[x]")
        assert value == 1.0 and isinstance(value, float)

    def test_uses_default_when_absent(self) -> None:
        assert read_float({}, "f", 0.3, label="[x]") == 0.3

    def test_bool_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="must be a number"):
            read_float({"f": True}, "f", 0.0, label="[x]")

    def test_string_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="must be a number"):
            read_float({"f": "x"}, "f", 0.0, label="[x]")


class TestReadBool:
    def test_reads_a_bool(self) -> None:
        assert read_bool({"b": True}, "b", False, label="[x]") is True

    def test_default_when_absent(self) -> None:
        assert read_bool({}, "b", True, label="[x]") is True

    def test_non_bool_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="must be a boolean"):
            read_bool({"b": 1}, "b", False, label="[x]")


class TestReadString:
    def test_reads_a_string(self) -> None:
        assert read_string({"s": "hi"}, "s", "", label="[x]") == "hi"

    def test_default_when_absent(self) -> None:
        assert read_string({}, "s", "d", label="[x]") == "d"

    def test_non_string_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="must be a string"):
            read_string({"s": 5}, "s", "", label="[x]")


class TestReadStringList:
    def test_reads_a_list(self) -> None:
        assert read_string_list({"xs": ["a", "b"]}, "xs", label="[x]") == ("a", "b")

    def test_absent_is_empty(self) -> None:
        assert read_string_list({}, "xs", label="[x]") == ()

    def test_bare_string_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="must be an array of strings"):
            read_string_list({"xs": "abc"}, "xs", label="[x]")

    def test_non_string_entry_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="must be an array of strings"):
            read_string_list({"xs": ["a", 2]}, "xs", label="[x]")
