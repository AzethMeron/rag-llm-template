"""RuleSet loading across the three policy tiers."""
from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.core.config import ConfigError
from ragkit.core.rules import Severity
from ragkit.harness.rules import RuleSet

RULES = """
[limits]
max_line_columns = 100
max_columns_tolerance = 0.1
require_nonempty = true
keep_flagged_rules = ["line_width"]

[lexicon]
severity = "error"

[style]
directives = ["be concise", "stay formal"]

[[forbidden]]
pattern = "^(here is|sure)"
reason = "conversational preamble"

[[advisory]]
id = "register"
description = "keep the register consistent"
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "rules.toml"
    path.write_text(text, encoding="utf-8")
    return path


class TestLoad:
    def test_loads_all_tiers(self, tmp_path: Path) -> None:
        rs = RuleSet.load(_write(tmp_path, RULES))
        assert rs.max_line_columns == 100 and rs.max_columns_tolerance == 0.1
        assert rs.keep_flagged_rules == frozenset({"line_width"})
        assert rs.lexicon_severity is Severity.ERROR
        assert rs.style_directives == ("be concise", "stay formal")
        assert rs.forbidden_patterns == (("^(here is|sure)", "conversational preamble"),)
        assert rs.advisory_rules == (("register", "keep the register consistent"),)

    def test_defaults_when_absent(self, tmp_path: Path) -> None:
        rs = RuleSet.load(_write(tmp_path, ""))
        assert rs.max_line_columns == 110 and rs.lexicon_severity is Severity.WARNING

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="rules file not found"):
            RuleSet.load(tmp_path / "absent.toml")

    def test_invalid_toml(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="invalid TOML"):
            RuleSet.load(_write(tmp_path, "= x"))

    def test_unknown_key(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="unknown key"):
            RuleSet.load(_write(tmp_path, "[limits]\nbogus = 1\n"))

    def test_bad_max_line_columns_type(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="must be an integer"):
            RuleSet.load(_write(tmp_path, "[limits]\nmax_line_columns = true\n"))

    def test_implausibly_small_width(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="implausibly small"):
            RuleSet.load(_write(tmp_path, "[limits]\nmax_line_columns = 5\n"))

    def test_tolerance_out_of_range(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="fraction in"):
            RuleSet.load(_write(tmp_path, "[limits]\nmax_columns_tolerance = 2.0\n"))

    def test_forbidden_missing_pattern(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="non-empty 'pattern'"):
            RuleSet.load(_write(tmp_path, "[[forbidden]]\nreason = 'x'\n"))

    def test_forbidden_invalid_regex(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not a valid regex"):
            RuleSet.load(_write(tmp_path, "[[forbidden]]\npattern = '([unclosed'\n"))

    def test_advisory_missing_fields(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="needs 'id' and 'description'"):
            RuleSet.load(_write(tmp_path, "[[advisory]]\nid = 'x'\n"))

    def test_bad_lexicon_severity(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="must be 'error' or 'warning'"):
            RuleSet.load(_write(tmp_path, "[lexicon]\nseverity = 'critical'\n"))


class TestDirect:
    def test_tolerance_bound(self) -> None:
        with pytest.raises(ValueError, match="fraction"):
            RuleSet(max_columns_tolerance=-0.1)
