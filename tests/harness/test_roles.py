"""Persona/panel loading and the budget/leniency arithmetic."""
from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.core.config import ConfigError
from ragkit.harness.roles import Leniency, Limits, Panel, Persona, load_panel

PANEL = """
[revision]
max_revisions = 3
max_repairs = 1

[limits]
review_tokens = 512

[[persona]]
id = "producer"
kind = "producer"
model = "big"
instructions = "translate from {source_language}"

[[persona]]
id = "accuracy"
kind = "reviewer"
model = "small"
instructions = "check accuracy"

[[persona]]
id = "compliance"
kind = "reviewer"
model = "small"
from_rules = true
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "personas.toml"
    path.write_text(text, encoding="utf-8")
    return path


class TestLoad:
    def test_loads_a_panel(self, tmp_path: Path) -> None:
        panel = load_panel(_write(tmp_path, PANEL), {"source_language": "English"})
        assert panel.producer.id == "producer" and panel.producer.model == "big"
        assert "translate from English" in panel.producer.instructions
        assert [r.id for r in panel.reviewers] == ["accuracy", "compliance"]
        assert panel.reviewers[1].from_rules
        assert panel.max_revisions == 3 and panel.limits.review_tokens == 512
        assert panel.models_in_use() == {"big", "small"}

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="personas file not found"):
            load_panel(tmp_path / "absent.toml")

    def test_invalid_toml(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="invalid TOML"):
            load_panel(_write(tmp_path, "= broken"))

    def test_unknown_top_level_key(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="unknown key"):
            load_panel(_write(tmp_path, "bogus = 1\n"))

    def test_unfilled_placeholder_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="unknown placeholder"):
            load_panel(_write(tmp_path, PANEL))  # no substitutions provided

    def test_no_producer(self, tmp_path: Path) -> None:
        text = '[[persona]]\nid="r"\nkind="reviewer"\nmodel="m"\ninstructions="x"\n'
        with pytest.raises(ConfigError, match=r"exactly one persona.*producer"):
            load_panel(_write(tmp_path, text))

    def test_two_producers(self, tmp_path: Path) -> None:
        text = ('[[persona]]\nid="a"\nkind="producer"\nmodel="m"\ninstructions="x"\n'
                '[[persona]]\nid="b"\nkind="producer"\nmodel="m"\ninstructions="y"\n')
        with pytest.raises(ConfigError, match="found 2"):
            load_panel(_write(tmp_path, text))

    def test_no_reviewers(self, tmp_path: Path) -> None:
        text = '[[persona]]\nid="p"\nkind="producer"\nmodel="m"\ninstructions="x"\n'
        with pytest.raises(ConfigError, match="no reviewer personas"):
            load_panel(_write(tmp_path, text))

    def test_duplicate_id(self, tmp_path: Path) -> None:
        text = ('[[persona]]\nid="p"\nkind="producer"\nmodel="m"\ninstructions="x"\n'
                '[[persona]]\nid="p"\nkind="reviewer"\nmodel="m"\ninstructions="y"\n')
        with pytest.raises(ConfigError, match="duplicate persona id"):
            load_panel(_write(tmp_path, text))

    def test_missing_id(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="has no 'id'"):
            load_panel(_write(tmp_path, '[[persona]]\nkind="producer"\nmodel="m"\n'))

    def test_bad_kind(self, tmp_path: Path) -> None:
        text = '[[persona]]\nid="p"\nkind="boss"\nmodel="m"\ninstructions="x"\n'
        with pytest.raises(ConfigError, match="kind must be one of"):
            load_panel(_write(tmp_path, text))

    def test_missing_model(self, tmp_path: Path) -> None:
        text = '[[persona]]\nid="p"\nkind="producer"\ninstructions="x"\n'
        with pytest.raises(ConfigError, match="needs a 'model'"):
            load_panel(_write(tmp_path, text))

    def test_from_rules_on_producer(self, tmp_path: Path) -> None:
        text = ('[[persona]]\nid="p"\nkind="producer"\nmodel="m"\nfrom_rules=true\n'
                '[[persona]]\nid="r"\nkind="reviewer"\nmodel="m"\ninstructions="x"\n')
        with pytest.raises(ConfigError, match="applies only to a reviewer"):
            load_panel(_write(tmp_path, text))

    def test_from_rules_with_instructions_conflict(self, tmp_path: Path) -> None:
        text = ('[[persona]]\nid="p"\nkind="producer"\nmodel="m"\ninstructions="x"\n'
                '[[persona]]\nid="r"\nkind="reviewer"\nmodel="m"\nfrom_rules=true\n'
                'instructions="also this"\n')
        with pytest.raises(ConfigError, match="both from_rules and instructions"):
            load_panel(_write(tmp_path, text))

    def test_reviewer_without_instructions_or_from_rules(self, tmp_path: Path) -> None:
        text = ('[[persona]]\nid="p"\nkind="producer"\nmodel="m"\ninstructions="x"\n'
                '[[persona]]\nid="r"\nkind="reviewer"\nmodel="m"\n')
        with pytest.raises(ConfigError, match="act against nothing"):
            load_panel(_write(tmp_path, text))

    def test_per_reviewer_leniency_and_max_tokens(self, tmp_path: Path) -> None:
        text = ('[[persona]]\nid="p"\nkind="producer"\nmodel="m"\ninstructions="x"\n'
                '[[persona]]\nid="r"\nkind="reviewer"\nmodel="m"\ninstructions="y"\n'
                'max_tokens = 256\nleniency = {window = 5, max_bad = 1}\n')
        panel = load_panel(_write(tmp_path, text))
        assert panel.reviewers[0].max_tokens == 256
        assert panel.reviewers[0].leniency == Leniency(window=5, max_bad=1)


class TestBudgets:
    def test_produce_budget_is_bounded(self) -> None:
        limits = Limits(produce_tokens_per_source_char=8, produce_tokens_floor=100,
                        produce_tokens_ceiling=500)
        assert limits.produce_budget(1) == 100  # floor
        assert limits.produce_budget(1000) == 500  # ceiling
        assert limits.produce_budget(20) == 160  # scaled

    def test_floor_above_ceiling_is_refused(self) -> None:
        with pytest.raises(ValueError, match="exceeds"):
            Limits(produce_tokens_floor=1000, produce_tokens_ceiling=100)

    def test_review_budget_prefers_the_persona_ceiling(self) -> None:
        limits = Limits(review_tokens=1000)
        persona = Persona("r", "reviewer", "m", instructions="x", max_tokens=256)
        default = Persona("d", "reviewer", "m", instructions="x")
        assert limits.review_budget(persona) == 256
        assert limits.review_budget(default) == 1000


class TestLeniency:
    def test_surfaces_past_the_allowance(self) -> None:
        leniency = Leniency(window=10, max_bad=2)
        assert not leniency.surfaces(2)
        assert leniency.surfaces(3)

    def test_invalid_window(self) -> None:
        with pytest.raises(ValueError, match="window"):
            Leniency(window=0)


class TestPanelInvariants:
    def test_negative_budget_is_refused(self) -> None:
        with pytest.raises(ValueError, match="max_revisions"):
            Panel(producer=Persona("p", "producer", "m", instructions="x"),
                  reviewers=(Persona("r", "reviewer", "m", instructions="y"),), max_revisions=-1)

    def test_persona_bad_kind(self) -> None:
        with pytest.raises(ValueError, match="kind must be one of"):
            Persona("p", "boss", "m", instructions="x")
