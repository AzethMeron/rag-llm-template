"""Parsing and validating retrieval.toml into RetrievalSettings."""
from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.core.config import ConfigError
from ragkit.retrieve.tuning import RetrievalSettings, load_retrieval


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "retrieval.toml"
    path.write_text(text, encoding="utf-8")
    return path


class TestLoad:
    def test_defaults_to_lexical(self, tmp_path: Path) -> None:
        settings = load_retrieval(_write(tmp_path, "[retrieval]\nkind = \"lexical\"\n"))
        assert settings.kind == "lexical" and not settings.needs_embedding

    def test_full_hybrid(self, tmp_path: Path) -> None:
        text = ("[retrieval]\nkind = \"hybrid\"\ncandidate_pool = 25\nmmr_lambda = 0.5\n"
                "[retrieval.lexical]\nmin_score = 0.2\n"
                "[retrieval.dense]\nmodel = \"embedder\"\nmin_score = 0.6\n"
                "[retrieval.rerank]\nenabled = true\nmodel = \"reranker\"\n")
        s = load_retrieval(_write(tmp_path, text))
        assert s.kind == "hybrid" and s.candidate_pool == 25 and s.mmr_lambda == 0.5
        assert s.lexical_min_score == 0.2 and s.dense_min_score == 0.6
        assert s.embedding_model == "embedder" and s.needs_embedding
        assert s.rerank_enabled and s.rerank_model == "reranker"

    def test_unknown_section_key_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="unknown key"):
            load_retrieval(_write(tmp_path, "[retrieval]\nkind = \"lexical\"\nbogus = 1\n"))

    def test_unknown_subsection_key_refused(self, tmp_path: Path) -> None:
        text = "[retrieval]\nkind = \"lexical\"\n[retrieval.lexical]\nfloor = 0.3\n"
        with pytest.raises(ConfigError, match="unknown key"):
            load_retrieval(_write(tmp_path, text))

    def test_bad_kind_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="kind must be one of"):
            load_retrieval(_write(tmp_path, "[retrieval]\nkind = \"fuzzy\"\n"))

    def test_dense_needs_a_model(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match=r"needs \[retrieval\.dense\]\.model"):
            load_retrieval(_write(tmp_path, "[retrieval]\nkind = \"dense\"\n"))

    def test_rerank_enabled_needs_a_model(self, tmp_path: Path) -> None:
        text = ("[retrieval]\nkind = \"hybrid\"\n[retrieval.dense]\nmodel = \"e\"\n"
                "[retrieval.rerank]\nenabled = true\n")
        with pytest.raises(ConfigError, match="no model is set"):
            load_retrieval(_write(tmp_path, text))

    def test_rerank_under_a_single_arm_kind_is_refused(self, tmp_path: Path) -> None:
        # The bug this pins: kind="dense" + rerank enabled used to validate, assemble, and run
        # with no reranking and no warning.
        text = ("[retrieval]\nkind = \"dense\"\n[retrieval.dense]\nmodel = \"e\"\n"
                "[retrieval.rerank]\nenabled = true\nmodel = \"r\"\n")
        with pytest.raises(ConfigError, match="only the hybrid stack builds"):
            load_retrieval(_write(tmp_path, text))

    @pytest.mark.parametrize(("kind", "extra", "wanted"), [
        ("lexical", "candidate_pool = 25\n", r"\[retrieval\]\.candidate_pool"),
        ("lexical", "mmr_lambda = 0.5\n", r"\[retrieval\]\.mmr_lambda"),
        ("lexical", "[retrieval.lexical]\nmin_score = 0.2\n", r"\[retrieval\.lexical\]\.min_score"),
        ("lexical", "[retrieval.rerank]\nenabled = false\n", r"\[retrieval\.rerank\]\.enabled"),
        ("lexical", "[retrieval.rerank]\nmodel = \"r\"\n", r"\[retrieval\.rerank\]\.model"),
        ("dense", "[retrieval.lexical]\nmin_score = 0.2\n", r"\[retrieval\.lexical\]\.min_score"),
        ("dense", "candidate_pool = 25\n", r"\[retrieval\]\.candidate_pool"),
    ])
    def test_hybrid_only_keys_refused_for_single_arm_kinds(
            self, tmp_path: Path, kind: str, extra: str, wanted: str) -> None:
        dense_model = "[retrieval.dense]\nmodel = \"e\"\n" if kind == "dense" else ""
        text = f"[retrieval]\nkind = \"{kind}\"\n{extra}{dense_model}"
        with pytest.raises(ConfigError, match=wanted):
            load_retrieval(_write(tmp_path, text))

    def test_an_unknown_rerank_score_scale_is_refused(self, tmp_path: Path) -> None:
        text = ("[retrieval]\nkind = \"hybrid\"\n[retrieval.dense]\nmodel = \"e\"\n"
                "[retrieval.rerank]\nenabled = true\nmodel = \"r\"\nscore_scale = \"percent\"\n")
        with pytest.raises(ConfigError, match="score_scale must be one of"):
            load_retrieval(_write(tmp_path, text))

    def test_rerank_score_scale_round_trips(self, tmp_path: Path) -> None:
        text = ("[retrieval]\nkind = \"hybrid\"\n[retrieval.dense]\nmodel = \"e\"\n"
                "[retrieval.rerank]\nenabled = true\nmodel = \"r\"\nscore_scale = \"unit\"\n")
        assert load_retrieval(_write(tmp_path, text)).rerank_score_scale == "unit"

    def test_dense_min_score_refused_for_the_dense_kind(self, tmp_path: Path) -> None:
        # A fusion-input floor, not a stack floor: even the arm's *own* kind cannot use it.
        text = "[retrieval]\nkind = \"dense\"\n[retrieval.dense]\nmodel = \"e\"\nmin_score = 0.6\n"
        with pytest.raises(ConfigError, match=r"min_score in context\.toml"):
            load_retrieval(_write(tmp_path, text))

    def test_dense_model_refused_for_the_lexical_kind(self, tmp_path: Path) -> None:
        text = "[retrieval]\nkind = \"lexical\"\n[retrieval.dense]\nmodel = \"e\"\n"
        with pytest.raises(ConfigError, match="never embeds"):
            load_retrieval(_write(tmp_path, text))

    def test_mistyped_value_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="must be a number"):
            load_retrieval(_write(tmp_path, "[retrieval]\nmmr_lambda = \"hot\"\n"))

    def test_out_of_range_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="mmr_lambda must be in"):
            load_retrieval(_write(tmp_path, "[retrieval]\nmmr_lambda = 2.0\n"))


class TestSettings:
    def test_pool_floor(self) -> None:
        with pytest.raises(ValueError, match="candidate_pool must be >= 1"):
            RetrievalSettings(kind="lexical", candidate_pool=0)

    def test_min_score_range(self) -> None:
        with pytest.raises(ValueError, match=r"lexical\.min_score must be in"):
            RetrievalSettings(kind="lexical", lexical_min_score=1.5)

    def test_rerank_requires_the_hybrid_kind(self) -> None:
        # Guarded on the dataclass too, so a caller building settings in Python (no TOML, so no
        # key-presence check) cannot assemble a reranker that would never be called.
        with pytest.raises(ValueError, match="only the hybrid stack builds"):
            RetrievalSettings(kind="dense", embedding_model="e", rerank_enabled=True,
                              rerank_model="r")

    def test_embedding_model_requires_an_embedding_kind(self) -> None:
        with pytest.raises(ValueError, match="never embeds"):
            RetrievalSettings(kind="lexical", embedding_model="e")
