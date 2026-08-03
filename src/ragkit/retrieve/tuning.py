"""Config-driven retrieval tuning: ``retrieval.toml`` -> a validated :class:`RetrievalSettings`.

The retrieval stack (lexical + dense arms, RRF fusion, optional cross-encoder rerank, MMR) is
otherwise wired in Python; this makes it assemble from config so a recipe can pick the arms, the
per-arm score **floors**, the candidate pool, the MMR trade-off, and the rerank model without
editing code — the "everything tunable is TOML, including retrieval floors" convention applied to
the whole stack rather than only the block-level floor. Parsing lives here (pure, dependency-free,
testable in isolation); the actual wiring of endpoints and indexes is the CLI's, in ``cli/app.py``.

The final relevance floor for a run stays the retrieved *block*'s ``min_score`` (``context.toml``);
the per-arm ``lexical``/``dense`` floors here are the fusion-input floors that were previously
hardcoded constants.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragkit.core.config import (
    ConfigError,
    load_toml,
    read_bool,
    read_float,
    read_int,
    read_string,
    reject_unknown,
)

from .rerank import SCORE_SCALES

_KINDS = frozenset({"lexical", "dense", "hybrid"})

_HYBRID_ONLY = frozenset({"hybrid"})
_EMBEDDING_KINDS = frozenset({"dense", "hybrid"})

_FLOOR_NOTE = ("the per-arm floors gate what each arm contributes to fusion; a single-arm stack's "
               "floor is the retrieved block's min_score in context.toml")
_RERANK_NOTE = "reranking re-scores the fused candidate pool, which only the hybrid stack builds"

_APPLICABILITY: tuple[tuple[str, str, frozenset[str], str], ...] = (
    ("[retrieval]", "candidate_pool", _HYBRID_ONLY, "only fusion fetches a per-arm candidate pool"),
    ("[retrieval]", "mmr_lambda", _HYBRID_ONLY, "only the hybrid stack diversifies with MMR"),
    ("[retrieval.lexical]", "min_score", _HYBRID_ONLY, _FLOOR_NOTE),
    ("[retrieval.dense]", "min_score", _HYBRID_ONLY, _FLOOR_NOTE),
    ("[retrieval.dense]", "model", _EMBEDDING_KINDS, "only the dense and hybrid stacks embed"),
    ("[retrieval.rerank]", "enabled", _HYBRID_ONLY, _RERANK_NOTE),
    ("[retrieval.rerank]", "model", _HYBRID_ONLY, _RERANK_NOTE),
    ("[retrieval.rerank]", "score_scale", _HYBRID_ONLY, _RERANK_NOTE),
)
"""Which ``kind``s each tunable actually reaches. Only the hybrid stack fuses, so only it has a
candidate pool, per-arm floors, MMR, and a rerank stage; a key outside its kind's column would be
read, validated, and then dropped on the floor. Refusing it names the misconfiguration instead."""


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    """A fully-validated retrieval configuration. ``kind`` selects the stack; the dense/hybrid kinds
    name an embedding model, and rerank (when enabled) names a rerank model — both logical models
    from ``models.toml`` (of the matching ``kind``)."""

    kind: str = "lexical"
    candidate_pool: int = 40
    mmr_lambda: float = 0.7
    lexical_min_score: float = 0.30
    dense_min_score: float = 0.55
    embedding_model: str = ""
    rerank_enabled: bool = False
    rerank_model: str = ""
    rerank_score_scale: str = "logit"

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"retrieval.kind must be one of {sorted(_KINDS)}, got {self.kind!r}")
        if self.candidate_pool < 1:
            raise ValueError(f"candidate_pool must be >= 1, got {self.candidate_pool}")
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError(f"mmr_lambda must be in [0, 1], got {self.mmr_lambda}")
        for value, what in ((self.lexical_min_score, "lexical.min_score"),
                            (self.dense_min_score, "dense.min_score")):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{what} must be in [0, 1], got {value}")
        if self.needs_embedding and not self.embedding_model:
            raise ValueError(f"retrieval.kind={self.kind!r} needs [retrieval.dense].model (an "
                             f"embedding model from models.toml)")
        if self.embedding_model and not self.needs_embedding:
            raise ValueError(f"[retrieval.dense].model is set but retrieval.kind={self.kind!r} "
                             f"never embeds; use kind='dense' or 'hybrid'")
        if self.rerank_enabled and not self.rerank_model:
            raise ValueError("[retrieval.rerank].enabled is true but no model is set")
        if self.rerank_score_scale not in SCORE_SCALES:
            raise ValueError(f"[retrieval.rerank].score_scale must be one of "
                             f"{sorted(SCORE_SCALES)}, got {self.rerank_score_scale!r}")
        # Only the hybrid stack builds the fused pool a reranker re-scores, so an enabled reranker
        # under any other kind would be assembled, never called, and never missed -- exactly the
        # silent degradation this project forbids. Refuse it here rather than at query time.
        if self.rerank_enabled and self.kind != "hybrid":
            raise ValueError(f"[retrieval.rerank].enabled is true but retrieval.kind={self.kind!r};"
                             f" {_RERANK_NOTE}")

    @property
    def needs_embedding(self) -> bool:
        return self.kind in ("dense", "hybrid")


_DEFAULTS = RetrievalSettings()
"""One home for the retrieval defaults: the loader reads them from here rather than re-typing each
literal, so a field default and its loader default cannot drift (the Leniency loader pattern)."""


def _reject_inapplicable(kind: str, sections: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    """Refuse a key the chosen ``kind`` would ignore.

    ``reject_unknown`` catches a key the *file* has no meaning for; this catches one the *kind* has
    no meaning for. Both are misconfigurations the reader would otherwise never learn about: a
    ``candidate_pool`` under ``kind="lexical"`` parses, validates, and is then discarded. Presence
    is what is checked -- a default the loader supplies is not a claim the user made.
    """
    for label, key, kinds, why in _APPLICABILITY:
        if key in sections[label] and kind not in kinds:
            raise ConfigError(
                f"{label}.{key} applies only to retrieval.kind in {sorted(kinds)}, but kind is "
                f"{kind!r} -- {why}", path=path)


def load_retrieval(path: Path) -> RetrievalSettings:
    """Parse and validate ``retrieval.toml``. Unknown keys, keys the chosen ``kind`` would ignore,
    and mistyped values are refused, per the framework's config rules; range checks and the
    cross-key invariants live on :class:`RetrievalSettings`."""
    data = load_toml(path, what="retrieval file")
    reject_unknown(data, {"retrieval"}, label="the retrieval file", path=path)
    section = reject_unknown(data.get("retrieval", {}),
                             {"kind", "candidate_pool", "mmr_lambda", "lexical", "dense", "rerank"},
                             label="[retrieval]", path=path)
    lexical = reject_unknown(section.get("lexical", {}), {"min_score"},
                             label="[retrieval.lexical]", path=path)
    dense = reject_unknown(section.get("dense", {}), {"model", "min_score"},
                           label="[retrieval.dense]", path=path)
    rerank = reject_unknown(section.get("rerank", {}), {"enabled", "model", "score_scale"},
                            label="[retrieval.rerank]", path=path)
    try:
        settings = RetrievalSettings(
            kind=read_string(section, "kind", _DEFAULTS.kind, label="[retrieval]", path=path),
            candidate_pool=read_int(section, "candidate_pool", _DEFAULTS.candidate_pool,
                                    label="[retrieval]", path=path),
            mmr_lambda=read_float(section, "mmr_lambda", _DEFAULTS.mmr_lambda,
                                  label="[retrieval]", path=path),
            lexical_min_score=read_float(lexical, "min_score", _DEFAULTS.lexical_min_score,
                                         label="[retrieval.lexical]", path=path),
            dense_min_score=read_float(dense, "min_score", _DEFAULTS.dense_min_score,
                                       label="[retrieval.dense]", path=path),
            embedding_model=read_string(dense, "model", _DEFAULTS.embedding_model,
                                        label="[retrieval.dense]", path=path),
            rerank_enabled=read_bool(rerank, "enabled", _DEFAULTS.rerank_enabled,
                                     label="[retrieval.rerank]", path=path),
            rerank_model=read_string(rerank, "model", _DEFAULTS.rerank_model,
                                     label="[retrieval.rerank]", path=path),
            rerank_score_scale=read_string(rerank, "score_scale", _DEFAULTS.rerank_score_scale,
                                           label="[retrieval.rerank]", path=path))
    except ValueError as exc:
        raise ConfigError(f"[retrieval]: {exc}", path=path) from exc
    # After construction, so `kind` is already known to be one of the three -- an applicability
    # complaint about a kind that does not exist would bury the real error.
    _reject_inapplicable(settings.kind, {"[retrieval]": section, "[retrieval.lexical]": lexical,
                                         "[retrieval.dense]": dense, "[retrieval.rerank]": rerank},
                         path)
    return settings
