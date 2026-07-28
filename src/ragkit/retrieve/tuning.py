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

from dataclasses import dataclass
from pathlib import Path

from ragkit.core.config import (
    ConfigError,
    load_toml,
    read_bool,
    read_float,
    read_int,
    read_string,
    reject_unknown,
)

_KINDS = frozenset({"lexical", "dense", "hybrid"})


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
        if self.rerank_enabled and not self.rerank_model:
            raise ValueError("[retrieval.rerank].enabled is true but no model is set")

    @property
    def needs_embedding(self) -> bool:
        return self.kind in ("dense", "hybrid")


def load_retrieval(path: Path) -> RetrievalSettings:
    """Parse and validate ``retrieval.toml``. Unknown keys and mistyped values are refused, per the
    framework's config rules; range checks live on :class:`RetrievalSettings`."""
    data = load_toml(path, what="retrieval file")
    reject_unknown(data, {"retrieval"}, label="the retrieval file", path=path)
    section = reject_unknown(data.get("retrieval", {}),
                             {"kind", "candidate_pool", "mmr_lambda", "lexical", "dense", "rerank"},
                             label="[retrieval]", path=path)
    lexical = reject_unknown(section.get("lexical", {}), {"min_score"},
                             label="[retrieval.lexical]", path=path)
    dense = reject_unknown(section.get("dense", {}), {"model", "min_score"},
                           label="[retrieval.dense]", path=path)
    rerank = reject_unknown(section.get("rerank", {}), {"enabled", "model"},
                            label="[retrieval.rerank]", path=path)
    try:
        return RetrievalSettings(
            kind=read_string(section, "kind", "lexical", label="[retrieval]", path=path),
            candidate_pool=read_int(section, "candidate_pool", 40, label="[retrieval]", path=path),
            mmr_lambda=read_float(section, "mmr_lambda", 0.7, label="[retrieval]", path=path),
            lexical_min_score=read_float(lexical, "min_score", 0.30,
                                         label="[retrieval.lexical]", path=path),
            dense_min_score=read_float(dense, "min_score", 0.55,
                                       label="[retrieval.dense]", path=path),
            embedding_model=read_string(dense, "model", "", label="[retrieval.dense]", path=path),
            rerank_enabled=read_bool(rerank, "enabled", False,
                                     label="[retrieval.rerank]", path=path),
            rerank_model=read_string(rerank, "model", "", label="[retrieval.rerank]", path=path))
    except ValueError as exc:
        raise ConfigError(f"[retrieval]: {exc}", path=path) from exc
