"""Pure retrieval-fusion math: Reciprocal Rank Fusion and Maximal Marginal Relevance.

Stdlib-only, no I/O, no numpy — plain list/dict algorithms over whatever hashable identity the
caller uses for a candidate (a chunk id, an entry, anything ``==``/``hash``-stable). Kept separate
from the retrievers that use them so each is provable in isolation.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar

H = TypeVar("H")

RRF_K = 60
"""Cormack et al.'s rank-fusion constant, and the conventional default. Exposed because a caller
that wants to read a fused score on an absolute scale needs the same ``k`` the fusion used -- see
:func:`best_possible_rrf`."""


def _require_finite(value: float, *, what: str, where: str) -> float:
    """Reject NaN/±inf at a boundary. A non-finite number does not merely give a wrong value here,
    it gives a wrong *order* silently: every comparison against NaN is ``False``, so a NaN never
    wins a maximum and never loses one -- it is ranked wherever the loop happens to leave it."""
    if not math.isfinite(value):
        raise ValueError(f"{what} must be finite, got {value!r} ({where})")
    return value


def rrf(rankings: Sequence[Sequence[H]], *, k: int = RRF_K) -> dict[H, float]:
    """Reciprocal Rank Fusion over any number of best-first rankings.

    Each ranking contributes ``1 / (k + rank)`` (rank 1-based) to every candidate it contains,
    summed across rankings — so a candidate near the top of several rankings outscores one near the
    top of only one. Deterministic, and unlike weighted-score fusion needs no calibration between
    the rankings' otherwise incomparable scales. A NaN ``k`` (which slips past a plain ``k < 1``
    test) would make every fused score NaN and sort arbitrarily, so it is rejected."""
    _require_finite(k, what="k", where="rrf")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    fused: dict[H, float] = {}
    for ranking in rankings:
        for rank, candidate in enumerate(ranking, start=1):
            fused[candidate] = fused.get(candidate, 0.0) + 1.0 / (k + rank)
    return fused


def best_possible_rrf(ranking_count: int, *, k: int = RRF_K) -> float:
    """The largest score :func:`rrf` can assign: every ranking placing a candidate first. Dividing
    a fused score by this gives a **query-independent** ``[0, 1]`` scale on which a floor keeps a
    stable meaning — 1.0 is "every retriever's top hit", 0.5 (with two rankings) is "one
    retriever's top hit, the other never found it". Deliberately not min-max over the query's own
    candidates, which would hand the best of three terrible candidates a perfect 1.0."""
    _require_finite(ranking_count, what="ranking_count", where="best_possible_rrf")
    _require_finite(k, what="k", where="best_possible_rrf")
    if ranking_count < 1:
        raise ValueError(f"ranking_count must be >= 1, got {ranking_count}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return ranking_count / (k + 1)


def mmr_order(candidates: Sequence[H], relevance: Mapping[H, float],
              similarity: Callable[[H, H], float], *, lambda_: float, k: int) -> list[H]:
    """Greedy Maximal Marginal Relevance selection.

    Repeatedly picks the remaining candidate maximising
    ``lambda_ * relevance[c] - (1 - lambda_) * max(similarity(c, s) for s in selected)``, so once a
    candidate is picked a later one resembling it is penalised — the standard way to keep a top-k
    list from being dominated by near-duplicates. Ties break by input order (first-seen wins), so
    the result is deterministic for deterministic callables. A non-finite relevance or similarity
    raises rather than producing a silently wrong order."""
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError(f"lambda_ must be in [0, 1], got {lambda_}")  # NaN/±inf fail this too
    if k <= 0:
        return []
    remaining = list(candidates)
    for candidate in remaining:
        if candidate not in relevance:
            raise ValueError(f"no relevance for candidate {candidate!r} (mmr_order)")
        _require_finite(relevance[candidate], what="relevance",
                        where=f"mmr_order candidate {candidate!r}")
    selected: list[H] = []
    while remaining and len(selected) < k:
        scores = [_mmr_score(candidate, selected, relevance, similarity, lambda_)
                  for candidate in remaining]
        selected.append(remaining.pop(max(range(len(remaining)), key=scores.__getitem__)))
    return selected


def _mmr_score(candidate: H, selected: Sequence[H], relevance: Mapping[H, float],
               similarity: Callable[[H, H], float], lambda_: float) -> float:
    # default=0.0 (not a 0.0 seed in the max) so a wholly dissimilar set, whose similarities are all
    # negative, keeps its negative -- a genuine anti-redundancy bonus -- while an empty selection
    # contributes nothing.
    redundancy = max((_require_finite(similarity(candidate, other), what="similarity",
                                      where=f"mmr_order between {candidate!r} and {other!r}")
                      for other in selected), default=0.0)
    return lambda_ * relevance[candidate] - (1.0 - lambda_) * redundancy
