"""BM25 scoring and FTS5 query-escaping helpers, shared by every lexical pairing-store driver.

Extracted so the score convention and query-escaping live in exactly one place (CLAUDE.md: single
source of truth). Both shipped drivers — SQLite/FTS5 (:mod:`ragkit.store.pairings.sqlite`) and
DuckDB/fts (:mod:`ragkit.store.pairings.duckdb`) — map their engine's raw BM25 through
:func:`bm25_to_relevance`, so a ranking bug can only exist once and a ``min_score`` keeps its
meaning across a ``[pairings].driver`` swap. The FTS5 escaping is SQLite-only (DuckDB's
``match_bm25`` takes the query as a plain string, with no query language to escape into).
"""
from __future__ import annotations


def bm25_to_relevance(raw: float) -> float:
    """Map a raw, higher-is-better BM25 magnitude to a relevance in ``[0, 1)``.

    ``raw`` is ``>= 0``, larger meaning a better match — the convention the callers normalise
    *to*: DuckDB's ``match_bm25`` is already in it, while FTS5's ``bm25()`` is negated
    (``<= 0``, more negative is better) and is passed here as ``-bm25()``.

    ``raw / (1 + raw)`` is monotone, so ranking and RRF are unaffected by the choice; what the
    choice decides is what a *threshold* means. This is the one home for it because the two
    drivers used to disagree — SQLite applied ``1 - 2**bm25`` and DuckDB ``s / (1 + s)``, both
    monotone into ``[0, 1)`` but with different absolute values, so the same ``min_score`` admitted
    different hits after a driver swap and the conformance suite (which only asserted
    ``score >= 0``) could not see it. The rational form is kept over the exponential one because
    it saturates far more slowly: ``1 - 2**-raw`` is 0.999 by ``raw = 10``, collapsing every strong
    match onto the same score, whereas this still separates them.

    **What this does and does not make portable.** One shared transform means a ``min_score``
    keeps the same *shape* of meaning on either driver, and ranking is identical. It does **not**
    make the numbers equal, because the two engines do not compute the same raw BM25: FTS5 and
    DuckDB's ``fts`` differ in IDF handling in particular, so a term present in every document
    scores ``0.0`` on SQLite and ``~0.19`` on DuckDB for the same corpus. Converging further would
    mean reimplementing BM25 instead of using each engine's native index. So a floor tuned on one
    driver is a good starting point on the other, not an exact transfer — re-check it after a
    swap. The conformance suite pins what genuinely holds: identical ranking, and ``[0, 1)``.
    """
    return raw / (1.0 + raw)


def as_match(query: str) -> str:
    """Turn a free-text query into an FTS5 MATCH expression that treats every word as a term,
    OR-combined, quoting each so punctuation cannot be read as FTS5 query syntax (which would raise
    on an ordinary user query containing a quote, a colon, or a bare ``AND``)."""
    words = [w.replace('"', '""') for w in query.split() if w]
    return " OR ".join(f'"{w}"' for w in words) if words else '""'
