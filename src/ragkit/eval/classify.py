"""Scoring a recipe that predicts a **label** per record: a decision, a severity bucket, a class.

Four of the shipped recipes do exactly this and had four near-identical ``Outcome``/``Report``
pairs to prove it — differing only in the field name and which extra rates they reported. What is
genuinely shared is the shape (did it produce anything, what did it predict, what was the gold,
what fraction got it right); what is genuinely per-recipe is *which* rates matter. So the report
below supplies the shape and a generic :meth:`ClassificationReport.rate`, and a recipe adds its
own named properties over that rather than reimplementing the counting.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .gold import Pair, json_field


@dataclass(frozen=True, slots=True)
class Labelled:
    """One record's verdict: what the system predicted (``None`` when it produced nothing usable)
    against the gold label."""

    record_id: str
    produced: bool
    predicted: str | None
    gold: str

    @property
    def correct(self) -> bool:
        # A record that produced nothing has predicted=None, which equals no gold label, so a
        # miss can never score as correct.
        return self.predicted == self.gold


@dataclass(frozen=True, slots=True)
class ClassificationReport:
    """Counts and rates over a run's :class:`Labelled` outcomes.

    Subclass it to add a recipe's own metrics as properties over :meth:`rate` — an abstention
    rate, an "actionable" rate that forgives a within-group confusion — rather than recounting.
    """

    outcomes: tuple[Labelled, ...]

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def produced(self) -> int:
        """How many records the run produced any usable output for."""
        return sum(1 for outcome in self.outcomes if outcome.produced)

    @property
    def accuracy(self) -> float:
        """Correct over *every* evaluated record, so a record the run failed to produce counts
        against the score rather than vanishing from the denominator."""
        return self.rate(lambda outcome: outcome.correct)

    def rate(self, predicate: Callable[[Labelled], bool], *, over: int | None = None) -> float:
        """The fraction of outcomes satisfying ``predicate``, over ``total`` unless ``over`` says
        otherwise (a recipe scoring only the records it committed to). Zero when the denominator
        is zero, so an empty run reports 0.0 rather than dividing by it."""
        denominator = self.total if over is None else over
        if denominator <= 0:
            return 0.0
        return sum(1 for outcome in self.outcomes if predicate(outcome)) / denominator


def score_labels(pairs: Iterable[Pair], *, field: str) -> tuple[Labelled, ...]:
    """Turn ``(record_id, produced, gold)`` triples into :class:`Labelled` outcomes, reading the
    prediction from ``field`` of each produced JSON object. Gold is lower-cased and stripped to
    match, so casing in a gold file cannot silently cost accuracy."""
    return tuple(
        Labelled(record_id=record_id, produced=produced is not None,
                 predicted=json_field(produced, field), gold=str(gold).strip().lower())
        for record_id, produced, gold in pairs)
