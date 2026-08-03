"""The shared label-classification report the recipes build their own metrics on."""
from __future__ import annotations

import json

from ragkit.eval.classify import ClassificationReport, Labelled, score_labels


def _pairs(*triples: tuple[str, str | None, str]) -> list[tuple[str, str | None, object]]:
    return [(rid, produced, gold) for rid, produced, gold in triples]


def _decision(value: str) -> str:
    return json.dumps({"decision": value})


class TestScoreLabels:
    def test_reads_the_prediction_from_the_named_field(self) -> None:
        [outcome] = score_labels(_pairs(("r1", _decision("yes"), "yes")), field="decision")
        assert outcome == Labelled("r1", produced=True, predicted="yes", gold="yes")
        assert outcome.correct

    def test_gold_is_normalised_so_casing_cannot_cost_accuracy(self) -> None:
        [outcome] = score_labels(_pairs(("r1", _decision("yes"), "  YES ")), field="decision")
        assert outcome.gold == "yes" and outcome.correct

    def test_a_record_that_produced_nothing_is_a_miss(self) -> None:
        [outcome] = score_labels(_pairs(("r1", None, "yes")), field="decision")
        assert not outcome.produced and outcome.predicted is None and not outcome.correct

    def test_unusable_output_is_a_miss_but_still_counts_as_produced(self) -> None:
        # It produced *something*; it just was not usable. The two counts differ on purpose.
        [outcome] = score_labels(_pairs(("r1", "not json", "yes")), field="decision")
        assert outcome.produced and outcome.predicted is None and not outcome.correct


class TestClassificationReport:
    def _report(self) -> ClassificationReport:
        return ClassificationReport(score_labels(_pairs(
            ("r1", _decision("yes"), "yes"),      # correct
            ("r2", _decision("no"), "yes"),       # wrong
            ("r3", None, "yes"),                  # never produced
        ), field="decision"))

    def test_counts(self) -> None:
        report = self._report()
        assert report.total == 3 and report.produced == 2

    def test_accuracy_is_over_every_record_not_just_the_produced_ones(self) -> None:
        # A record the run failed to produce counts against the score rather than vanishing from
        # the denominator, which would flatter a run that crashed on its hardest inputs.
        assert self._report().accuracy == 1 / 3

    def test_rate_over_a_caller_supplied_denominator(self) -> None:
        report = self._report()
        assert report.rate(lambda outcome: outcome.correct, over=2) == 0.5

    def test_a_zero_denominator_is_zero_not_a_division_error(self) -> None:
        report = self._report()
        assert report.rate(lambda outcome: outcome.correct, over=0) == 0.0
        assert ClassificationReport(()).accuracy == 0.0
