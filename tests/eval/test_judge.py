"""The blinded A/B judge: de-blinding correctness, aggregate rates, and the self-comparison guard.
Every model call is a scripted in-memory transport -- no server, no network."""
from __future__ import annotations

import json

import httpx
import pytest

from ragkit.eval import AbItem, CircularEvaluationError, JudgeError, evaluate_ab
from ragkit.eval.judge import AbSummary, AbVerdict, _deblind
from ragkit.llm import LlmClient, ServerConfig
from ragkit.llm.backends import resolve_backend


def _client(choice: str) -> LlmClient:
    """A judge client that always returns ``choice`` ("1"/"2"/"tie")."""
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [
            {"message": {"content": json.dumps({"choice": choice, "reason": "because"})},
             "finish_reason": "stop"}], "usage": {}})
    return LlmClient(ServerConfig(), backend=resolve_backend("auto", "m"),
                     client=httpx.Client(transport=httpx.MockTransport(handler)))


def _items(n: int = 1) -> list[AbItem]:
    return [AbItem(f"i{j}", f"input {j}", f"A-output {j}", f"B-output {j}") for j in range(n)]


class TestDeblind:
    def test_choice_maps_through_the_shown_order(self) -> None:
        # a shown first, judge picks output 1 -> A wins.
        assert _deblind("i", "1", "", a_first=True).winner == "a"
        # b shown first, judge picks output 1 -> B wins.
        assert _deblind("i", "1", "", a_first=False).winner == "b"
        assert _deblind("i", "2", "", a_first=True).winner == "b"
        assert _deblind("i", "2", "", a_first=False).winner == "a"
        assert _deblind("i", "tie", "", a_first=True).winner == "tie"

    def test_an_out_of_enum_choice_fails_loud(self) -> None:
        """This branch carried `pragma: no cover` claiming the schema's enum made it unreachable.
        It does not: the client's shape check verifies `required` and `type`, never `enum`, so a
        backend in json_object mode (shape described in the prompt, not grammar-constrained) can
        return any string here."""
        with pytest.raises(JudgeError, match="out-of-range choice 'maybe'"):
            _deblind("i7", "maybe", "", a_first=True)

    def test_an_out_of_enum_choice_reaches_deblind_through_a_real_reply(self) -> None:
        # End to end, through the client: nothing between the model and _deblind rejects it.
        with pytest.raises(JudgeError, match="out-of-range choice"):
            evaluate_ab(_client("both"), _items(1), system_a="x", system_b="y",
                        criterion="better")


class TestEvaluateAb:
    def test_winner_is_deblinded_regardless_of_shown_order(self) -> None:
        # The judge always says "output 1 wins"; with A forced first, A must win; forced second, B.
        a_first = evaluate_ab(_client("1"), _items(1), system_a="x", system_b="y",
                              criterion="better", order=lambda _i: True)
        b_first = evaluate_ab(_client("1"), _items(1), system_a="x", system_b="y",
                              criterion="better", order=lambda _i: False)
        assert a_first.verdicts[0].winner == "a" and a_first.a_wins == 1
        assert b_first.verdicts[0].winner == "b" and b_first.b_wins == 1

    def test_aggregate_rates_with_ties(self) -> None:
        summary = evaluate_ab(_client("tie"), _items(4), system_a="x", system_b="y",
                              criterion="better")
        assert summary.ties == 4 and summary.a_wins == 0
        assert summary.a_win_rate == 0.5  # all ties -> even split

    def test_a_win_rate_counts_wins_and_half_ties(self) -> None:
        summary = evaluate_ab(_client("1"), _items(3), system_a="x", system_b="y",
                              criterion="better", order=lambda _i: True)
        assert summary.a_wins == 3 and summary.a_win_rate == 1.0

    def test_self_comparison_is_refused(self) -> None:
        with pytest.raises(CircularEvaluationError, match="against itself"):
            evaluate_ab(_client("1"), _items(1), system_a="x", system_b="x", criterion="better")

    def test_empty_summary_rate_is_zero(self) -> None:
        assert AbSummary("x", "y", ()).a_win_rate == 0.0

    def test_the_judge_is_not_told_which_system_is_which(self) -> None:
        # The prompt must show neutral "Output 1/2" labels, never the system names.
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"choices": [
                {"message": {"content": json.dumps({"choice": "tie"})},
                 "finish_reason": "stop"}], "usage": {}})

        client = LlmClient(ServerConfig(), backend=resolve_backend("auto", "m"),
                           client=httpx.Client(transport=httpx.MockTransport(handler)))
        evaluate_ab(client, _items(1), system_a="alpha", system_b="beta", criterion="better")
        prompt = seen["body"]["messages"][-1]["content"]
        assert "Output 1" in prompt and "Output 2" in prompt
        assert "alpha" not in prompt and "beta" not in prompt


def test_verdict_dataclass_is_frozen() -> None:
    verdict = AbVerdict("i", "a", "r", a_shown_first=True)
    with pytest.raises(AttributeError):
        verdict.winner = "b"  # type: ignore[misc]
