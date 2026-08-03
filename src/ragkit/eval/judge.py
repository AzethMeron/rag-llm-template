"""Blinded A/B output evaluation: an impartial LLM judge compares two systems' outputs per item.

Two independence disciplines, both from the same rule that a metric must not be entangled with what
it ranks:

* **Blinding.** The judge never sees which system produced which output; the two are shown as
  "Output 1" / "Output 2" in an order decided by an injected function, so position bias is balanced
  deterministically (no hidden RNG) and the mapping back to A/B is recorded, not guessed.
* **Distinctness.** A/B-ing a system against itself is refused — the comparison would be between a
  system and its own copy, and any "win rate" would be an artefact.

The judge call goes through the ordinary :class:`~ragkit.llm.client.LlmClient` with a strict schema,
so a reply that is not one of the three allowed verdicts is a content error, not a silent miscount.
Every model call in the test suite is a scripted fake — no server, no network.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ragkit.core.errors import RagkitError
from ragkit.core.ports import Message, SamplingParams
from ragkit.llm.client import LlmClient

from .retrieval import CircularEvaluationError

_VERDICT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["choice"],
    "properties": {
        "choice": {"type": "string", "enum": ["1", "2", "tie"]},
        "reason": {"type": "string"},
    },
}


class JudgeError(RagkitError):
    """The judge returned a verdict outside the allowed set."""


@dataclass(frozen=True, slots=True)
class AbItem:
    """One comparison: the task input, and the two systems' outputs for it."""

    item_id: str
    task_input: str
    output_a: str
    output_b: str


@dataclass(frozen=True, slots=True)
class AbVerdict:
    """The de-blinded verdict for one item: which system won (``'a'``/``'b'``/``'tie'``)."""

    item_id: str
    winner: str
    reason: str
    a_shown_first: bool


@dataclass(frozen=True, slots=True)
class AbSummary:
    """Aggregate A/B outcome for a batch."""

    system_a: str
    system_b: str
    verdicts: tuple[AbVerdict, ...]

    @property
    def total(self) -> int:
        return len(self.verdicts)

    @property
    def a_wins(self) -> int:
        return sum(1 for v in self.verdicts if v.winner == "a")

    @property
    def b_wins(self) -> int:
        return sum(1 for v in self.verdicts if v.winner == "b")

    @property
    def ties(self) -> int:
        return sum(1 for v in self.verdicts if v.winner == "tie")

    @property
    def a_win_rate(self) -> float:
        # Ties count as half a win to each side, so the two rates sum to 1.
        return (self.a_wins + 0.5 * self.ties) / self.total if self.total else 0.0


def _alternate(index: int) -> bool:
    """Default ordering: show A first on even items, B first on odd — balancing position bias
    across the batch without a random source."""
    return index % 2 == 0


def evaluate_ab(client: LlmClient, items: Sequence[AbItem], *, system_a: str, system_b: str,
                criterion: str, model: str | None = None,
                order: Callable[[int], bool] = _alternate) -> AbSummary:
    """Judge every item's two outputs and return the aggregate. ``criterion`` is what "better"
    means for this task (e.g. "a more faithful, fluent translation"). ``order(index)`` decides
    whether system A is shown first for that item (injected for determinism)."""
    if system_a == system_b:
        raise CircularEvaluationError(
            f"cannot A/B a system against itself (both are {system_a!r}); the comparison would be "
            f"a system against its own copy and any win rate would be an artefact")

    verdicts = []
    for index, item in enumerate(items):
        a_first = order(index)
        first, second = ((item.output_a, item.output_b) if a_first
                         else (item.output_b, item.output_a))
        reply = client.complete_json(
            [Message("system", _system_prompt(criterion)),
             Message("user", _user_prompt(item.task_input, first, second))],
            _VERDICT_SCHEMA, role="judge", sampling=SamplingParams(temperature=0.0), model=model)
        verdicts.append(_deblind(item.item_id, str(reply["choice"]),
                                 str(reply.get("reason", "")), a_first=a_first))
    return AbSummary(system_a=system_a, system_b=system_b, verdicts=tuple(verdicts))


def _deblind(item_id: str, choice: str, reason: str, *, a_first: bool) -> AbVerdict:
    if choice == "tie":
        winner = "tie"
    elif choice in ("1", "2"):
        first_wins = choice == "1"
        winner = "a" if first_wins == a_first else "b"
    else:
        # Reachable, despite the schema declaring an enum: the client's shape check verifies
        # `required` and `type`, never `enum`, so a backend running in json_object mode (shape
        # described in the prompt rather than grammar-constrained) can return any string here.
        # This used to be marked `pragma: no cover` as unreachable, which both hid a real path
        # from the suite and would have made the guard look redundant to a later reader.
        raise JudgeError(f"judge returned an out-of-range choice {choice!r} for item {item_id!r}")
    return AbVerdict(item_id=item_id, winner=winner, reason=reason, a_shown_first=a_first)


def _system_prompt(criterion: str) -> str:
    return ("You are an impartial judge comparing two candidate outputs for the same task. Decide "
            f"which one is better by this criterion: {criterion}. Judge only the outputs shown; "
            "you are not told which system produced which. Answer with choice \"1\", \"2\", or "
            "\"tie\".")


def _user_prompt(task_input: str, first: str, second: str) -> str:
    return (f"Task input:\n{task_input}\n\nOutput 1:\n{first}\n\nOutput 2:\n{second}\n\n"
            "Which output is better? Reply with choice 1, 2, or tie, and a brief reason.")
