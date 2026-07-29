"""Drive the harness with no server: a model pool whose transport returns scripted replies,
routed to the produce or review queue by the schema the request carries."""
from __future__ import annotations

import json
from collections.abc import Sequence

import httpx

from ragkit.harness import (
    Harness,
    JsonFieldSchema,
    Panel,
    Persona,
    RuleSet,
    ValidatorPipeline,
)
from ragkit.harness.context import ContextAssembler, LiteralBlock
from ragkit.harness.context.assembler import _PlacedBlock
from ragkit.llm import EndpointSpec, ModelPool, ModelSpec

# A scripted reply is (content, finish_reason); content is None for a refusal.
Reply = tuple[str | None, str]


def ok(obj: dict) -> Reply:
    return (json.dumps(obj), "stop")


def raw(content: str) -> Reply:
    return (content, "stop")


def truncated(content: str) -> Reply:
    return (content, "length")


def refuse() -> Reply:
    return (None, "stop")


ACCEPT = ok({"acceptable": True, "issues": []})


class _Script:
    """Serves queued replies, choosing the produce or review queue by the request's schema."""

    def __init__(self, produce: Sequence[Reply], review: Sequence[Reply]) -> None:
        self._produce = list(produce)
        self._review = list(review)

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        schema = body.get("response_format", {}).get("json_schema", {}).get("schema", {})
        props = schema.get("properties", {})
        queue = self._review if "acceptable" in props else self._produce
        if not queue:
            raise AssertionError("scripted queue exhausted; the harness made an unexpected call")
        content, finish = queue.pop(0)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}, "finish_reason": finish}], "usage": {}})


def build_pool(produce: Sequence[Reply], review: Sequence[Reply]) -> ModelPool:
    script = _Script(produce, review)

    def factory(_base_url: str, _timeout: float) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(script.handler))

    endpoints = {"e": EndpointSpec("e")}
    models = {"prod": ModelSpec("prod", "e", "prod-model"),
              "rev": ModelSpec("rev", "e", "rev-model")}
    return ModelPool(endpoints, models, client_factory=factory)


def build_harness(produce: Sequence[Reply], review: Sequence[Reply], *,
                  ruleset: RuleSet | None = None, reviewers: int = 1,
                  max_revisions: int = 2, max_repairs: int = 2,
                  repair_truncated_json: bool = True, field: str = "output",
                  **harness_kwargs: object) -> Harness:
    """A ready harness with one producer and ``reviewers`` reviewers, all over the scripted pool.
    ``produce``/``review`` are the reply queues consumed in call order."""
    pool = build_pool(produce, review)
    ruleset = ruleset or RuleSet()
    panel = Panel(
        producer=Persona("p", "producer", "prod", instructions="produce the output"),
        reviewers=tuple(Persona(f"r{i}", "reviewer", "rev", instructions="review it")
                        for i in range(reviewers)),
        max_revisions=max_revisions, max_repairs=max_repairs,
        repair_truncated_json=repair_truncated_json)
    context = ContextAssembler([_PlacedBlock("literal", LiteralBlock("Do the task."))])
    return Harness(pool, panel, ruleset, JsonFieldSchema(field),
                   ValidatorPipeline(ruleset), context, **harness_kwargs)  # type: ignore[arg-type]
