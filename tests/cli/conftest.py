"""A config-directory builder and a scripted model pool for testing the CLI with no server."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx

MODELS = """
[endpoint.local]
resident_max = 2
[model.prod]
endpoint = "local"
model_id = "prod"
[model.rev]
endpoint = "local"
model_id = "rev"
"""

PERSONAS = """
[[persona]]
id = "p"
kind = "producer"
model = "prod"
instructions = "produce the output"
[[persona]]
id = "r"
kind = "reviewer"
model = "rev"
instructions = "review it"
"""

CONTEXT = """
[[context.block]]
kind = "literal"
text = "Do the task."
"""

RECIPE = """
[task]
output_schema = "json_field"
input_label = "Line:"
[task.output_schema_options]
field = "translation"
"""


def write_config(config_dir: Path, *, models: str = MODELS, personas: str = PERSONAS,
                 rules: str = "", context: str = CONTEXT, recipe: str = RECIPE,
                 storage: str | None = None, reference: list[dict] | None = None) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "models.toml").write_text(models, encoding="utf-8")
    (config_dir / "personas.toml").write_text(personas, encoding="utf-8")
    (config_dir / "rules.toml").write_text(rules, encoding="utf-8")
    (config_dir / "context.toml").write_text(context, encoding="utf-8")
    (config_dir / "recipe.toml").write_text(recipe, encoding="utf-8")
    if storage is not None:
        (config_dir / "storage.toml").write_text(storage, encoding="utf-8")
    if reference is not None:
        (config_dir / "ref.jsonl").write_text(
            "\n".join(json.dumps(r) for r in reference), encoding="utf-8")
    return config_dir


def scripted_factory(produce: str = '{"translation": "RESULT"}',
                     review: str = '{"acceptable": true, "issues": []}',
                     ) -> Callable[[str, float], httpx.Client]:
    """A client factory whose transport returns ``produce`` for a produce call and ``review`` for a
    review call, routed by the request schema."""
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        schema = body.get("response_format", {}).get("json_schema", {}).get("schema", {})
        content = review if "acceptable" in schema.get("properties", {}) else produce
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {}})

    def factory(_base_url: str, _timeout: float) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler))

    return factory
