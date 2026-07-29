"""The model pool: config loading, the thrash guard (count and VRAM), per-model client dedup,
and RAII."""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from ragkit.core.config import ConfigError
from ragkit.llm import EndpointSpec, ModelPool, ModelPoolError, ModelSpec, load_models

from .conftest import always, chat_reply


def _factory(_base_url: str, _timeout: float) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(always(chat_reply('{"a": "b"}'))))


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


GOOD = """
[endpoint.local]
provider = "llamacpp-router"
base_url = "http://127.0.0.1:8080/v1"
resident_max = 2
parallel = 2

[model.producer]
endpoint = "local"
model_id = "qwen3-2b"
backend = "auto"
context_window = 8192

[model.reviewer]
endpoint = "local"
model_id = "qwen3-2b"

[model.embed]
endpoint = "local"
model_id = "bge-m3"
kind = "embedding"
"""


class TestLoad:
    def test_loads_endpoints_and_models(self, tmp_path: Path) -> None:
        pool = load_models(_write(tmp_path / "m.toml", GOOD), client_factory=_factory)
        assert pool.model_spec("producer").model_id == "qwen3-2b"
        assert pool.models_of_kind("embedding") == {"embed": pool.model_spec("embed")}

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="models file not found"):
            load_models(tmp_path / "absent.toml")

    def test_no_endpoints(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match=r"no \[endpoint"):
            load_models(_write(tmp_path / "m.toml", '[model.x]\nendpoint="a"\nmodel_id="m"\n'))

    def test_no_models(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match=r"no \[model"):
            load_models(_write(tmp_path / "m.toml", '[endpoint.local]\n'))

    def test_model_names_an_undefined_endpoint(self, tmp_path: Path) -> None:
        text = '[endpoint.local]\n[model.x]\nendpoint = "elsewhere"\nmodel_id = "m"\n'
        with pytest.raises(ConfigError, match=r"not.*defined"):
            load_models(_write(tmp_path / "m.toml", text))

    def test_model_missing_model_id(self, tmp_path: Path) -> None:
        text = '[endpoint.local]\n[model.x]\nendpoint = "local"\n'
        with pytest.raises(ConfigError, match="non-empty model_id"):
            load_models(_write(tmp_path / "m.toml", text))

    def test_model_missing_endpoint(self, tmp_path: Path) -> None:
        text = '[endpoint.local]\n[model.x]\nmodel_id = "m"\n'
        with pytest.raises(ConfigError, match="needs an endpoint"):
            load_models(_write(tmp_path / "m.toml", text))

    def test_unknown_endpoint_key(self, tmp_path: Path) -> None:
        text = '[endpoint.local]\nbogus = 1\n[model.x]\nendpoint="local"\nmodel_id="m"\n'
        with pytest.raises(ConfigError, match="unknown key"):
            load_models(_write(tmp_path / "m.toml", text))

    def test_unknown_top_level_key(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="unknown key"):
            load_models(_write(tmp_path / "m.toml", "bogus = 1\n"))

    def test_bad_provider(self, tmp_path: Path) -> None:
        text = ('[endpoint.local]\nprovider = "vllm"\n'
                '[model.x]\nendpoint="local"\nmodel_id="m"\n')
        with pytest.raises(ConfigError, match="unknown provider"):
            load_models(_write(tmp_path / "m.toml", text))

    def test_bad_kind(self, tmp_path: Path) -> None:
        text = ('[endpoint.local]\n[model.x]\nendpoint="local"\nmodel_id="m"\nkind="vision"\n')
        with pytest.raises(ConfigError, match="unknown kind"):
            load_models(_write(tmp_path / "m.toml", text))

    def test_bad_resident_max(self, tmp_path: Path) -> None:
        text = ('[endpoint.local]\nresident_max = 0\n'
                '[model.x]\nendpoint="local"\nmodel_id="m"\n')
        with pytest.raises(ConfigError, match="resident_max must be >= 1"):
            load_models(_write(tmp_path / "m.toml", text))


class TestThrashGuard:
    def _pool(self) -> ModelPool:
        endpoints = {"local": EndpointSpec("local", resident_max=2)}
        models = {
            "a": ModelSpec("a", "local", "model-a"),
            "b": ModelSpec("b", "local", "model-b"),
            "c": ModelSpec("c", "local", "model-c"),
            "dup": ModelSpec("dup", "local", "model-a"),  # same served id as 'a'
        }
        return ModelPool(endpoints, models, client_factory=_factory)

    def test_within_capacity_passes(self) -> None:
        self._pool().check_capacity(["a", "b"])

    def test_duplicate_served_ids_count_once(self) -> None:
        # 'a' and 'dup' are the same served model, so together they are one resident model.
        self._pool().check_capacity(["a", "dup", "b"])

    def test_too_many_distinct_models_is_refused_at_load_time(self) -> None:
        with pytest.raises(ModelPoolError, match="would hold 3 distinct models"):
            self._pool().check_capacity(["a", "b", "c"])

    def test_unknown_active_model_is_refused(self) -> None:
        with pytest.raises(ModelPoolError, match="unknown model 'ghost'"):
            self._pool().check_capacity(["ghost"])

    def test_vram_budget_refuses_an_over_budget_set_when_all_declare_size(self) -> None:
        endpoints = {"local": EndpointSpec("local", resident_max=4, vram_budget_mb=4000)}
        models = {"a": ModelSpec("a", "local", "ma", approx_vram_mb=3000),
                  "b": ModelSpec("b", "local", "mb", approx_vram_mb=3000)}
        pool = ModelPool(endpoints, models, client_factory=_factory)
        with pytest.raises(ModelPoolError, match="will not fit together"):
            pool.check_capacity(["a", "b"])

    def test_vram_check_is_skipped_when_a_size_is_undeclared(self) -> None:
        # Only the count check runs; the byte check must not fire on incomplete data.
        endpoints = {"local": EndpointSpec("local", resident_max=4, vram_budget_mb=1)}
        models = {"a": ModelSpec("a", "local", "ma", approx_vram_mb=3000),
                  "b": ModelSpec("b", "local", "mb")}  # b undeclared
        pool = ModelPool(endpoints, models, client_factory=_factory)
        pool.check_capacity(["a", "b"])  # no error despite the tiny budget

    def test_vram_within_budget_passes(self) -> None:
        endpoints = {"local": EndpointSpec("local", vram_budget_mb=8000)}
        models = {"a": ModelSpec("a", "local", "ma", approx_vram_mb=3000)}
        ModelPool(endpoints, models, client_factory=_factory).check_capacity(["a"])


class TestClientRouting:
    def _pool(self, tmp_path: Path) -> ModelPool:
        return load_models(_write(tmp_path / "m.toml", GOOD), client_factory=_factory)

    def test_personas_sharing_a_model_share_one_client(self, tmp_path: Path) -> None:
        pool = self._pool(tmp_path)
        client_a, model_a = pool.client_for("producer")
        client_b, _ = pool.client_for("producer")
        assert client_a is client_b and model_a == "qwen3-2b"

    def test_distinct_models_get_distinct_clients(self, tmp_path: Path) -> None:
        pool = self._pool(tmp_path)
        assert pool.client_for("producer")[0] is not pool.client_for("reviewer")[0]

    def test_requesting_a_non_chat_model_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ModelPoolError, match="not a chat model"):
            self._pool(tmp_path).client_for("embed")

    def test_unknown_model_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ModelPoolError, match="unknown model 'ghost'"):
            self._pool(tmp_path).client_for("ghost")

    def test_models_of_kind_rejects_an_unknown_kind(self, tmp_path: Path) -> None:
        with pytest.raises(ModelPoolError, match="unknown kind"):
            self._pool(tmp_path).models_of_kind("vision")

    def test_close_is_idempotent_and_usable_as_context_manager(self, tmp_path: Path) -> None:
        with self._pool(tmp_path) as pool:
            pool.client_for("producer")
        pool.close()  # second close is harmless

    def test_model_spec_unknown_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ModelPoolError, match="unknown model 'ghost'"):
            self._pool(tmp_path).model_spec("ghost")

    def test_without_a_factory_a_real_client_is_built(self) -> None:
        # No client_factory: the pool opens a real httpx.Client (no network until a request is
        # made). Building and closing it exercises the default transport path.
        endpoints = {"local": EndpointSpec("local")}
        models = {"p": ModelSpec("p", "local", "qwen3")}
        with ModelPool(endpoints, models) as pool:
            client, model_id = pool.client_for("p")
            assert model_id == "qwen3" and client is not None


class TestSpecInvariants:
    def test_negative_vram_budget_is_refused(self) -> None:
        with pytest.raises(ModelPoolError, match="vram_budget_mb must be >= 0"):
            EndpointSpec("local", vram_budget_mb=-1)

    def test_negative_context_window_is_refused(self) -> None:
        with pytest.raises(ModelPoolError, match="context_window must be >= 0"):
            ModelSpec("m", "local", "id", context_window=-1)

    def test_negative_approx_vram_is_refused(self) -> None:
        with pytest.raises(ModelPoolError, match="approx_vram_mb must be >= 0"):
            ModelSpec("m", "local", "id", approx_vram_mb=-1)
