"""Rendering llama-server launch flags from models.toml — the serve-config side of 'settings that
are not per-request live in the serve file'."""
from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.llm.pool import EndpointSpec
from ragkit.llm.serveargs import (
    ServeArgsError,
    flags_for,
    host_port,
    main,
    render_flags,
)

_MODELS = """
[endpoint.local]
base_url = "http://0.0.0.0:9001/v1"
resident_max = 3
server_args = ["-ngl", "99", "--flash-attn", "on"]

[model.author]
endpoint = "local"
model_id = "qwen"
"""


def _write(tmp_path: Path, text: str = _MODELS) -> Path:
    path = tmp_path / "models.toml"
    path.write_text(text, encoding="utf-8")
    return path


class TestHostPort:
    def test_parses_host_and_port(self) -> None:
        assert host_port("http://127.0.0.1:8080/v1") == ("127.0.0.1", 8080)

    def test_missing_port_refused(self) -> None:
        with pytest.raises(ServeArgsError, match="no explicit port"):
            host_port("http://127.0.0.1/v1")

    def test_missing_host_refused(self) -> None:
        with pytest.raises(ServeArgsError, match="no host"):
            host_port(":8080")


class TestRenderFlags:
    def test_includes_bind_cap_and_server_args_verbatim(self) -> None:
        endpoint = EndpointSpec(name="local", base_url="http://0.0.0.0:9001/v1", resident_max=3,
                                server_args=("-ngl", "99"))
        flags = render_flags(endpoint, models_dir="/models")
        assert flags == ["--host", "0.0.0.0", "--port", "9001", "--models-dir", "/models",
                         "--models-max", "3", "--jinja", "-ngl", "99"]

    def test_no_server_args_is_just_the_router_flags(self) -> None:
        endpoint = EndpointSpec(name="local", base_url="http://127.0.0.1:8080/v1")
        assert "-ngl" not in render_flags(endpoint, models_dir="/m")

    def test_models_preset_pointing_at_a_missing_file_is_refused(self, tmp_path: Path) -> None:
        # A moved/deleted --models-preset file must fail fast here with a structured error naming
        # the flag and path, not surface only inside llama-server's own startup output.
        endpoint = EndpointSpec(name="embed", base_url="http://127.0.0.1:8081/v1",
                                server_args=("--models-preset", str(tmp_path / "nope.ini")))
        with pytest.raises(ServeArgsError, match="does not exist as a file"):
            render_flags(endpoint, models_dir="/m")

    def test_models_preset_pointing_at_a_real_file_is_accepted(self, tmp_path: Path) -> None:
        preset = tmp_path / "presets.ini"
        preset.write_text("[embed]\nmodel = x.gguf\n", encoding="utf-8")
        endpoint = EndpointSpec(name="embed", base_url="http://127.0.0.1:8081/v1",
                                server_args=("--models-preset", str(preset)))
        flags = render_flags(endpoint, models_dir="/m")
        assert flags[-2:] == ["--models-preset", str(preset)]

    def test_models_preset_with_no_value_is_refused(self) -> None:
        endpoint = EndpointSpec(name="embed", base_url="http://127.0.0.1:8081/v1",
                                server_args=("--models-preset",))
        with pytest.raises(ServeArgsError, match="has no value following it"):
            render_flags(endpoint, models_dir="/m")

    def test_an_unrelated_flag_is_never_checked_as_a_file(self) -> None:
        # -ngl's value ("99") is not a path and must not be validated as one.
        endpoint = EndpointSpec(name="local", base_url="http://127.0.0.1:8080/v1",
                                server_args=("-ngl", "99"))
        assert render_flags(endpoint, models_dir="/m")[-2:] == ["-ngl", "99"]


class TestFlagsFor:
    def test_reads_the_named_endpoint_from_config(self, tmp_path: Path) -> None:
        flags = flags_for(_write(tmp_path), "local", models_dir="/models")
        assert flags[:4] == ["--host", "0.0.0.0", "--port", "9001"]
        assert flags[-4:] == ["-ngl", "99", "--flash-attn", "on"]

    def test_unknown_endpoint_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ServeArgsError, match="unknown endpoint 'remote'"):
            flags_for(_write(tmp_path), "remote", models_dir="/models")


class TestMain:
    def test_prints_flags_one_per_line(self, tmp_path: Path,
                                       capsys: pytest.CaptureFixture) -> None:
        code = main(["--config", str(_write(tmp_path)), "--endpoint", "local",
                     "--models-dir", "/models"])
        out = capsys.readouterr().out.splitlines()
        assert code == 0 and out[0] == "--host" and "99" in out

    def test_error_goes_to_stderr_with_exit_1(self, tmp_path: Path,
                                              capsys: pytest.CaptureFixture) -> None:
        code = main(["--config", str(_write(tmp_path)), "--endpoint", "nope",
                     "--models-dir", "/m"])
        assert code == 1 and "error:" in capsys.readouterr().err
