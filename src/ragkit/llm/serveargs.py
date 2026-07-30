"""Render the ``llama-server`` launch command for an endpoint from ``models.toml``, so the serve
script and the model pool read one source of truth.

The pool consumes an endpoint's *routing* facts (base_url, resident_max); the serve script needs
the *launch* facts (host, port, models-max, and the endpoint's ``server_args``). Both come from the
same :class:`~ragkit.llm.pool.EndpointSpec`, so a change to ``models.toml`` moves both together
rather than drifting between a config file and a hand-written command line.

``python -m ragkit.llm.serveargs --config models.toml --endpoint local`` prints the flags, one
per line, for ``serve_models.sh`` to read into an array. Router mode launches with *no* model, so
only host/port/models-dir/models-max plus the passthrough flags are emitted; the model files
themselves live in ``--models-dir``.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

from ragkit.core.errors import RagkitError

from .pool import EndpointSpec, load_models


class ServeArgsError(RagkitError):
    """The requested endpoint cannot be turned into a launch command."""


# server_args flags known to take a file path as their next value. Checked before launch so a
# missing/moved file (e.g. tools/embed_presets.ini) fails fast with a structured error naming the
# flag and path, rather than surfacing only inside llama-server's own startup output.
_FILE_TAKING_FLAGS = frozenset({"--models-preset"})


def _validate_server_args_files(server_args: Sequence[str]) -> None:
    for i, arg in enumerate(server_args):
        if arg not in _FILE_TAKING_FLAGS:
            continue
        if i + 1 >= len(server_args):
            raise ServeArgsError(f"server_args' {arg} has no value following it (expected a file "
                                 f"path)")
        path = server_args[i + 1]
        if not Path(path).is_file():
            raise ServeArgsError(
                f"server_args' {arg} {path!r} does not exist as a file (checked relative to "
                f"{Path.cwd()}, the directory this is launched from)")


def host_port(base_url: str) -> tuple[str, int]:
    """The host and port a server must bind to answer ``base_url``. A URL without an explicit port
    is refused rather than guessed: serving on the wrong port silently is exactly the class of
    failure the framework rejects."""
    parsed = urlparse(base_url)
    if not parsed.hostname:
        raise ServeArgsError(f"base_url {base_url!r} has no host to bind")
    if parsed.port is None:
        raise ServeArgsError(
            f"base_url {base_url!r} has no explicit port; the serve script needs one to bind "
            f"(e.g. http://127.0.0.1:8080/v1)")
    return parsed.hostname, parsed.port


def render_flags(endpoint: EndpointSpec, *, models_dir: str) -> list[str]:
    """The ``llama-server`` router-mode flags for ``endpoint``: bind address, resident cap, models
    directory, then the endpoint's own launch ``server_args`` verbatim."""
    _validate_server_args_files(endpoint.server_args)
    host, port = host_port(endpoint.base_url)
    flags = ["--host", host, "--port", str(port), "--models-dir", models_dir,
             "--models-max", str(endpoint.resident_max), "--jinja"]
    flags.extend(endpoint.server_args)
    return flags


def flags_for(config: Path, endpoint_name: str, *, models_dir: str) -> list[str]:
    pool = load_models(config)
    try:
        endpoint = pool.endpoint(endpoint_name)
    except RagkitError as exc:
        raise ServeArgsError(str(exc)) from exc
    return render_flags(endpoint, models_dir=models_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit llama-server launch flags from models.toml.")
    parser.add_argument("--config", type=Path, required=True, help="path to models.toml")
    parser.add_argument("--endpoint", required=True, help="the [endpoint.<name>] to serve")
    parser.add_argument("--models-dir", required=True, help="directory of GGUF model files")
    args = parser.parse_args(argv)
    try:
        flags = flags_for(args.config, args.endpoint, models_dir=args.models_dir)
    except RagkitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("\n".join(flags))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via main() in tests
    raise SystemExit(main())
