"""The command-line entry point: assemble a run from a config directory and execute it. See
:mod:`ragkit.cli.app` for the config-driven assembly and :mod:`ragkit.cli.main` for the CLI
(run it with ``python -m ragkit.cli``)."""
from __future__ import annotations

from .app import Assembled, CliError, assemble

__all__ = ["assemble", "Assembled", "CliError"]
