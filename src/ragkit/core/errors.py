"""The one base every deliberate error in the framework derives from.

Errors here are *structured and diagnostic*, not bare strings (CLAUDE.md): each carries a
machine-readable ``reason`` and a ``context`` mapping — expected vs. actual values, the
offending input, the operation — so a failure can be caught, inspected, and debugged
without reproducing it. A caller that wants to handle "any error this framework raised on
purpose" catches :class:`RagkitError`; a caller that wants a specific failure catches the
subclass a subsystem defines beside its own code (``ConfigError`` in :mod:`ragkit.core.config`,
``RegistryError`` in :mod:`ragkit.core.registry`, ``CatalogError`` in
:mod:`ragkit.core.records`, and so on).

The subclasses live with their subsystems rather than here, so this module depends on
nothing and every layer can raise a framework error without importing half the framework.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class RagkitError(Exception):
    """Base for every error the framework raises deliberately.

    Never raised directly — a subclass names the subsystem, so a handler can be as broad or
    as narrow as it needs. ``reason`` is the stable, human-and-machine-readable statement of
    *what* went wrong; keyword ``context`` carries *where* and the diagnostic detail. A
    ``None`` value in ``context`` is dropped from the rendered message so an absent optional
    (a path that was in memory, say) does not print as ``path=None``.
    """

    def __init__(self, reason: str, **context: Any) -> None:
        self.reason = reason
        # Kept as a plain dict so a handler can read individual fields (e.g. ``exc.context
        # ["path"]``) rather than parse them back out of the message string.
        self.context: Mapping[str, Any] = {k: v for k, v in context.items() if v is not None}
        super().__init__(self._render())

    def _render(self) -> str:
        if not self.context:
            return self.reason
        detail = ", ".join(f"{key}={value!r}" for key, value in self.context.items())
        return f"{self.reason} ({detail})"
