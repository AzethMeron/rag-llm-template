"""The model-layer error taxonomy, typed by **blast radius** — the single most important
distinction in the engine, because it decides whether one bad request stops a whole run.

* :class:`LlmError` — the transport or server is broken, so every remaining request would fail
  the same way. It propagates; the run stops with its journal intact and a resume loses
  nothing.
* :class:`LlmContentError` — *this* request's output is unusable (unparseable JSON, a
  repetition loop, a refusal). It says nothing about the next request, so the caller records the
  failure and continues.

Turning the second into the first once burned every remaining record on a run that still exited
0, which is why the hierarchy is load-bearing rather than cosmetic. All of them subclass
:class:`~ragkit.core.errors.RagkitError`, so a caller can also catch "any framework error".
"""
from __future__ import annotations

from typing import Any

from ragkit.core.errors import RagkitError


class LlmError(RagkitError):
    """A request to the inference server failed or returned unusable output.

    The base of the taxonomy and the *broad* blast radius: absent a more specific subclass,
    this means the server or transport is broken and the run should stop.
    """

    def __init__(self, reason: str, *, role: str = "<none>", status: int | None = None,
                 body: str = "") -> None:
        super().__init__(reason, role=role, status=status,
                         body=(body[:400] if body else None))
        self.role = role
        self.status = status
        self.body = body


class LlmContentError(LlmError):
    """The server answered, but *this* request's output is unusable.

    Distinguished from its base by blast radius: a bare :class:`LlmError` means every remaining
    request would fail alike (stop the run); this means the model produced garbage for one
    particular input (record it and carry on).
    """


class LlmRefusalError(LlmContentError):
    """The model declined to produce output. Counted separately from other malformed output so a
    caller can distinguish a refusal from a repetition loop."""


class LlmTruncationError(LlmContentError):
    """Generation hit the token ceiling before the model finished. Its own type because the fix
    differs: the text is not wrong, it is unfinished — usually a repetition loop on a hard
    input. Not retried, since the ceiling is a property of this prompt."""


class LlmIncompleteJsonError(LlmContentError):
    """The reply's JSON envelope was cut short, but its content parses once closed.

    A narrow, verified failure shape: the model emitted end-of-sequence inside or just after the
    output value, before the object's closing brace. The envelope is repaired and carried in
    ``.recovered``, but **the content may still be cut off mid-clause**, and nothing at this
    layer can tell the two apart. So the caller must treat ``.recovered`` as an unverified
    *candidate* — re-check and re-review it, never certify it — which is exactly why this is a
    distinct type. Because it is still a :class:`LlmContentError`, a caller that does nothing
    special still gets the safe behaviour (record and continue).
    """

    def __init__(self, reason: str, *, role: str = "<none>", body: str = "",
                 recovered: dict[str, Any]) -> None:
        super().__init__(reason, role=role, body=body)
        self.recovered = recovered
