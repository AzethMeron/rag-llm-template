"""Text normalisation before indexing: the cheap, deterministic cleanup that upstream extraction
quality depends on far more than the choice of ANN index.

Unicode is folded to NFC (so the same text spelled two ways matches, and display width measures
the same), whitespace is collapsed, and — optionally — words hyphenated across a line break (a PDF
artefact) are rejoined. Kept small and side-effect-free.
"""
from __future__ import annotations

import re
from typing import Any, cast
import unicodedata

_WHITESPACE = re.compile(r"\s+")
_DEHYPHENATE = re.compile(r"(\w)-\n(\w)")


def normalise(text: str, *, form: str = "NFC", collapse_whitespace: bool = True,
              dehyphenate: bool = False) -> str:
    """Normalise ``text``. ``form`` is a Unicode normalisation form (NFC/NFKC/NFD/NFKD);
    ``collapse_whitespace`` folds every run of whitespace to one space and strips the ends;
    ``dehyphenate`` rejoins a word split ``foo-\\nbar`` -> ``foobar`` (a line-break artefact).

    ``collapse_whitespace`` destroys paragraph structure, so run this **after** chunking if the
    chunker is structure-aware — see the ordering note in :mod:`ragkit.ingest`."""
    if form not in ("NFC", "NFKC", "NFD", "NFKD"):
        raise ValueError(f"unknown Unicode normalisation form {form!r}")
    if dehyphenate:
        text = _DEHYPHENATE.sub(r"\1\2", text)
    text = unicodedata.normalize(cast("Any", form), text)
    if collapse_whitespace:
        text = _WHITESPACE.sub(" ", text).strip()
    return text
