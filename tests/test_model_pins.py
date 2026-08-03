"""The pinned-model contract between ``tools/fetch_models.sh`` and every recipe's ``models.toml``.

The setup story is "run ``tools/fetch_models.sh``, then ``tools/serve_models.sh``, then run the
recipe". That only holds if every chat ``model_id`` a recipe names is a GGUF the fetch script
actually downloads — the router exposes each file under its basename, so the two are one
contract split across two files, with nothing but convention keeping them aligned.

Convention was not enough: five of the six recipes once named a Qwen3.5 GGUF family that has
never existed, so the documented one-command setup could not complete for any of them, and no
test noticed because no test related the two files. These do.

Deliberately offline. Whether a repo *exists on Hugging Face* cannot be checked without the
network, and the suite is network-free by design (see ``pytest.ini``); what is checked here is
the half that is knowable locally — internal consistency — plus the script's behaviour when a
download fails, driven through stubbed download commands rather than a real one.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FETCH_SCRIPT = REPO_ROOT / "tools" / "fetch_models.sh"
RECIPES = REPO_ROOT / "recipes"

# `REPO[name]="org/repo";  FILE[name]="thing.gguf"` -- both halves share one declaration line, so
# this is not anchored to the line start.
_ENTRY = re.compile(r"""(REPO|FILE)\[(\w+)\]\s*=\s*"([^"]+)\"""")


def _pinned() -> dict[str, dict[str, str]]:
    """``{name: {"REPO": ..., "FILE": ...}}`` parsed out of the fetch script's declaration block."""
    entries: dict[str, dict[str, str]] = {}
    for kind, name, value in _ENTRY.findall(FETCH_SCRIPT.read_text(encoding="utf-8")):
        entries.setdefault(name, {})[kind] = value
    return entries


def _recipe_models() -> list[tuple[str, str, str]]:
    """``(recipe, logical_name, model_id)`` for every model in every recipe's models.toml."""
    found = []
    for path in sorted(RECIPES.glob("*/config/models.toml")):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        for name, body in data.get("model", {}).items():
            found.append((path.parent.parent.name, name, body))
    return [(recipe, name, body) for recipe, name, body in found]


class TestTheFetchScriptParses:
    def test_every_entry_declares_both_a_repo_and_a_file(self) -> None:
        pinned = _pinned()
        assert pinned, "no REPO[...]/FILE[...] entries found -- has the script's format changed?"
        for name, halves in pinned.items():
            assert set(halves) == {"REPO", "FILE"}, f"{name} declares only {sorted(halves)}"

    def test_every_pinned_file_is_a_gguf(self) -> None:
        for name, halves in _pinned().items():
            assert halves["FILE"].endswith(".gguf"), f"{name}: {halves['FILE']!r} is not a GGUF"

    def test_every_pinned_repo_is_org_slash_name(self) -> None:
        # A bare name would resolve to a canonical-model URL that is not what any entry means.
        for name, halves in _pinned().items():
            assert halves["REPO"].count("/") == 1, f"{name}: {halves['REPO']!r} is not 'org/repo'"


class TestRecipesNameFetchableModels:
    """The regression: a recipe's chat model_id must be a GGUF the fetch script downloads."""

    def test_every_chat_model_id_is_a_pinned_gguf(self) -> None:
        fetchable = {halves["FILE"].removesuffix(".gguf") for halves in _pinned().values()}
        for recipe, name, body in _recipe_models():
            # Embedding/rerank models are served through a *preset* (tools/embed_presets.ini),
            # which names the model under its own label rather than the GGUF basename.
            if body.get("kind", "chat") != "chat":
                continue
            model_id = body["model_id"]
            assert model_id in fetchable, (
                f"{recipe}/config/models.toml [model.{name}] names model_id {model_id!r}, which "
                f"tools/fetch_models.sh does not download. Fetchable: {sorted(fetchable)}")

    def test_presets_cover_the_non_chat_model_ids(self) -> None:
        presets = " ".join(p.read_text(encoding="utf-8")
                           for p in (REPO_ROOT / "tools").glob("*_presets.ini"))
        for recipe, name, body in _recipe_models():
            if body.get("kind", "chat") == "chat":
                continue
            assert body["model_id"] in presets, (
                f"{recipe} [model.{name}] uses preset name {body['model_id']!r}, which no "
                f"tools/*_presets.ini defines")


@pytest.mark.skipif(os.name != "posix", reason="the tool scripts are bash")
class TestFetchReportsEveryFailure:
    """A single unreachable entry must not hide the state of the rest.

    ``fetch()`` used to ``die`` on the first failure, so the entries after it in the loop were
    never attempted and the operator could not tell how much of the setup had actually worked --
    which is why the two dead repos read as one error rather than "five recipes cannot run".
    """

    def _run(self, tmp_path: Path) -> subprocess.CompletedProcess[str]:
        # Stub both download commands so nothing touches the network: whichever the script picks,
        # it fails, which is the condition under test.
        stub_dir = tmp_path / "bin"
        stub_dir.mkdir()
        for command in ("curl", "huggingface-cli"):
            stub = stub_dir / command
            stub.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
            stub.chmod(0o755)
        env = {**os.environ, "PATH": f"{stub_dir}:{os.environ['PATH']}"}
        return subprocess.run([str(FETCH_SCRIPT), "--dir", str(tmp_path / "models")],
                              capture_output=True, text=True, env=env, check=False)

    def test_it_attempts_every_model_and_lists_them_all(self, tmp_path: Path) -> None:
        result = self._run(tmp_path)
        assert result.returncode != 0, "a failed fetch must still exit non-zero"
        expected = {name for name in _pinned() if name != "med_author"}  # optional, skipped
        for name in expected:
            assert name in result.stderr, f"{name} was never attempted or never reported"
        assert f"{len(expected)} model(s) could not be downloaded" in result.stderr

    def test_a_failed_download_leaves_no_truncated_file(self, tmp_path: Path) -> None:
        # `curl -o` creates the destination before it knows the request will fail; leaving that
        # behind would make the next run report "already have" for a zero-byte model.
        self._run(tmp_path)
        leftovers = list((tmp_path / "models").glob("*"))
        assert leftovers == [], f"a failed fetch left {leftovers} behind"


class TestTheScriptRunsFromAnywhere:
    def test_unknown_model_name_is_refused(self, tmp_path: Path) -> None:
        result = subprocess.run([str(FETCH_SCRIPT), "--only", "nope", "--dir", str(tmp_path)],
                                capture_output=True, text=True, check=False, cwd=sys.path[0] or "/")
        assert result.returncode != 0 and "unknown model" in result.stderr
