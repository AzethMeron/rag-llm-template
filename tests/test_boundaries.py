"""The layer boundaries, enforced by walking every module's imports.

The framework is decoupled only as long as the dependency direction holds and the contract
stays stdlib-only. Both are easy to break invisibly with one convenient import, so they are
checked here rather than trusted:

* ``core`` imports nothing outside the standard library (and its own submodules), so the
  contract can never break because an optional dependency was upgraded or is absent.
* No layer imports a layer above it (``core`` at the bottom, ``cli`` at the top).
* No framework module names a concrete driver module -- every driver is reached through a
  registry, never imported by type, which is what makes a driver swap a config edit with no
  source change.

Plus a subprocess check that actually imports ``ragkit.core`` with the optional third-party
dependencies made unimportable -- the strongest form of "the contract needs none of them".
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
TOP = "ragkit"

# Bottom-to-top. A module in a layer may import its own layer and any layer below it, never
# above. Layers not yet built simply contribute no modules; the rule already holds for them.
LAYER_ORDER = ("core", "store", "llm", "ingest", "retrieve", "harness", "cli")
LAYER_RANK = {name: rank for rank, name in enumerate(LAYER_ORDER)}


def _third_party_and_layers(module: Path) -> tuple[set[str], set[str]]:
    """Return ``(third_party_top_packages, ragkit_layers)`` imported by ``module``.

    Relative imports (``from . import x``) are internal to a package and ignored. An absolute
    ``ragkit.<layer>....`` import contributes ``<layer>``; anything else outside the standard
    library is a third-party top-level package.
    """
    tree = ast.parse(module.read_text(encoding="utf-8"))
    stdlib = sys.stdlib_module_names
    third_party: set[str] = set()
    layers: set[str] = set()

    def classify(dotted: str) -> None:
        parts = dotted.split(".")
        if parts[0] == TOP:
            if len(parts) >= 2 and parts[1] in LAYER_RANK:
                layers.add(parts[1])
            return
        if parts[0] not in stdlib:
            third_party.add(parts[0])

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            classify(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                classify(alias.name)
    return third_party, layers


def _module_layer(relative: Path) -> str | None:
    """The layer a module belongs to, from its path under ``ragkit/``."""
    parts = relative.parts
    if len(parts) >= 2 and parts[0] == TOP and parts[1] in LAYER_RANK:
        return parts[1]
    return None


def _modules() -> list[tuple[Path, Path]]:
    return [(path, path.relative_to(SRC)) for path in sorted((SRC / TOP).rglob("*.py"))]


class TestTheContractIsStdlibOnly:
    def test_core_imports_nothing_outside_the_standard_library(self) -> None:
        for path, relative in _modules():
            if _module_layer(relative) != "core":
                continue
            third_party, _ = _third_party_and_layers(path)
            assert not third_party, f"{relative} imports third-party package(s) {third_party}"

    def test_core_imports_no_higher_layer(self) -> None:
        for path, relative in _modules():
            if _module_layer(relative) != "core":
                continue
            _, layers = _third_party_and_layers(path)
            assert layers <= {"core"}, f"{relative} imports non-core layer(s) {layers - {'core'}}"


class TestTheDependencyDirectionHolds:
    def test_no_module_imports_a_layer_above_it(self) -> None:
        for path, relative in _modules():
            layer = _module_layer(relative)
            if layer is None:
                continue
            _, layers = _third_party_and_layers(path)
            above = {name for name in layers if LAYER_RANK[name] > LAYER_RANK[layer]}
            assert not above, f"{relative} (layer {layer!r}) imports layer(s) above it: {above}"


# A driver dependency is a third-party package that a single driver module owns. Framework code
# must reach the driver through a registry, so its dependency may appear in exactly its own
# module (a leaf path substring) and nowhere else. That confinement is what keeps the core import
# path clean and lets a run without that feature omit the dependency entirely.
DRIVER_DEP_LOCATIONS = {
    "lancedb": ("store/vector/lancedb.py",),
    "pyarrow": ("store/vector/lancedb.py",),
    "numpy": ("store/vector/", "retrieve/", "ingest/"),  # dense-vector math
    "usearch": ("store/vector/",),
    "qdrant_client": ("store/vector/",),
    "sqlite_vec": ("store/vector/",),
    "psycopg": ("store/",),
    "chromadb": ("store/vector/",),
}


class TestDriverDependenciesAreConfinedToTheirModules:
    def test_a_driver_dependency_appears_only_in_its_own_module(self) -> None:
        for path, relative in _modules():
            posix = relative.as_posix()
            third_party, _ = _third_party_and_layers(path)
            for dep in third_party & DRIVER_DEP_LOCATIONS.keys():
                allowed = DRIVER_DEP_LOCATIONS[dep]
                assert any(location in posix for location in allowed), (
                    f"{relative} imports driver dependency {dep!r}, which is confined to "
                    f"{allowed}")


class TestTheContractRunsWithTheOptionalDepsUnimportable:
    def test_core_imports_with_httpx_numpy_lancedb_made_unimportable(self, tmp_path: Path) -> None:
        script = tmp_path / "check.py"
        script.write_text(
            "import sys\n"
            f"sys.path.insert(0, {str(SRC)!r})\n"
            # Prove the contract needs none of the optional third-party stack.
            "for name in ('httpx', 'numpy', 'lancedb', 'pyarrow'):\n"
            "    sys.modules[name] = None\n"
            "import ragkit.core as c\n"
            "r = c.Record(record_id='x', source='hi')\n"
            "assert c.display_columns('ab') == 2\n"
            "assert c.placeholder_indices('[[0]][[1]]') == [0, 1]\n"
            "print('ok')\n",
            encoding="utf-8")
        result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout
