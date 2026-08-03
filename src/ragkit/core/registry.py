"""The one extension mechanism: a typed component registry, one per port.

Everything swappable in the framework — a store driver, a retriever, a chunker, a model
backend, a context block, a validator — is a *component* resolved through a
:class:`Registry`. This is what lets a user replace most things without editing framework
code: a component is selected in config by name, by a dotted import path, or by a published
entry point, and constructed from its own validated options.

A registry is parameterised by the ``Protocol`` its components must satisfy, and it refuses,
at registration, anything that does not. Each component declares its own allowed option keys
(``CONFIG_KEYS``), so the same unknown-key strictness the built-in config loaders enforce
extends to third-party components too — a typo in *their* options is an error, not a silently
ignored default.

**Global-state discipline.** A module-level registry is global mutable state, which
``CLAUDE.md`` forbids in general; this is the deliberate, bounded exception (the codec-registry
pattern). It is earned, not inherited, by two rules this class enforces:

1. A registry holds only *component types and their option schemas* — never per-run state.
   The state a run mutates lives on the objects :meth:`create` constructs, owned by whoever
   constructs them.
2. Resolution is a pure function of ``(registry contents, spec, options)``. A run reads the
   registry; it does not write it. Registration is meant to happen once, at import, so the
   built-in set is fixed by the time any run resolves against it — and a test can build a
   throwaway :class:`Registry` rather than mutating a shared one.
"""
from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from .config import ConfigError, reject_unknown
from .errors import RagkitError

P = TypeVar("P")


class RegistryError(RagkitError):
    """A component could not be registered, found, or constructed."""


@runtime_checkable
class Component(Protocol):
    """The convention a registrable component follows.

    ``CONFIG_KEYS`` is the set of option keys the component accepts; the registry rejects any
    other key before construction. ``from_config`` builds an instance from its validated
    options — a component with no options may omit it, and the registry constructs it with no
    arguments instead. Neither member is part of the *port* Protocol a component implements
    (that describes what the component *does*); this describes only how the registry builds it.
    """

    CONFIG_KEYS: frozenset[str]

    @classmethod
    def from_config(cls, options: Mapping[str, Any]) -> Any: ...


@dataclass(frozen=True, slots=True)
class _Entry:
    """A registered component: the class itself, resolved once at registration."""

    component: type
    canonical_name: str


# The signature importlib.metadata.entry_points(group=...) satisfies, injected so a test can
# drive entry-point discovery without installing a package.
EntryPointLoader = Callable[[str], Iterable[tuple[str, str]]]


def _as_dotted_path(value: str) -> str:
    """A setuptools entry-point value as this registry's ``module:attr`` form.

    Object references normally already carry the colon; a bare ``pkg.mod.Attr`` splits at the
    *last* dot only. Replacing every dot (what this used to do) turned ``pkg.mod.Attr`` into
    ``pkg:mod:Attr``, which names no module at all. Latent, because the colon form is the norm.
    """
    if ":" in value:
        return value
    module, _, attr = value.rpartition(".")
    return f"{module}:{attr}"


def _missing_keywords(expected: object, actual: object) -> list[str]:
    """The port's keyword-only parameter names that ``actual`` cannot accept — empty when there is
    nothing to compare (a data attribute, or a callable with no introspectable signature)."""
    if not callable(expected) or not callable(actual):
        return []
    try:
        want, got = inspect.signature(expected), inspect.signature(actual)
    except (TypeError, ValueError):  # a builtin/C callable exposes no signature
        return []
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in got.parameters.values()):
        return []  # **kwargs absorbs anything
    return [name for name, parameter in want.parameters.items()
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY and name not in got.parameters]


def _default_entry_point_loader(group: str) -> list[tuple[str, str]]:
    """Discover ``(name, "module:attr")`` pairs published under an entry-point ``group``."""
    from importlib.metadata import entry_points
    return [(ep.name, ep.value) for ep in entry_points(group=group)]


class Registry(Generic[P]):
    """A named collection of components that all satisfy one ``Protocol``.

    Construct one per port (``VectorIndex``, ``Retriever``, ``Backend``, ...). The ``protocol``
    must be ``runtime_checkable``, because conformance is verified with ``isinstance`` at
    construction time — the universal check that works whether the port is defined by methods,
    attributes, or both.
    """

    def __init__(self, kind: str, protocol: type[P], *,
                 entry_point_group: str | None = None,
                 entry_point_loader: EntryPointLoader | None = None) -> None:
        if not getattr(protocol, "_is_runtime_protocol", False):
            raise RegistryError(
                f"the {kind!r} registry's protocol {protocol.__name__!r} is not "
                f"runtime_checkable, so components cannot be verified against it; decorate it "
                f"with @runtime_checkable")
        self._kind = kind
        self._protocol = protocol
        self._entry_point_group = entry_point_group
        self._load_entry_points = entry_point_loader or _default_entry_point_loader
        self._entries: dict[str, _Entry] = {}

    # -- registration --------------------------------------------------------

    def register(self, name: str, component: type, *, aliases: Iterable[str] = ()) -> type:
        """Register ``component`` under ``name`` and any ``aliases``. Returns it, so this can
        decorate a class.

        Refuses a component that cannot satisfy the port (checked structurally now, and again
        by ``isinstance`` on the built instance). A name collision with a *different* component
        is an error rather than a silent overwrite — two components answering to one name would
        make selection depend on import order; re-registering the *same* component is a no-op,
        so importing a module twice is harmless.
        """
        if not isinstance(component, type):
            raise RegistryError(
                f"a {self._kind} component must be a class, got {type(component).__name__}",
                )
        self._assert_structural_conformance(component)
        for key in (name, *aliases):
            lowered = key.lower()
            existing = self._entries.get(lowered)
            if existing is not None and existing.component is not component:
                raise RegistryError(
                    f"{self._kind} name {key!r} is already registered to "
                    f"{existing.component.__name__!r}; cannot also register "
                    f"{component.__name__!r}")
            self._entries[lowered] = _Entry(component=component, canonical_name=name)
        return component

    def _assert_structural_conformance(self, component: type) -> None:
        """Best-effort early check that ``component`` implements the port.

        Uses ``issubclass`` against the runtime-checkable protocol, which decides it for a
        method-only port. A port that also declares data members makes ``issubclass`` raise
        ``TypeError``; there the structural check is skipped and the authoritative
        ``isinstance`` check on the built instance (in :meth:`create`) is what enforces
        conformance — never silently, just later.
        """
        try:
            conforms = issubclass(component, self._protocol)
        except TypeError:
            return
        if not conforms:
            missing = self._missing_members(component)
            raise RegistryError(
                f"{component.__name__!r} does not satisfy the {self._protocol.__name__!r} "
                f"port required by the {self._kind} registry (missing: {sorted(missing)})")

    def _missing_members(self, component: type) -> set[str]:
        members: set[str] = getattr(self._protocol, "__protocol_attrs__", set())
        return {member for member in members if not hasattr(component, member)}

    def _signature_problems(self, component: type) -> list[str]:
        """Port methods the component has, but cannot be *called* the way the port says.

        ``runtime_checkable`` ``isinstance`` is signature-blind: it checks only that the named
        members exist. A driver whose ``search(self, vector, k)`` omits ``where=`` passed both
        that check and the ``create`` gate, then failed later with a ``TypeError`` deep inside
        ``search``, far from where the driver was chosen.

        Only **keyword-only** parameters are compared, which is where the risk actually is: the
        ports put every load-bearing option there (``k``, ``where``, ``min_score``), callers pass
        them by name, and so a missing or renamed one is a genuine break. Positional parameters
        are deliberately not name-checked -- renaming ``query`` to ``q`` is harmless and flagging
        it would be a false positive. A component taking ``**kwargs`` absorbs anything and is
        exempt.
        """
        problems = []
        for member in sorted(getattr(self._protocol, "__protocol_attrs__", set())):
            missing = _missing_keywords(getattr(self._protocol, member, None),
                                        getattr(component, member, None))
            if missing:
                problems.append(f"{member}() is missing {missing}")
        return problems

    # -- resolution ----------------------------------------------------------

    def create(self, spec: str, options: Mapping[str, Any] | None = None, *,
               path: Path | None = None) -> P:
        """Resolve ``spec`` to a component and build it from ``options``.

        ``spec`` is resolved in this precedence: an explicit dotted path (``"pkg.mod:Attr"``,
        recognised by the colon), then a registered built-in name, then — if this registry was
        given an entry-point group — a published entry point. The resolved component's own
        ``CONFIG_KEYS`` gate ``options`` (unknown keys refused), then ``from_config`` builds it,
        then the result is checked against the port: it must have every one of the port's members,
        and each of those must accept the port's keyword-only parameters. A component failing
        either raises :class:`RegistryError` here, where the driver was named — not later, from
        somewhere deep inside a call. That is a strong structural check, not a total one:
        parameter types, return values, and behaviour are not verified, so "satisfies the port"
        means it can be *called* as the port specifies, not that it does the right thing.
        """
        options = options or {}
        component, source = self._resolve(spec)
        label = f"{self._kind} {spec!r}"
        config_keys = self._config_keys(component, label=label)
        reject_unknown(options, config_keys, label=label, path=path)
        instance = self._build(component, options, label=label, path=path)
        if not isinstance(instance, self._protocol):
            raise RegistryError(
                f"{label} resolved to {type(instance).__name__!r} (via {source}), which does "
                f"not satisfy the {self._protocol.__name__!r} port")
        problems = self._signature_problems(type(instance))
        if problems:
            raise RegistryError(
                f"{label} resolved to {type(instance).__name__!r} (via {source}), which has the "
                f"{self._protocol.__name__!r} port's members but cannot be called as the port "
                f"specifies: {'; '.join(problems)}")
        return instance

    def _resolve(self, spec: str) -> tuple[type, str]:
        """The component for ``spec``, and a short phrase naming how it was found (for errors)."""
        if ":" in spec:
            return self._import_dotted(spec), "dotted path"
        lowered = spec.lower()
        if lowered in self._entries:
            return self._entries[lowered].component, "built-in name"
        component = self._from_entry_points(spec)
        if component is not None:
            return component, f"entry point {self._entry_point_group!r}"
        raise RegistryError(
            f"unknown {self._kind} {spec!r}; available: {', '.join(self.available()) or '(none)'}"
            + (f", or a dotted path 'module:attr', or an entry point under "
               f"{self._entry_point_group!r}" if self._entry_point_group else
               ", or a dotted path 'module:attr'"))

    def _import_dotted(self, spec: str) -> type:
        module_name, _, attr = spec.partition(":")
        if not module_name or not attr:
            raise RegistryError(
                f"malformed dotted path {spec!r}; expected 'module.path:AttributeName'")
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise RegistryError(
                f"could not import module {module_name!r} for {self._kind} {spec!r}: {exc}"
                ) from exc
        try:
            component = getattr(module, attr)
        except AttributeError as exc:
            raise RegistryError(
                f"module {module_name!r} has no attribute {attr!r} for {self._kind} {spec!r}"
                ) from exc
        if not isinstance(component, type):
            raise RegistryError(
                f"{spec!r} resolved to {type(component).__name__}, not a class; a {self._kind} "
                f"component must be a class")
        return component

    def _from_entry_points(self, name: str) -> type | None:
        if self._entry_point_group is None:
            return None
        for ep_name, value in self._load_entry_points(self._entry_point_group):
            if ep_name.lower() == name.lower():
                return self._import_dotted(_as_dotted_path(value))
        return None

    def _config_keys(self, component: type, *, label: str) -> frozenset[str]:
        keys: object = getattr(component, "CONFIG_KEYS", frozenset())
        if not isinstance(keys, (frozenset, set)):
            raise RegistryError(
                f"{label}: {component.__name__}.CONFIG_KEYS must be a set of option names, got "
                f"{type(keys).__name__}")
        return frozenset(keys)

    def _build(self, component: type, options: Mapping[str, Any], *, label: str,
               path: Path | None) -> Any:
        from_config = getattr(component, "from_config", None)
        if callable(from_config):
            try:
                return from_config(options)
            except (ConfigError, RegistryError):
                raise
            except (ValueError, TypeError) as exc:
                # A component's own dataclass invariant (a range check left to it, per the
                # config module's contract) raised. Surface it as a config error naming the
                # component rather than as a bare traceback from inside a constructor.
                raise ConfigError(f"{label}: {exc}", path=path) from exc
        if options:
            raise RegistryError(
                f"{label}: {component.__name__} defines no from_config classmethod but was "
                f"given options {sorted(options)}; add from_config to accept them")
        return component()

    def available(self) -> list[str]:
        """The distinct registered built-in names, sorted."""
        return sorted({entry.canonical_name for entry in self._entries.values()})
