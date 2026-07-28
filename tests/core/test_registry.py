"""The typed component registry — the extension API, so its guarantees are tested hard:
the three discovery paths, unknown-key rejection on a component's own options, protocol
conformance refused at registration, collision refused, and resolution staying pure."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pytest

from ragkit.core.config import ConfigError
from ragkit.core.registry import Registry, RegistryError


@runtime_checkable
class Greeter(Protocol):
    def greet(self) -> str: ...


class Hello:
    CONFIG_KEYS = frozenset({"name"})

    def __init__(self, name: str = "world") -> None:
        self.name = name

    @classmethod
    def from_config(cls, options: Any) -> Hello:
        return cls(name=options.get("name", "world"))

    def greet(self) -> str:
        return f"hello {self.name}"


class NoOptions:
    def greet(self) -> str:
        return "hi"


class NotAGreeter:
    def shout(self) -> str:
        return "!"


def _registry(**kwargs: Any) -> Registry[Greeter]:
    return Registry("greeter", Greeter, **kwargs)


class TestRegistrationConformance:
    def test_registers_and_creates_by_name(self) -> None:
        reg = _registry()
        reg.register("hello", Hello)
        assert reg.create("hello", {"name": "pl"}).greet() == "hello pl"

    def test_available_lists_canonical_names(self) -> None:
        reg = _registry()
        reg.register("hello", Hello, aliases=("hi",))
        assert reg.available() == ["hello"]  # the alias is resolvable but not a distinct name

    def test_alias_resolves(self) -> None:
        reg = _registry()
        reg.register("hello", Hello, aliases=("greetings",))
        assert reg.create("greetings").greet() == "hello world"

    def test_register_returns_the_class_so_it_can_decorate(self) -> None:
        reg = _registry()
        assert reg.register("hello", Hello) is Hello

    def test_a_non_conforming_component_is_refused_at_registration(self) -> None:
        reg = _registry()
        with pytest.raises(RegistryError, match=r"does not satisfy.*Greeter"):
            reg.register("bad", NotAGreeter)

    def test_a_non_class_is_refused(self) -> None:
        reg = _registry()
        with pytest.raises(RegistryError, match="must be a class"):
            reg.register("bad", lambda: None)  # type: ignore[arg-type]

    def test_a_non_runtime_checkable_protocol_is_refused_at_construction(self) -> None:
        class Plain(Protocol):
            def greet(self) -> str: ...

        with pytest.raises(RegistryError, match="not runtime_checkable"):
            Registry("greeter", Plain)


class TestCollisions:
    def test_a_different_component_under_a_taken_name_is_refused(self) -> None:
        reg = _registry()
        reg.register("g", Hello)

        class OtherGreeter:
            def greet(self) -> str:
                return "hola"

        with pytest.raises(RegistryError, match="already registered"):
            reg.register("g", OtherGreeter)

    def test_re_registering_the_same_component_is_a_no_op(self) -> None:
        reg = _registry()
        reg.register("g", Hello)
        reg.register("g", Hello)  # idempotent
        assert reg.create("g").greet() == "hello world"


class TestOptionValidation:
    def test_unknown_option_key_is_refused(self) -> None:
        reg = _registry()
        reg.register("hello", Hello)
        with pytest.raises(ConfigError, match=r"unknown key\(s\) \['bogus'\]"):
            reg.create("hello", {"bogus": 1})

    def test_a_component_with_no_from_config_takes_no_options(self) -> None:
        reg = _registry()
        reg.register("plain", NoOptions)
        assert reg.create("plain").greet() == "hi"

    def test_options_for_an_optionless_component_are_refused(self) -> None:
        reg = _registry()
        reg.register("plain", NoOptions)
        # NoOptions has no CONFIG_KEYS, so any key is unknown -> ConfigError first.
        with pytest.raises(ConfigError, match=r"unknown key\(s\)"):
            reg.create("plain", {"x": 1})

    def test_from_config_value_error_becomes_a_config_error(self) -> None:
        class Strict:
            CONFIG_KEYS = frozenset({"n"})

            def __init__(self, n: int) -> None:
                if n < 0:
                    raise ValueError("n must be >= 0")
                self.n = n

            @classmethod
            def from_config(cls, options: Any) -> Strict:
                return cls(n=options.get("n", 0))

            def greet(self) -> str:
                return "ok"

        reg = _registry()
        reg.register("strict", Strict)
        with pytest.raises(ConfigError, match="n must be >= 0"):
            reg.create("strict", {"n": -1})

    def test_bad_config_keys_type_is_refused(self) -> None:
        class WrongKeys:
            CONFIG_KEYS = ["name"]  # noqa: RUF012  -- deliberately the wrong type, under test

            def greet(self) -> str:
                return "x"

        reg = _registry()
        reg.register("wrong", WrongKeys)
        with pytest.raises(RegistryError, match="CONFIG_KEYS must be a set"):
            reg.create("wrong")


class TestDottedPathResolution:
    def test_resolves_a_dotted_path(self) -> None:
        reg = _registry()
        # This very test module is importable; Hello lives in it.
        spec = f"{__name__}:Hello"
        assert reg.create(spec, {"name": "z"}).greet() == "hello z"

    def test_malformed_dotted_path_is_refused(self) -> None:
        reg = _registry()
        with pytest.raises(RegistryError, match="malformed dotted path"):
            reg.create(":Hello")

    def test_unimportable_module_is_named(self) -> None:
        reg = _registry()
        with pytest.raises(RegistryError, match="could not import module"):
            reg.create("ragkit_nonexistent.mod:Thing")

    def test_missing_attribute_is_named(self) -> None:
        reg = _registry()
        with pytest.raises(RegistryError, match="has no attribute 'Absent'"):
            reg.create(f"{__name__}:Absent")

    def test_dotted_path_to_a_non_class_is_refused(self) -> None:
        reg = _registry()
        with pytest.raises(RegistryError, match="not a class"):
            reg.create(f"{__name__}:a_module_level_value")

    def test_dotted_path_component_that_violates_the_port_is_refused(self) -> None:
        reg = _registry()
        # NotAGreeter has no CONFIG_KEYS and no greet(); build succeeds but isinstance fails.
        with pytest.raises(RegistryError, match="does not satisfy"):
            reg.create(f"{__name__}:NotAGreeter")


class TestEntryPointResolution:
    def test_resolves_via_an_injected_entry_point_loader(self) -> None:
        def loader(group: str) -> list[tuple[str, str]]:
            assert group == "ragkit.greeters"
            return [("plug", f"{__name__}:Hello")]

        reg = _registry(entry_point_group="ragkit.greeters", entry_point_loader=loader)
        assert reg.create("plug", {"name": "ep"}).greet() == "hello ep"

    def test_unknown_spec_lists_the_entry_point_group(self) -> None:
        reg = _registry(entry_point_group="ragkit.greeters", entry_point_loader=lambda g: [])
        with pytest.raises(RegistryError, match=r"entry point under 'ragkit\.greeters'"):
            reg.create("absent")

    def test_unknown_spec_without_entry_points_still_helps(self) -> None:
        reg = _registry()
        reg.register("hello", Hello)
        with pytest.raises(RegistryError, match=r"unknown greeter 'absent'.*available: hello"):
            reg.create("absent")

    def test_a_non_matching_entry_point_is_passed_over(self) -> None:
        reg = _registry(entry_point_group="ragkit.greeters",
                        entry_point_loader=lambda g: [("other", f"{__name__}:Hello")])
        with pytest.raises(RegistryError, match="unknown greeter 'absent'"):
            reg.create("absent")

    def test_the_default_entry_point_loader_runs_when_none_is_injected(self) -> None:
        # No loader injected: resolution falls through to importlib.metadata.entry_points for a
        # group nothing publishes, which yields nothing -> the usual unknown-spec error.
        reg = _registry(entry_point_group="ragkit.nonexistent.group")
        with pytest.raises(RegistryError, match="unknown greeter 'absent'"):
            reg.create("absent")


@runtime_checkable
class Named(Protocol):
    # A port with a data member: issubclass() cannot check it, so the registry's early
    # structural check is skipped and the isinstance backstop enforces conformance instead.
    label: str

    def greet(self) -> str: ...


class TestDataMemberProtocolBackstop:
    def test_registration_skips_the_unusable_structural_check(self) -> None:
        class WithLabel:
            label = "x"

            def greet(self) -> str:
                return "hi"

        reg: Registry[Named] = Registry("named", Named)
        reg.register("wl", WithLabel)  # issubclass raises TypeError -> early return, no error
        assert reg.create("wl").greet() == "hi"

    def test_isinstance_backstop_rejects_a_non_conforming_component(self) -> None:
        class MissingLabel:
            def greet(self) -> str:
                return "hi"

        reg: Registry[Named] = Registry("named", Named)
        reg.register("ml", MissingLabel)  # passes the skipped structural check
        with pytest.raises(RegistryError, match="does not satisfy"):
            reg.create("ml")  # ... but the isinstance check catches the missing attribute


class TestBuildPaths:
    def test_from_config_registry_error_propagates(self) -> None:
        class Composite:
            CONFIG_KEYS = frozenset()

            @classmethod
            def from_config(cls, options: Any) -> Composite:
                raise RegistryError("a sub-component could not be built")

            def greet(self) -> str:
                return "x"

        reg = _registry()
        reg.register("composite", Composite)
        with pytest.raises(RegistryError, match="sub-component could not be built"):
            reg.create("composite")

    def test_declared_keys_without_from_config_is_refused(self) -> None:
        class NeedsBuilder:
            CONFIG_KEYS = frozenset({"x"})  # accepts an option but has no way to consume it

            def greet(self) -> str:
                return "x"

        reg = _registry()
        reg.register("needs", NeedsBuilder)
        with pytest.raises(RegistryError, match="defines no from_config"):
            reg.create("needs", {"x": 1})


a_module_level_value = 42
