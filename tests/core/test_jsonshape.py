"""JSON value-type checking: the bool-is-not-a-number rule, unions, and the unknown-name refusal."""
from __future__ import annotations

import pytest

from ragkit.core.jsonshape import JSON_TYPE_CHECKS, JsonShapeError, json_type_matches


class TestKnownTypes:
    @pytest.mark.parametrize(("value", "declared"), [
        ("s", "string"), (1, "integer"), (1, "number"), (1.5, "number"),
        (True, "boolean"), ([1], "array"), ({"k": 1}, "object"), (None, "null"),
    ])
    def test_a_matching_value_passes(self, value: object, declared: str) -> None:
        assert json_type_matches(value, declared)

    @pytest.mark.parametrize(("value", "declared"), [
        (1, "string"), ("1", "integer"), ("1", "number"), (1, "boolean"),
        ({"k": 1}, "array"), ([1], "object"), (0, "null"),
    ])
    def test_a_mismatching_value_fails(self, value: object, declared: str) -> None:
        assert not json_type_matches(value, declared)

    def test_a_bool_is_not_an_integer_or_a_number(self) -> None:
        # Python makes bool a subclass of int; JSON does not. This is the rule the module exists
        # to keep in one place.
        assert not json_type_matches(True, "integer")
        assert not json_type_matches(False, "number")

    def test_a_union_matches_any_member(self) -> None:
        assert json_type_matches(None, ["array", "null"])
        assert json_type_matches([1], ["array", "null"])
        assert not json_type_matches("s", ["array", "null"])


class TestUnknownTypesAreRefused:
    """Regression: an unrecognised type name returned True for every value, so a schema author's
    typo silently switched that field's check off -- the check would then report the field fine no
    matter what came back, which is worse than not having the check at all."""

    def test_a_typo_raises(self) -> None:
        with pytest.raises(JsonShapeError, match="unknown JSON type 'boolena'"):
            json_type_matches(True, "boolena")

    def test_the_error_lists_the_known_types(self) -> None:
        with pytest.raises(JsonShapeError, match="known types are"):
            json_type_matches(1, "str")

    def test_a_non_string_declaration_raises(self) -> None:
        with pytest.raises(JsonShapeError, match="unknown JSON type"):
            json_type_matches(1, 7)

    def test_a_union_containing_an_unknown_name_raises(self) -> None:
        # Not swallowed by the union's `any()`: a typo inside a union is still a typo.
        with pytest.raises(JsonShapeError, match="unknown JSON type 'nul'"):
            json_type_matches("s", ["nul"])

    def test_every_json_type_has_a_name_here(self) -> None:
        assert set(JSON_TYPE_CHECKS) == {
            "string", "integer", "number", "boolean", "array", "object", "null"}
