"""SamplingParams: range validation at construction, and the request-body payload it builds."""
from __future__ import annotations

import pytest

from ragkit.core.ports import SamplingParams


class TestValidation:
    def test_defaults_are_valid_and_low_temperature(self) -> None:
        assert SamplingParams().temperature == 0.2

    def test_negative_temperature_refused(self) -> None:
        with pytest.raises(ValueError, match="temperature must be >= 0"):
            SamplingParams(temperature=-0.1)

    @pytest.mark.parametrize("field", ["top_p", "min_p"])
    def test_fraction_knobs_bounded_to_unit_interval(self, field: str) -> None:
        with pytest.raises(ValueError, match=f"{field} must be in"):
            SamplingParams(**{field: 1.5})
        assert getattr(SamplingParams(**{field: 0.0}), field) == 0.0  # boundary accepted

    @pytest.mark.parametrize("field", ["presence_penalty", "frequency_penalty"])
    def test_penalties_bounded(self, field: str) -> None:
        with pytest.raises(ValueError, match=f"{field} must be in"):
            SamplingParams(**{field: 2.5})
        with pytest.raises(ValueError, match=f"{field} must be in"):
            SamplingParams(**{field: -2.5})

    def test_negative_top_k_refused_but_zero_disables(self) -> None:
        with pytest.raises(ValueError, match="top_k must be >= 0"):
            SamplingParams(top_k=-1)
        assert SamplingParams(top_k=0).top_k == 0

    def test_nonpositive_repeat_penalty_refused(self) -> None:
        with pytest.raises(ValueError, match="repeat_penalty must be > 0"):
            SamplingParams(repeat_penalty=0.0)

    def test_is_hashable_and_frozen(self) -> None:
        # A persona holds one; frozen + hashable means it cannot drift and can key a cache.
        params = SamplingParams(temperature=0.5, stop=("x",))
        assert hash(params) == hash(SamplingParams(temperature=0.5, stop=("x",)))
        with pytest.raises(AttributeError):
            params.temperature = 0.9  # type: ignore[misc]


class TestPayload:
    def test_only_temperature_by_default(self) -> None:
        assert SamplingParams().payload() == {"temperature": 0.2}

    def test_set_knobs_are_included_unset_omitted(self) -> None:
        payload = SamplingParams(temperature=0.7, top_p=0.9, seed=7).payload()
        assert payload == {"temperature": 0.7, "top_p": 0.9, "seed": 7}

    def test_stop_is_a_list(self) -> None:
        assert SamplingParams(stop=("a", "b")).payload()["stop"] == ["a", "b"]

    def test_zero_valued_knob_is_still_sent(self) -> None:
        # 0 is a deliberate value (top_k=0 disables top-k), not "unset" — it must reach the server.
        assert SamplingParams(top_k=0, min_p=0.0).payload() == {
            "temperature": 0.2, "top_k": 0, "min_p": 0.0}
