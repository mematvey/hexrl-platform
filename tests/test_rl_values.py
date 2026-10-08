import math

import pytest

from hexrl_platform.rl.values import Probability


@pytest.mark.parametrize("value", [0.0, 0.25, 1.0])
def test_probability_accepts_values_in_unit_interval(value):
    assert Probability(value).value == value


@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan, math.inf, -math.inf])
def test_probability_rejects_values_outside_unit_interval(value):
    with pytest.raises(ValueError):
        Probability(value)


def test_probability_is_immutable():
    probability = Probability(0.5)
    with pytest.raises(AttributeError):
        probability.value = 0.7  # ty: ignore[invalid-assignment]


def test_probability_compares_and_hashes_by_value():
    assert Probability(0.3) == Probability(0.3)
    assert Probability(0.3) != Probability(0.4)
    assert len({Probability(0.3), Probability(0.3)}) == 1
