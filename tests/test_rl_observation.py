import dataclasses

import numpy as np
import pytest

from hexrl_platform.rl.hex_grid import HexCoord, HexMap
from hexrl_platform.rl.navigation import NavigationObservation
from hexrl_platform.rl.observation import ENCODER_VERSION, ObservationEncoder
from hexrl_platform.rl.replay import default_map


def make_encoder() -> ObservationEncoder:
    return ObservationEncoder(default_map(), max_steps=60)


def test_encoded_observation_fits_declared_space():
    encoder = make_encoder()
    space = encoder.observation_space
    for position in encoder.hex_map.cells:
        for goal in (HexCoord(4, -4), HexCoord(-4, 4), HexCoord(0, 0)):
            if goal == position or not encoder.hex_map.is_walkable(goal):
                continue
            encoded = encoder.encode(NavigationObservation(position, goal, steps_remaining=60))
            assert space.contains(encoded)


def test_encoding_is_deterministic_and_normalized():
    encoder = make_encoder()
    observation = NavigationObservation(HexCoord(-4, 0), HexCoord(4, 0), steps_remaining=15)
    first = encoder.encode(observation)
    second = encoder.encode(observation)
    assert first.keys() == second.keys()
    assert all(np.array_equal(first[key], second[key]) for key in first)
    np.testing.assert_allclose(first["position"], [-1.0, 0.0])
    np.testing.assert_allclose(first["goal_delta"], [1.0, 0.0])
    np.testing.assert_allclose(first["steps_left"], [0.25])


def test_walkable_neighbors_follow_map_obstacles_and_border():
    hex_map = HexMap(1, frozenset({HexCoord(1, 0)}))
    encoder = ObservationEncoder(hex_map, max_steps=5)
    encoded = encoder.encode(NavigationObservation(HexCoord(0, 0), HexCoord(-1, 0), 5))
    # Action 0 points at the blocked (1, 0) cell, the rest stay inside radius 1
    np.testing.assert_array_equal(encoded["walkable_neighbors"], [0, 1, 1, 1, 1, 1])

    corner = encoder.encode(NavigationObservation(HexCoord(-1, 0), HexCoord(0, 0), 5))
    assert corner["walkable_neighbors"].sum() == 3


def test_encoding_does_not_leak_evaluation_only_information():
    encoded = make_encoder().encode(NavigationObservation(HexCoord(0, 0), HexCoord(3, 0), 60))
    assert set(encoded) == {"position", "goal", "goal_delta", "steps_left", "walkable_neighbors"}


def test_encoder_config_round_trip_and_version_check():
    encoder = make_encoder()
    config = encoder.to_config()
    assert config["version"] == ENCODER_VERSION
    assert ObservationEncoder.from_config(config) == encoder

    with pytest.raises(ValueError, match="version"):
        ObservationEncoder.from_config({**config, "version": "nav-obs-v0"})


def test_encoder_rejects_invalid_step_limit():
    with pytest.raises(ValueError, match="max_steps"):
        ObservationEncoder(default_map(), max_steps=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        make_encoder().max_steps = 10  # ty: ignore[invalid-assignment]
