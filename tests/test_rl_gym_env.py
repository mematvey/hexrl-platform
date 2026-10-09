import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from hexrl_platform.rl.gym_env import HexNavigationEnv
from hexrl_platform.rl.hex_grid import HexCoord
from hexrl_platform.rl.navigation import NavigationEnv, NavigationTask
from hexrl_platform.rl.observation import ObservationEncoder
from hexrl_platform.rl.replay import default_map


def make_tasks(max_steps: int = 12) -> list[NavigationTask]:
    hex_map = default_map()
    pairs = [
        (HexCoord(-3, 0), HexCoord(3, 0)),
        (HexCoord(0, -3), HexCoord(0, 3)),
        (HexCoord(2, 1), HexCoord(-2, -1)),
    ]
    return [NavigationTask(hex_map, start, goal, max_steps) for start, goal in pairs]


def make_env(max_steps: int = 12) -> HexNavigationEnv:
    return HexNavigationEnv(make_tasks(max_steps), ObservationEncoder(default_map(), max_steps))


def test_env_follows_gymnasium_and_sb3_contracts():
    check_env(make_env(), skip_render_check=True)
    sb3_check_env(make_env(), warn=True)


def test_same_seed_gives_same_task_sequence():
    first, second = make_env(), make_env()
    starts_first = [first.reset(seed=3)[1]["start"]] + [first.reset()[1]["start"] for _ in range(9)]
    starts_second = [second.reset(seed=3)[1]["start"]] + [
        second.reset()[1]["start"] for _ in range(9)
    ]
    assert starts_first == starts_second


def test_adapter_delegates_rules_to_navigation_env():
    task = make_tasks()[0]
    adapter = make_env()
    _, info = adapter.reset(options={"task": task})
    assert info["optimal_steps"] == task.hex_map.shortest_distance(task.start, task.goal)
    reference = NavigationEnv(task)
    for action in [0, 0, 3, 1, 5, 0, 0, 0]:
        expected = reference.step(action)
        observation, reward, terminated, truncated, step_info = adapter.step(action)
        position = expected.observation.position
        assert step_info["position"] == (position.q, position.r)
        assert reward == expected.reward
        assert (terminated, truncated) == (expected.terminated, expected.truncated)
        assert step_info["blocked_move"] == expected.blocked_move
        expected_observation = adapter.encoder.encode(expected.observation)
        assert all(
            np.array_equal(observation[key], expected_observation[key]) for key in observation
        )
        if terminated or truncated:
            break


def test_time_limit_is_truncation_not_termination():
    adapter = make_env(max_steps=2)
    adapter.reset(options={"task": make_tasks(max_steps=2)[0]})
    adapter.step(3)
    *_, terminated, truncated, info = adapter.step(3)
    assert truncated and not terminated
    assert not info["is_success"]


def test_adapter_validates_tasks_and_call_order():
    encoder = ObservationEncoder(default_map(), max_steps=12)
    with pytest.raises(ValueError, match="At least one"):
        HexNavigationEnv([], encoder)
    with pytest.raises(ValueError, match="max_steps"):
        HexNavigationEnv(make_tasks(max_steps=30), encoder)
    with pytest.raises(RuntimeError, match="reset"):
        make_env().step(0)
