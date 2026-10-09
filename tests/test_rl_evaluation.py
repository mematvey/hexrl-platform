import math

import pytest

from hexrl_platform.rl.evaluation import EpisodeResult, evaluate_policy, summarize_results
from hexrl_platform.rl.hex_grid import HexCoord, HexMap
from hexrl_platform.rl.navigation import NavigationObservation, NavigationTask
from hexrl_platform.rl.policies import Policy, RandomPolicy, ShortestPathPolicy
from hexrl_platform.rl.replay import default_map


class AlwaysAction(Policy):
    def __init__(self, action: int) -> None:
        self.action = action
        self.calls = 0

    def act(self, observation: NavigationObservation) -> int:
        self.calls += 1
        return self.action


def make_tasks(hex_map: HexMap, max_steps: int = 30) -> list[NavigationTask]:
    pairs = [
        (HexCoord(0, 2), HexCoord(0, -1)),
        (HexCoord(-3, 1), HexCoord(3, -1)),
        (HexCoord(2, 0), HexCoord(-2, 0)),
    ]
    return [NavigationTask(hex_map, start, goal, max_steps) for start, goal in pairs]


def test_optimal_policy_solves_every_task_on_the_shortest_path():
    hex_map = default_map()
    tasks = make_tasks(hex_map)
    results = evaluate_policy(ShortestPathPolicy(hex_map), tasks)

    assert len(results) == len(tasks)
    for task, result in zip(tasks, results, strict=True):
        assert (result.start, result.goal) == (task.start, task.goal)
        assert result.optimal_steps == hex_map.shortest_distance(task.start, task.goal)
        assert result.success
        assert result.length == result.optimal_steps
        assert result.blocked_moves == 0
        assert result.trajectory[0] == task.start
        assert result.trajectory[-1] == task.goal
        assert len(result.trajectory) == result.length + 1
        expected_return = result.length * task.step_penalty + task.success_reward
        assert result.total_reward == pytest.approx(expected_return)


def test_failed_episode_runs_until_the_step_limit_and_records_wall_hits():
    hex_map = HexMap(1, frozenset({HexCoord(1, 0)}))
    task = NavigationTask(hex_map, HexCoord(0, 0), HexCoord(-1, 0), max_steps=5)
    policy = AlwaysAction(0)  # always walks into the blocked (1, 0) cell

    (result,) = evaluate_policy(policy, [task])

    assert policy.calls == 5
    assert not result.success
    assert result.length == 5
    assert result.blocked_moves == 5
    assert result.trajectory == (HexCoord(0, 0),) * 6
    assert result.total_reward == pytest.approx(5 * (task.step_penalty + task.blocked_penalty))


def test_evaluation_with_the_same_seed_is_reproducible():
    hex_map = default_map()
    tasks = make_tasks(hex_map, max_steps=20)
    assert evaluate_policy(RandomPolicy(seed=4), tasks) == evaluate_policy(
        RandomPolicy(seed=4), tasks
    )


def make_result(*, success: bool, length: int, optimal: int, reward: float, blocked: int):
    return EpisodeResult(
        start=HexCoord(0, 0),
        goal=HexCoord(2, 0),
        optimal_steps=optimal,
        success=success,
        length=length,
        total_reward=reward,
        blocked_moves=blocked,
        trajectory=(HexCoord(0, 0),) * (length + 1),
    )


def test_summary_metrics():
    results = [
        make_result(success=True, length=4, optimal=2, reward=0.96, blocked=0),
        make_result(success=True, length=3, optimal=3, reward=0.97, blocked=1),
        make_result(success=False, length=10, optimal=2, reward=-0.5, blocked=3),
    ]
    metrics = summarize_results(results)
    assert metrics["episodes"] == 3
    assert metrics["success_rate"] == pytest.approx(2 / 3)
    assert metrics["mean_return"] == pytest.approx((0.96 + 0.97 - 0.5) / 3)
    assert metrics["mean_length"] == pytest.approx(17 / 3)
    # Only successful episodes have a meaningful path length ratio
    assert metrics["path_ratio"] == pytest.approx((4 / 2 + 3 / 3) / 2)
    # Wall hits are counted per step, like in the replay EDA
    assert metrics["blocked_move_rate"] == pytest.approx(4 / 17)


def test_path_ratio_is_nan_without_successful_episodes():
    metrics = summarize_results(
        [make_result(success=False, length=5, optimal=2, reward=-1, blocked=0)]
    )
    assert metrics["success_rate"] == 0
    assert math.isnan(metrics["path_ratio"])


def test_summary_rejects_empty_results():
    with pytest.raises(ValueError, match="empty"):
        summarize_results([])
