from collections.abc import Sequence
from dataclasses import dataclass

from hexrl_platform.rl.hex_grid import HexCoord
from hexrl_platform.rl.navigation import NavigationEnv, NavigationTask
from hexrl_platform.rl.policies import Policy


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    start: HexCoord
    goal: HexCoord
    optimal_steps: int
    success: bool
    length: int
    total_reward: float
    blocked_moves: int
    trajectory: tuple[HexCoord, ...]


def run_episode(policy: Policy, task: NavigationTask) -> EpisodeResult:
    optimal_steps = task.hex_map.shortest_distance(task.start, task.goal)
    if optimal_steps is None:
        raise ValueError("Task goal is unreachable")
    environment = NavigationEnv(task)
    trajectory = [task.start]
    total_reward = 0.0
    blocked_moves = 0
    success = False
    while not environment.done:
        result = environment.step(policy.act(environment.observe()))
        trajectory.append(result.observation.position)
        total_reward += result.reward
        blocked_moves += result.blocked_move
        success = result.terminated
    return EpisodeResult(
        start=task.start,
        goal=task.goal,
        optimal_steps=optimal_steps,
        success=success,
        length=environment.steps_taken,
        total_reward=total_reward,
        blocked_moves=blocked_moves,
        trajectory=tuple(trajectory),
    )


def evaluate_policy(policy: Policy, tasks: Sequence[NavigationTask]) -> list[EpisodeResult]:
    return [run_episode(policy, task) for task in tasks]


def summarize_results(results: Sequence[EpisodeResult]) -> dict[str, float]:
    if not results:
        raise ValueError("Cannot summarize an empty list of episodes")
    episodes = len(results)
    steps = sum(result.length for result in results)
    successful = [result for result in results if result.success]
    path_ratio = (
        sum(result.length / result.optimal_steps for result in successful) / len(successful)
        if successful
        else float("nan")
    )
    return {
        "episodes": float(episodes),
        "success_rate": len(successful) / episodes,
        "mean_return": sum(result.total_reward for result in results) / episodes,
        "mean_length": steps / episodes,
        "path_ratio": path_ratio,
        "blocked_move_rate": sum(result.blocked_moves for result in results) / steps,
    }
