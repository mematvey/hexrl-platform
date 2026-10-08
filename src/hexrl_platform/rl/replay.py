import argparse
import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hexrl_platform.core.config import APP_VERSION
from hexrl_platform.rl.hex_grid import DIRECTIONS, HexCoord, HexMap
from hexrl_platform.rl.navigation import NavigationEnv, NavigationObservation, NavigationTask
from hexrl_platform.rl.values import Probability

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ReplayGenerationConfig:
    episodes: int = 1000
    seed: int = 42
    max_steps: int = 60
    random_fraction: Probability = Probability(0.5)
    exploration_probability: Probability = Probability(0.25)

    def __post_init__(self) -> None:
        if self.episodes < 1:
            raise ValueError("episodes must be positive")


def default_map(radius: int = 4) -> HexMap:
    obstacles = (
        HexCoord(-2, 1),
        HexCoord(-1, 1),
        HexCoord(0, 1),
        HexCoord(1, -2),
        HexCoord(2, -2),
    )
    return HexMap(
        radius, frozenset(cell for cell in obstacles if cell.distance_to(HexCoord(0, 0)) <= radius)
    )


def observation_dict(observation: NavigationObservation) -> dict[str, int]:
    return {
        "q": observation.position.q,
        "r": observation.position.r,
        "goal_q": observation.goal.q,
        "goal_r": observation.goal.r,
        "steps_remaining": observation.steps_remaining,
    }


def action_probabilities(
    hex_map: HexMap,
    position: HexCoord,
    goal: HexCoord,
    policy: str,
    exploration_probability: Probability,
    distances_to_goal: dict[HexCoord, int],
) -> tuple[float, ...]:
    if policy == "random":
        return (1 / len(DIRECTIONS),) * len(DIRECTIONS)
    if policy != "goal_directed":
        raise ValueError(f"Unknown behavior policy: {policy}")

    candidate_distances = [
        distances_to_goal.get(HexCoord(position.q + direction.q, position.r + direction.r))
        if hex_map.is_walkable(HexCoord(position.q + direction.q, position.r + direction.r))
        else None
        for direction in DIRECTIONS
    ]
    reachable = [distance for distance in candidate_distances if distance is not None]
    if not reachable:
        raise ValueError("Current position has no route to goal")
    best_distance = min(reachable)
    best_actions = [
        index for index, distance in enumerate(candidate_distances) if distance == best_distance
    ]
    exploration = exploration_probability.value
    probabilities = [exploration / len(DIRECTIONS)] * len(DIRECTIONS)
    for action in best_actions:
        probabilities[action] += (1 - exploration) / len(best_actions)
    return tuple(probabilities)


def generate_replays(
    output: Path,
    config: ReplayGenerationConfig,
    hex_map: HexMap | None = None,
) -> dict[str, Any]:
    hex_map = hex_map or default_map()
    candidate_tasks = [
        (start, goal, distance)
        for start in hex_map.cells
        for goal, distance in hex_map.distances_from(start).items()
        if distance >= 2
    ]
    if not candidate_tasks:
        raise ValueError("Map must contain a reachable start-goal pair at least two moves apart")

    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    digest = hashlib.sha256()
    transition_count = 0
    success_count = 0
    policy_counts = {"random": 0, "goal_directed": 0}
    goal_distances: dict[HexCoord, dict[HexCoord, int]] = {}

    with tempfile.NamedTemporaryFile(mode="wb", dir=output.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            for episode_id in range(config.episodes):
                start, goal, optimal_steps = rng.choice(candidate_tasks)
                policy = (
                    "random" if rng.random() < config.random_fraction.value else "goal_directed"
                )
                policy_counts[policy] += 1
                task = NavigationTask(hex_map, start, goal, config.max_steps)
                environment = NavigationEnv(task)
                if goal not in goal_distances:
                    goal_distances[goal] = hex_map.distances_from(goal)
                distances = goal_distances[goal]

                for step_id in range(config.max_steps):
                    observation = environment.observe()
                    probabilities = action_probabilities(
                        hex_map,
                        observation.position,
                        goal,
                        policy,
                        config.exploration_probability,
                        distances,
                    )
                    action = rng.choices(range(len(DIRECTIONS)), weights=probabilities, k=1)[0]
                    result = environment.step(action)
                    record = {
                        "episode_id": episode_id,
                        "step_id": step_id,
                        "behavior_policy": policy,
                        "state": observation_dict(observation),
                        "action": action,
                        "action_probability": probabilities[action],
                        "reward": result.reward,
                        "next_state": observation_dict(result.observation),
                        "terminated": result.terminated,
                        "truncated": result.truncated,
                        "blocked_move": result.blocked_move,
                        "optimal_steps": optimal_steps,
                    }
                    encoded = (
                        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                    ).encode()
                    temporary.write(encoded)
                    digest.update(encoded)
                    transition_count += 1
                    if result.terminated or result.truncated:
                        success_count += int(result.terminated)
                        break
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary.close()
            os.replace(temporary_path, output)
        finally:
            temporary.close()
            temporary_path.unlink(missing_ok=True)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": "generated://hexrl/navigation",
        "generator": "hexrl_platform.rl.replay:generate_replays",
        "simulator_version": APP_VERSION,
        "dataset_file": output.name,
        "sha256": digest.hexdigest(),
        "episodes": config.episodes,
        "transitions": transition_count,
        "successful_episodes": success_count,
        "seed": config.seed,
        "max_steps": config.max_steps,
        "map": {
            "radius": hex_map.radius,
            "blocked": [{"q": cell.q, "r": cell.r} for cell in sorted(hex_map.blocked)],
        },
        "rewards": {
            "step_penalty": task.step_penalty,
            "blocked_penalty": task.blocked_penalty,
            "success_reward": task.success_reward,
        },
        "behavior_policy": {
            "episode_counts": policy_counts,
            "random_fraction": config.random_fraction.value,
            "exploration_probability": config.exploration_probability.value,
        },
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reproducible hex-navigation replays")
    parser.add_argument("--output", type=Path, default=Path("data/replays/navigation.jsonl"))
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--random-fraction", type=float, default=0.5)
    parser.add_argument("--exploration-probability", type=float, default=0.25)
    args = parser.parse_args()
    config = ReplayGenerationConfig(
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        random_fraction=Probability(args.random_fraction),
        exploration_probability=Probability(args.exploration_probability),
    )
    manifest = generate_replays(args.output, config, default_map(args.radius))
    print(
        f"Generated {manifest['episodes']} episodes, {manifest['transitions']} transitions, "
        f"SHA-256 {manifest['sha256']}"
    )


if __name__ == "__main__":
    main()
