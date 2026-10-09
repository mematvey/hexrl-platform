from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hexrl_platform.rl.hex_grid import DIRECTIONS
from hexrl_platform.rl.navigation import NavigationEnv, NavigationTask
from hexrl_platform.rl.observation import ObservationEncoder


class HexNavigationEnv(gym.Env[dict[str, Any], np.int64]):
    """Gymnasium adapter; movement rules stay in NavigationEnv."""

    metadata = {"render_modes": []}  # noqa: RUF012 - Gymnasium reads it as a class attribute

    def __init__(self, tasks: Sequence[NavigationTask], encoder: ObservationEncoder) -> None:
        if not tasks:
            raise ValueError("At least one task is required")
        if any(task.hex_map != encoder.hex_map for task in tasks):
            raise ValueError("All tasks must use the encoder map")
        if any(task.max_steps != encoder.max_steps for task in tasks):
            raise ValueError("All tasks must use the encoder max_steps")
        self.tasks = tuple(tasks)
        self.encoder = encoder
        self.action_space = spaces.Discrete(len(DIRECTIONS))
        self.observation_space = encoder.observation_space
        self._navigation: NavigationEnv | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if options is not None and "task" in options:
            task = options["task"]
        else:
            task = self.tasks[int(self.np_random.integers(len(self.tasks)))]
        self._navigation = NavigationEnv(task)
        info = {
            "start": (task.start.q, task.start.r),
            "goal": (task.goal.q, task.goal.r),
            "optimal_steps": task.hex_map.shortest_distance(task.start, task.goal),
        }
        return self.encoder.encode(self._navigation.observe()), info

    def step(
        self, action: np.int64 | int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._navigation is None:
            raise RuntimeError("Call reset before step")
        result = self._navigation.step(int(action))
        position = result.observation.position
        info = {
            "blocked_move": result.blocked_move,
            "is_success": result.terminated,
            "position": (position.q, position.r),
        }
        return (
            self.encoder.encode(result.observation),
            result.reward,
            result.terminated,
            result.truncated,
            info,
        )
