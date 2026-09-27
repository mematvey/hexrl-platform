from dataclasses import dataclass

from hexrl_platform.rl.hex_grid import DIRECTIONS, HexCoord, HexMap


@dataclass(frozen=True, slots=True)
class NavigationTask:
    hex_map: HexMap
    start: HexCoord
    goal: HexCoord
    max_steps: int
    step_penalty: float = -0.01
    blocked_penalty: float = -0.1
    success_reward: float = 1.0

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.start == self.goal:
            raise ValueError("Start and goal must differ")
        if not self.hex_map.is_walkable(self.goal):
            raise ValueError("Goal must be walkable")
        if self.hex_map.shortest_distance(self.start, self.goal) is None:
            raise ValueError("Goal must be reachable from start")


@dataclass(frozen=True, slots=True)
class NavigationObservation:
    position: HexCoord
    goal: HexCoord
    steps_remaining: int


@dataclass(frozen=True, slots=True)
class StepResult:
    observation: NavigationObservation
    reward: float
    terminated: bool
    truncated: bool
    blocked_move: bool


class NavigationEnv:
    def __init__(self, task: NavigationTask) -> None:
        self.task = task
        self.position = task.start
        self.steps_taken = 0
        self.done = False

    def observe(self) -> NavigationObservation:
        return NavigationObservation(
            position=self.position,
            goal=self.task.goal,
            steps_remaining=self.task.max_steps - self.steps_taken,
        )

    def step(self, action: int) -> StepResult:
        if self.done:
            raise RuntimeError("Episode has ended")
        if not 0 <= action < len(DIRECTIONS):
            raise ValueError("Action must be a direction index from 0 to 5")

        direction = DIRECTIONS[action]
        target = HexCoord(self.position.q + direction.q, self.position.r + direction.r)
        blocked_move = not self.task.hex_map.is_walkable(target)
        if not blocked_move:
            self.position = target
        self.steps_taken += 1

        terminated = self.position == self.task.goal
        truncated = not terminated and self.steps_taken >= self.task.max_steps
        self.done = terminated or truncated
        reward = self.task.step_penalty
        if blocked_move:
            reward += self.task.blocked_penalty
        if terminated:
            reward += self.task.success_reward
        return StepResult(self.observe(), reward, terminated, truncated, blocked_move)
