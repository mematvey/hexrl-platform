import random
from typing import Protocol

from hexrl_platform.rl.hex_grid import DIRECTIONS, HexCoord, HexMap
from hexrl_platform.rl.navigation import NavigationObservation
from hexrl_platform.rl.replay import action_probabilities
from hexrl_platform.rl.values import Probability


class Policy(Protocol):
    def act(self, observation: NavigationObservation) -> int: ...


class RandomPolicy(Policy):
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def act(self, observation: NavigationObservation) -> int:
        return self._rng.randrange(len(DIRECTIONS))


class ShortestPathPolicy(Policy):
    """Upper-bound reference: knows the map and follows a BFS shortest path."""

    def __init__(self, hex_map: HexMap) -> None:
        self.hex_map = hex_map
        self._distances: dict[HexCoord, dict[HexCoord, int]] = {}

    def act(self, observation: NavigationObservation) -> int:
        goal = observation.goal
        if goal not in self._distances:
            self._distances[goal] = self.hex_map.distances_from(goal)
        probabilities = action_probabilities(
            self.hex_map,
            observation.position,
            goal,
            "goal_directed",
            Probability(0.0),
            self._distances[goal],
        )
        return max(range(len(probabilities)), key=probabilities.__getitem__)
