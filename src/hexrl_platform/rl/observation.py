from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium import spaces

from hexrl_platform.rl.hex_grid import DIRECTIONS, HexCoord, HexMap
from hexrl_platform.rl.navigation import NavigationObservation

ENCODER_VERSION = "nav-obs-v1"


@dataclass(frozen=True, slots=True)
class ObservationEncoder:
    hex_map: HexMap
    max_steps: int

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")

    @property
    def version(self) -> str:
        return ENCODER_VERSION

    @property
    def observation_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "position": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                "goal": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                "goal_delta": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                "steps_left": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "walkable_neighbors": spaces.Box(
                    0.0, 1.0, shape=(len(DIRECTIONS),), dtype=np.float32
                ),
            }
        )

    def encode(self, observation: NavigationObservation) -> dict[str, np.ndarray]:
        radius = self.hex_map.radius
        position = observation.position
        goal = observation.goal
        return {
            "position": np.array([position.q / radius, position.r / radius], dtype=np.float32),
            "goal": np.array([goal.q / radius, goal.r / radius], dtype=np.float32),
            "goal_delta": np.array(
                [(goal.q - position.q) / (2 * radius), (goal.r - position.r) / (2 * radius)],
                dtype=np.float32,
            ),
            "steps_left": np.array(
                [observation.steps_remaining / self.max_steps], dtype=np.float32
            ),
            "walkable_neighbors": np.array(
                [
                    self.hex_map.is_walkable(
                        HexCoord(position.q + direction.q, position.r + direction.r)
                    )
                    for direction in DIRECTIONS
                ],
                dtype=np.float32,
            ),
        }

    def to_config(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "max_steps": self.max_steps,
            "map": {
                "radius": self.hex_map.radius,
                "blocked": [{"q": cell.q, "r": cell.r} for cell in sorted(self.hex_map.blocked)],
            },
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ObservationEncoder":
        if config["version"] != ENCODER_VERSION:
            raise ValueError(f"Unsupported observation encoder version: {config['version']}")
        hex_map = HexMap(
            config["map"]["radius"],
            frozenset(HexCoord(cell["q"], cell["r"]) for cell in config["map"]["blocked"]),
        )
        return cls(hex_map, config["max_steps"])
