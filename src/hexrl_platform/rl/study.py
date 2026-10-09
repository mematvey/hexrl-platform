import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from hexrl_platform.rl.navigation import NavigationTask
from hexrl_platform.rl.observation import ObservationEncoder
from hexrl_platform.rl.tasks import Partition, TaskSplit
from hexrl_platform.rl.training import DQNConfig
from hexrl_platform.rl.values import Probability

RANDOM_LABEL = "random"
SHORTEST_PATH_LABEL = "shortest-path"
COMPARISON_LABEL = "comparison"
RESERVED_LABELS = {RANDOM_LABEL, SHORTEST_PATH_LABEL, COMPARISON_LABEL}


@dataclass(frozen=True, slots=True)
class StudyConfig:
    name: str
    experiment: str
    seeds: tuple[int, ...]
    variants: tuple[tuple[str, DQNConfig], ...]

    def __post_init__(self) -> None:
        if not self.name or not self.experiment:
            raise ValueError("study name and experiment must not be empty")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        labels = [label for label, _ in self.variants]
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("variants must be non-empty and have unique labels")
        if RESERVED_LABELS & set(labels):
            raise ValueError(f"variant labels must not use {sorted(RESERVED_LABELS)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "experiment": self.experiment,
            "seeds": list(self.seeds),
            "variants": {label: config.to_params() for label, config in self.variants},
        }


def dqn_config(values: dict[str, Any]) -> DQNConfig:
    unknown = set(values) - {field.name for field in fields(DQNConfig)}
    if unknown:
        raise ValueError(f"Unknown DQN parameters: {', '.join(sorted(unknown))}")
    converted = dict(values)
    for name in ("exploration_initial_eps", "exploration_final_eps"):
        if name in converted:
            converted[name] = Probability(converted[name])
    if "net_arch" in converted:
        converted["net_arch"] = tuple(converted["net_arch"])
    return DQNConfig(**converted)


def load_study_config(path: Path) -> StudyConfig:
    """Reads [study], shared [dqn] parameters and per-variant overrides from TOML."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    try:
        study = data["study"]
        variants = data["variants"]
        return StudyConfig(
            name=study["name"],
            experiment=study["experiment"],
            seeds=tuple(study["seeds"]),
            variants=tuple(
                (label, dqn_config({**data.get("dqn", {}), **overrides}))
                for label, overrides in variants.items()
            ),
        )
    except KeyError as error:
        raise ValueError(f"Study config {path} is missing {error}") from error


@dataclass(frozen=True, slots=True)
class StudyContext:
    """Everything shared by the runs of one study, computed once."""

    study: StudyConfig
    replay_manifest: dict[str, Any]
    split: TaskSplit
    split_manifest: dict[str, Any]
    split_path: Path
    encoder: ObservationEncoder
    tasks: dict[Partition, list[NavigationTask]]
    output_dir: Path
    code_tags: dict[str, str]
