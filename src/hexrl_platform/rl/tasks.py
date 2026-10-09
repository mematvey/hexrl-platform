import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from hexrl_platform.rl.hex_grid import HexCoord, HexMap
from hexrl_platform.rl.navigation import NavigationTask

SPLIT_VERSION = 1
SPLIT_RULE = "sha256 of the unordered start-goal pair mod 100: <70 train, <85 validation, else test"

Partition = Literal["train", "validation", "test"]
PARTITIONS: tuple[Partition, ...] = ("train", "validation", "test")
TaskPair = tuple[HexCoord, HexCoord]


def assign_partition(start: HexCoord, goal: HexCoord) -> Partition:
    first, second = sorted((start, goal))
    # Built-in hash() is salted per process, so the bucket comes from a stable digest
    key = f"{first.q},{first.r}|{second.q},{second.r}".encode()
    bucket = int(hashlib.sha256(key).hexdigest(), 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


@dataclass(frozen=True, slots=True)
class TaskSplit:
    train: tuple[TaskPair, ...]
    validation: tuple[TaskPair, ...]
    test: tuple[TaskPair, ...]
    parent_sha256: str

    def partition(self, name: Partition) -> tuple[TaskPair, ...]:
        return getattr(self, name)


def split_tasks(records: Iterable[dict[str, Any]], parent_sha256: str) -> TaskSplit:
    """Collects start-goal pairs from the first transition of each replay episode."""
    pairs = sorted(
        {
            (
                HexCoord(record["state"]["q"], record["state"]["r"]),
                HexCoord(record["state"]["goal_q"], record["state"]["goal_r"]),
            )
            for record in records
            if record["step_id"] == 0
        }
    )
    if not pairs:
        raise ValueError("Replay records contain no episodes")
    grouped: dict[Partition, list[TaskPair]] = {name: [] for name in PARTITIONS}
    for start, goal in pairs:
        grouped[assign_partition(start, goal)].append((start, goal))
    empty = [name for name in PARTITIONS if not grouped[name]]
    if empty:
        raise ValueError(f"Not enough distinct tasks, empty partitions: {', '.join(empty)}")
    return TaskSplit(
        train=tuple(grouped["train"]),
        validation=tuple(grouped["validation"]),
        test=tuple(grouped["test"]),
        parent_sha256=parent_sha256,
    )


def pairs_to_records(pairs: Sequence[TaskPair]) -> list[dict[str, int]]:
    return [
        {"start_q": start.q, "start_r": start.r, "goal_q": goal.q, "goal_r": goal.r}
        for start, goal in pairs
    ]


def _pairs_from_json(items: Sequence[dict[str, int]]) -> tuple[TaskPair, ...]:
    return tuple(
        (HexCoord(item["start_q"], item["start_r"]), HexCoord(item["goal_q"], item["goal_r"]))
        for item in items
    )


def partition_digest(pairs: Sequence[TaskPair]) -> str:
    encoded = json.dumps(pairs_to_records(pairs), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def save_task_split(split: TaskSplit, path: Path) -> dict[str, Any]:
    payload = {
        "split_version": SPLIT_VERSION,
        "parent_sha256": split.parent_sha256,
        "partitions": {name: pairs_to_records(split.partition(name)) for name in PARTITIONS},
    }
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    manifest = {
        "split_version": SPLIT_VERSION,
        "rule": SPLIT_RULE,
        "dataset_file": path.name,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "parent_sha256": split.parent_sha256,
        "counts": {name: len(split.partition(name)) for name in PARTITIONS},
        "partition_sha256": {name: partition_digest(split.partition(name)) for name in PARTITIONS},
    }
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_task_split(path: Path) -> tuple[TaskSplit, dict[str, Any]]:
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != manifest["sha256"]:
        raise ValueError("Task split digest does not match the manifest")
    payload = json.loads(encoded)
    if payload["split_version"] != SPLIT_VERSION:
        raise ValueError("Unsupported task split version")
    partitions = payload["partitions"]
    split = TaskSplit(
        train=_pairs_from_json(partitions["train"]),
        validation=_pairs_from_json(partitions["validation"]),
        test=_pairs_from_json(partitions["test"]),
        parent_sha256=payload["parent_sha256"],
    )
    return split, manifest


def navigation_tasks(
    pairs: Sequence[TaskPair], hex_map: HexMap, max_steps: int
) -> list[NavigationTask]:
    return [NavigationTask(hex_map, start, goal, max_steps) for start, goal in pairs]
