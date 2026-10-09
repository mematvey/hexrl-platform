import json
import os
import subprocess
import sys

import pytest

from hexrl_platform.rl.eda import load_replays
from hexrl_platform.rl.hex_grid import HexCoord
from hexrl_platform.rl.replay import ReplayGenerationConfig, default_map, generate_replays
from hexrl_platform.rl.tasks import (
    PARTITIONS,
    assign_partition,
    load_task_split,
    navigation_tasks,
    save_task_split,
    split_tasks,
)


def all_pairs() -> list[tuple[HexCoord, HexCoord]]:
    hex_map = default_map()
    return [
        (start, goal)
        for start in hex_map.cells
        for goal, distance in hex_map.distances_from(start).items()
        if distance >= 2
    ]


def test_partition_is_valid_and_symmetric():
    for start, goal in all_pairs():
        partition = assign_partition(start, goal)
        assert partition in PARTITIONS
        assert assign_partition(goal, start) == partition


def test_partition_proportions_are_close_to_70_15_15():
    pairs = all_pairs()
    counts = {name: 0 for name in PARTITIONS}
    for start, goal in pairs:
        counts[assign_partition(start, goal)] += 1
    shares = {name: count / len(pairs) for name, count in counts.items()}
    assert shares["train"] == pytest.approx(0.70, abs=0.05)
    assert shares["validation"] == pytest.approx(0.15, abs=0.05)
    assert shares["test"] == pytest.approx(0.15, abs=0.05)


def test_partition_does_not_change_between_python_processes():
    script = (
        "from hexrl_platform.rl.hex_grid import HexCoord;"
        "from hexrl_platform.rl.tasks import assign_partition;"
        "print([assign_partition(HexCoord(q, -q), HexCoord(-q, 2)) for q in range(-2, 3)])"
    )
    outputs = set()
    for hash_seed in ("1", "2"):
        environment = {**os.environ, "PYTHONHASHSEED": hash_seed}
        completed = subprocess.run(
            [sys.executable, "-c", script],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.add(completed.stdout)
    assert len(outputs) == 1


@pytest.fixture
def replay_records(tmp_path):
    path = tmp_path / "replays.jsonl"
    generate_replays(path, ReplayGenerationConfig(episodes=400, seed=11, max_steps=20))
    return load_replays(path)


def test_split_covers_replay_tasks_without_leakage(replay_records):
    records, manifest = replay_records
    split = split_tasks(records, manifest["sha256"])
    replay_pairs = {
        (
            HexCoord(record["state"]["q"], record["state"]["r"]),
            HexCoord(record["state"]["goal_q"], record["state"]["goal_r"]),
        )
        for record in records
        if record["step_id"] == 0
    }
    partitions = [set(split.partition(name)) for name in PARTITIONS]
    assert set().union(*partitions) == replay_pairs
    assert sum(len(part) for part in partitions) == len(replay_pairs)
    for name in PARTITIONS:
        for start, goal in split.partition(name):
            assert assign_partition(start, goal) == name
    assert split.parent_sha256 == manifest["sha256"]


def test_saved_split_is_reproducible_and_verified(replay_records, tmp_path):
    records, manifest = replay_records
    split = split_tasks(records, manifest["sha256"])
    first, second = tmp_path / "a" / "split.json", tmp_path / "b" / "split.json"
    first_manifest = save_task_split(split, first)
    save_task_split(split_tasks(records, manifest["sha256"]), second)

    assert first.read_bytes() == second.read_bytes()
    assert first_manifest["parent_sha256"] == manifest["sha256"]
    assert first_manifest["counts"]["train"] == len(split.train)

    loaded, loaded_manifest = load_task_split(first)
    assert loaded == split
    assert loaded_manifest == first_manifest

    payload = json.loads(first.read_bytes())
    payload["partitions"]["test"].append(payload["partitions"]["train"][0])
    first.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        load_task_split(first)


def test_split_rejects_too_few_tasks():
    record = {"step_id": 0, "state": {"q": 0, "r": 0, "goal_q": 2, "goal_r": 0}}
    with pytest.raises(ValueError, match="empty partitions"):
        split_tasks([record], "parent")
    with pytest.raises(ValueError, match="no episodes"):
        split_tasks([], "parent")


def test_navigation_tasks_keep_pairs_and_limits():
    pairs = [(HexCoord(0, 0), HexCoord(2, 0)), (HexCoord(-2, 0), HexCoord(0, 0))]
    tasks = navigation_tasks(pairs, default_map(), max_steps=17)
    assert [(task.start, task.goal) for task in tasks] == pairs
    assert all(task.max_steps == 17 for task in tasks)
