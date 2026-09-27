import json

import pytest

from hexrl_platform.rl.eda import eda_metrics, load_replays, summarize_replays
from hexrl_platform.rl.hex_grid import HexCoord, HexMap
from hexrl_platform.rl.navigation import NavigationEnv, NavigationTask
from hexrl_platform.rl.replay import action_probabilities, generate_replays


def test_hex_distance_and_obstacle_aware_path() -> None:
    hex_map = HexMap(2, frozenset({HexCoord(1, 0)}))

    assert HexCoord(0, 0).distance_to(HexCoord(2, -1)) == 2
    assert len(hex_map.cells) == 18
    assert hex_map.shortest_distance(HexCoord(0, 0), HexCoord(2, 0)) == 3
    assert HexCoord(1, 0) not in hex_map.neighbors(HexCoord(0, 0))


def test_invalid_map_and_task_are_rejected() -> None:
    with pytest.raises(ValueError, match="radius"):
        HexMap(0)
    with pytest.raises(ValueError, match="inside"):
        HexMap(1, frozenset({HexCoord(2, 0)}))
    with pytest.raises(ValueError, match="two walkable"):
        HexMap(1, frozenset(HexMap(1).cells) - {HexCoord(0, 0)})

    hex_map = HexMap(1)
    with pytest.raises(ValueError, match="max_steps"):
        NavigationTask(hex_map, HexCoord(0, 0), HexCoord(1, 0), 0)
    with pytest.raises(ValueError, match="differ"):
        NavigationTask(hex_map, HexCoord(0, 0), HexCoord(0, 0), 2)


def test_navigation_rewards_and_terminal_state() -> None:
    hex_map = HexMap(2, frozenset({HexCoord(1, 0)}))
    environment = NavigationEnv(NavigationTask(hex_map, HexCoord(0, 0), HexCoord(0, -1), 2))

    blocked = environment.step(0)
    assert blocked.blocked_move
    assert blocked.observation.position == HexCoord(0, 0)
    assert blocked.reward == pytest.approx(-0.11)
    assert not blocked.terminated

    success = environment.step(2)
    assert success.terminated
    assert not success.truncated
    assert success.reward == pytest.approx(0.99)
    with pytest.raises(RuntimeError, match="ended"):
        environment.step(2)


def test_navigation_truncates_and_rejects_invalid_actions() -> None:
    environment = NavigationEnv(NavigationTask(HexMap(1), HexCoord(0, 0), HexCoord(1, 0), 1))

    with pytest.raises(ValueError, match="direction index"):
        environment.step(6)
    result = environment.step(3)
    assert result.truncated
    assert not result.terminated


def test_replay_generation_is_reproducible_and_verifiable(tmp_path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first_manifest = generate_replays(first, episodes=24, seed=7, max_steps=30)
    second_manifest = generate_replays(second, episodes=24, seed=7, max_steps=30)

    assert first.read_bytes() == second.read_bytes()
    assert first_manifest["sha256"] == second_manifest["sha256"]
    assert first_manifest["episodes"] == 24
    assert sum(first_manifest["behavior_policy"]["episode_counts"].values()) == 24

    records, manifest = load_replays(first)
    summaries = summarize_replays(records)
    metrics = eda_metrics(summaries)
    assert len(records) == manifest["transitions"]
    assert len(summaries) == manifest["episodes"]
    assert metrics["success_rate"] == pytest.approx(manifest["successful_episodes"] / 24)
    assert 0 < metrics["mean_episode_length"] <= 30
    assert all(0 < record["action_probability"] <= 1 for record in records)
    assert all(record["optimal_steps"] >= 2 for record in records)

    first.write_bytes(first.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="digest"):
        load_replays(first)


def test_behavior_policy_probabilities() -> None:
    hex_map = HexMap(2)
    goal = HexCoord(2, 0)
    distribution = action_probabilities(
        hex_map,
        HexCoord(0, 0),
        goal,
        "goal_directed",
        0.2,
        hex_map.distances_from(goal),
    )
    assert sum(distribution) == pytest.approx(1)
    assert distribution[0] > distribution[3]
    assert (
        action_probabilities(
            hex_map, HexCoord(0, 0), goal, "random", 0.2, hex_map.distances_from(goal)
        )
        == (1 / 6,) * 6
    )
    with pytest.raises(ValueError, match="Unknown"):
        action_probabilities(
            hex_map, HexCoord(0, 0), goal, "unknown", 0.2, hex_map.distances_from(goal)
        )
    with pytest.raises(ValueError, match="between 0 and 1"):
        action_probabilities(
            hex_map, HexCoord(0, 0), goal, "goal_directed", 1.2, hex_map.distances_from(goal)
        )


def test_replay_generator_validates_configuration(tmp_path) -> None:
    path = tmp_path / "invalid.jsonl"
    with pytest.raises(ValueError, match="positive"):
        generate_replays(path, episodes=0)
    with pytest.raises(ValueError, match="random_fraction"):
        generate_replays(path, random_fraction=-0.1)
    with pytest.raises(ValueError, match="exploration_probability"):
        generate_replays(path, exploration_probability=1.1)
    assert not path.exists()


def test_manifest_describes_dataset_schema(tmp_path) -> None:
    path = tmp_path / "replays.jsonl"
    generate_replays(path, episodes=2, seed=1, max_steps=5)

    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["generator"] == "hexrl_platform.rl.replay:generate_replays"
    assert manifest["map"]["radius"] == 4

    manifest_path = path.with_suffix(".manifest.json")
    manifest["schema_version"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="schema version"):
        load_replays(path)


def test_eda_rejects_incomplete_episode() -> None:
    record = {
        "episode_id": 0,
        "step_id": 0,
        "behavior_policy": "random",
        "reward": -0.01,
        "blocked_move": False,
        "optimal_steps": 2,
        "terminated": False,
        "truncated": False,
    }
    with pytest.raises(ValueError, match="incomplete"):
        summarize_replays([record])
    with pytest.raises(ValueError, match="empty"):
        eda_metrics([])
