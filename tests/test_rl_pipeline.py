from collections import defaultdict

import pytest

from hexrl_platform.rl.eda import EpisodeSummary, eda_metrics, load_replays, summarize_replays
from hexrl_platform.rl.hex_grid import DIRECTIONS, HexCoord, HexMap
from hexrl_platform.rl.navigation import NavigationEnv, NavigationTask
from hexrl_platform.rl.replay import action_probabilities, generate_replays


@pytest.mark.parametrize("random_fraction", [0.0, 0.5, 1.0])
def test_generated_replays_can_be_played_back(tmp_path, random_fraction):
    path = tmp_path / "replays.jsonl"
    generate_replays(path, episodes=12, seed=19, max_steps=18, random_fraction=random_fraction)
    records, manifest = load_replays(path)
    hex_map = HexMap(
        manifest["map"]["radius"],
        frozenset(HexCoord(**cell) for cell in manifest["map"]["blocked"]),
    )
    episodes = defaultdict(list)
    for record in records:
        episodes[record["episode_id"]].append(record)

    assert sorted(episodes) == list(range(manifest["episodes"]))
    assert len(records) == manifest["transitions"]
    assert {
        policy: sum(group[0]["behavior_policy"] == policy for group in episodes.values())
        for policy in manifest["behavior_policy"]["episode_counts"]
    } == manifest["behavior_policy"]["episode_counts"]
    assert (
        sum(group[-1]["terminated"] for group in episodes.values())
        == manifest["successful_episodes"]
    )

    for transitions in episodes.values():
        first_state = transitions[0]["state"]
        environment = NavigationEnv(
            NavigationTask(
                hex_map,
                HexCoord(first_state["q"], first_state["r"]),
                HexCoord(first_state["goal_q"], first_state["goal_r"]),
                manifest["max_steps"],
                **manifest["rewards"],
            )
        )
        assert transitions[0]["optimal_steps"] == hex_map.shortest_distance(
            environment.task.start, environment.task.goal
        )
        for step_id, record in enumerate(transitions):
            observation = environment.observe()
            assert record["step_id"] == step_id
            assert record["state"] == {
                "q": observation.position.q,
                "r": observation.position.r,
                "goal_q": observation.goal.q,
                "goal_r": observation.goal.r,
                "steps_remaining": observation.steps_remaining,
            }
            assert 0 <= record["action"] < len(DIRECTIONS)
            probabilities = action_probabilities(
                hex_map,
                observation.position,
                observation.goal,
                record["behavior_policy"],
                manifest["behavior_policy"]["exploration_probability"],
                hex_map.distances_from(observation.goal),
            )
            assert record["action_probability"] == pytest.approx(probabilities[record["action"]])

            result = environment.step(record["action"])
            assert record["next_state"] == {
                "q": result.observation.position.q,
                "r": result.observation.position.r,
                "goal_q": result.observation.goal.q,
                "goal_r": result.observation.goal.r,
                "steps_remaining": result.observation.steps_remaining,
            }
            assert record["reward"] == pytest.approx(result.reward)
            assert record["blocked_move"] == result.blocked_move
            assert record["terminated"] == result.terminated
            assert record["truncated"] == result.truncated
            assert environment.done == (step_id == len(transitions) - 1)


@pytest.mark.parametrize(
    ("step_ids", "terminal_flags", "policies", "message"),
    [
        ([0, 2], [(False, False), (True, False)], ["random"] * 2, "steps"),
        ([0, 1], [(True, False), (True, False)], ["random"] * 2, "after completion"),
        ([0, 1], [(False, False), (True, False)], ["random", "goal_directed"], "policy"),
        ([0, 1], [(False, False), (False, False)], ["random"] * 2, "incomplete"),
    ],
)
def test_eda_rejects_unplayable_episode(step_ids, terminal_flags, policies, message):
    records = [
        {
            "episode_id": 0,
            "step_id": step_id,
            "behavior_policy": policy,
            "reward": -0.01,
            "blocked_move": False,
            "optimal_steps": 2,
            "terminated": terminated,
            "truncated": truncated,
        }
        for step_id, (terminated, truncated), policy in zip(
            step_ids, terminal_flags, policies, strict=True
        )
    ]
    with pytest.raises(ValueError, match=message):
        summarize_replays(records)


def test_eda_metrics_use_episodes_for_outcomes_and_steps_for_blocked_moves():
    summaries = [
        EpisodeSummary(0, "random", 2, 1.5, True, 1, 2),
        EpisodeSummary(1, "goal_directed", 4, -0.5, False, 2, 3),
    ]

    assert eda_metrics(summaries) == {
        "episodes": 2.0,
        "transitions": 6.0,
        "success_rate": 0.5,
        "mean_return": 0.5,
        "mean_episode_length": 3.0,
        "blocked_move_rate": 0.5,
    }
