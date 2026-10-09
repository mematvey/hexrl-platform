import json

import pytest

from hexrl_platform.rl.hex_grid import HexCoord
from hexrl_platform.rl.navigation import NavigationObservation, NavigationTask
from hexrl_platform.rl.observation import ObservationEncoder
from hexrl_platform.rl.replay import default_map
from hexrl_platform.rl.training import (
    POLICY_FILE,
    DQNConfig,
    SB3Policy,
    load_policy,
    train_dqn,
)
from hexrl_platform.rl.values import Probability

SMOKE_CONFIG = DQNConfig(
    total_timesteps=300,
    learning_starts=50,
    buffer_size=1_000,
    batch_size=16,
    target_update_interval=100,
    eval_freq=100,
    net_arch=(16,),
)


def make_tasks(max_steps: int = 10) -> list[NavigationTask]:
    hex_map = default_map()
    pairs = [(HexCoord(-2, 0), HexCoord(0, 0)), (HexCoord(0, -2), HexCoord(0, 0))]
    return [NavigationTask(hex_map, start, goal, max_steps) for start, goal in pairs]


def test_dqn_smoke_training_evaluates_periodically_and_saves_policies(tmp_path):
    encoder = ObservationEncoder(default_map(), max_steps=10)
    scores = iter([0.1, 0.5, 0.3, 0.2])
    evaluated_policies = []
    reported_steps = []

    def evaluate(policy):
        evaluated_policies.append(policy)
        return {"success_rate": next(scores), "mean_return": 0.0}

    result = train_dqn(
        SMOKE_CONFIG,
        encoder,
        make_tasks(),
        evaluate,
        seed=1,
        output_dir=tmp_path,
        on_evaluation=lambda step, metrics: reported_steps.append(step),
    )

    assert reported_steps == [100, 200, 300]
    assert [step for step, _ in result.curve] == reported_steps
    assert result.best_step == 200
    assert result.best_metrics["success_rate"] == 0.5
    assert result.device == "cpu"
    assert all(isinstance(policy, SB3Policy) for policy in evaluated_policies)

    for directory in (result.best_policy_dir, result.final_policy_dir):
        policy = load_policy(directory)
        observation = NavigationObservation(HexCoord(-2, 0), HexCoord(0, 0), 10)
        assert 0 <= policy.act(observation) < 6
        assert policy.encoder == encoder


def test_loaded_policy_acts_like_the_trained_one(tmp_path):
    encoder = ObservationEncoder(default_map(), max_steps=10)
    policies = []

    def evaluate(policy):
        policies.append(policy)
        return {"success_rate": 0.0, "mean_return": 0.0}

    result = train_dqn(SMOKE_CONFIG, encoder, make_tasks(), evaluate, seed=2, output_dir=tmp_path)
    trained = policies[-1]
    loaded = load_policy(result.final_policy_dir)
    observations = [
        NavigationObservation(cell, HexCoord(0, 0), 7)
        for cell in encoder.hex_map.cells
        if cell != HexCoord(0, 0)
    ]
    assert [loaded.act(item) for item in observations] == [
        trained.act(item) for item in observations
    ]


def test_load_policy_rejects_changed_action_order(tmp_path):
    encoder = ObservationEncoder(default_map(), max_steps=10)
    result = train_dqn(
        SMOKE_CONFIG,
        encoder,
        make_tasks(),
        lambda policy: {"success_rate": 0.0, "mean_return": 0.0},
        seed=3,
        output_dir=tmp_path,
    )
    description_path = result.final_policy_dir / POLICY_FILE
    description = json.loads(description_path.read_text(encoding="utf-8"))
    description["actions"].reverse()
    description_path.write_text(json.dumps(description), encoding="utf-8")
    with pytest.raises(ValueError, match="action order"):
        load_policy(result.final_policy_dir)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"total_timesteps": 0}, "total_timesteps"),
        ({"learning_starts": -1}, "learning_starts"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"gamma": 1.5}, "gamma"),
        ({"exploration_fraction": 0.0}, "exploration_fraction"),
        ({"net_arch": ()}, "net_arch"),
        ({"device": "gpu"}, "device"),
    ],
)
def test_dqn_config_rejects_invalid_values(overrides, message):
    with pytest.raises(ValueError, match=message):
        DQNConfig(**overrides)


def test_dqn_config_params_are_flat_and_loggable():
    params = DQNConfig(exploration_final_eps=Probability(0.1)).to_params()
    assert params["exploration_final_eps"] == 0.1
    assert params["net_arch"] == "64,64"
    assert all(isinstance(value, int | float | str) for value in params.values())
