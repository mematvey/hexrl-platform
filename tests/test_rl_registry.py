import json
import math
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from hexrl_platform.rl.experiment import run_study
from hexrl_platform.rl.registry import (
    PACKAGED_MODEL_TAG,
    Candidate,
    RegistryConfig,
    best_per_variant,
    choose_champion,
    load_registry_config,
    register_study,
)
from hexrl_platform.rl.replay import ReplayGenerationConfig, generate_replays
from hexrl_platform.rl.study import StudyConfig
from hexrl_platform.rl.training import DQNConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def candidate(run_id, variant="a", success=0.5, path_ratio=1.5, seed=1):
    return Candidate(run_id, variant, seed, success, path_ratio)


def test_champion_has_the_best_success_rate():
    winner = choose_champion(
        [candidate("x", success=0.6), candidate("y", success=0.9, path_ratio=3.0)], 0.01
    )
    assert winner.run_id == "y"


def test_near_tie_is_broken_by_shorter_routes():
    runs = [
        candidate("x", success=0.900, path_ratio=1.4),
        candidate("y", success=0.905, path_ratio=1.9),
        candidate("z", success=0.800, path_ratio=1.0),
    ]
    assert choose_champion(runs, 0.01).run_id == "x"
    assert choose_champion(runs, 0.0).run_id == "y"


def test_missing_path_ratio_loses_a_tie_and_empty_input_fails():
    runs = [candidate("x", path_ratio=math.nan), candidate("y", path_ratio=2.5)]
    assert choose_champion(runs, 0.01).run_id == "y"
    with pytest.raises(ValueError, match="No candidates"):
        choose_champion([], 0.01)


def test_best_per_variant_picks_one_run_for_each_variant():
    runs = [
        candidate("a1", "a", 0.5),
        candidate("a2", "a", 0.7),
        candidate("b1", "b", 0.4),
    ]
    assert [c.run_id for c in best_per_variant(runs, 0.01)] == ["a2", "b1"]


def test_each_map_registers_under_its_own_model_name():
    names = {
        load_registry_config(CONFIGS / name).model_name
        for name in ("navigation-dqn.toml", "navigation-dqn-maze.toml")
    }
    assert names == {"hex-navigation-default", "hex-navigation-maze"}
    with pytest.raises(ValueError, match="tie_tolerance"):
        RegistryConfig(model_name="m", tie_tolerance=1.5)


TINY_DQN = DQNConfig(
    total_timesteps=200,
    learning_starts=50,
    buffer_size=500,
    batch_size=16,
    target_update_interval=50,
    eval_freq=100,
    net_arch=(16,),
)


@pytest.fixture(scope="module")
def registered(tmp_path_factory):
    # One study and registration shared by the end-to-end tests below
    tmp_path = tmp_path_factory.mktemp("registry")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.chdir(tmp_path)
        yield from _register(tmp_path)


def _register(tmp_path):
    replay_path = tmp_path / "replays.jsonl"
    generate_replays(replay_path, ReplayGenerationConfig(episodes=300, seed=5, max_steps=12))
    study = StudyConfig(
        name="registry-test",
        experiment="registry-test",
        replays=replay_path,
        seeds=(1, 2),
        variants=(("dqn-a", TINY_DQN), ("dqn-b", TINY_DQN)),
    )
    registry = RegistryConfig(model_name="nav-test")
    tracking_uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    previous_uri = mlflow.get_tracking_uri()
    try:
        run_study(study, tracking_uri=tracking_uri, output_dir=tmp_path / "out")
        first = register_study(study, registry, tracking_uri=tracking_uri)
        second = register_study(study, registry, tracking_uri=tracking_uri)
        yield MlflowClient(tracking_uri=tracking_uri), study, first, second, tmp_path
    finally:
        mlflow.set_tracking_uri(previous_uri)


def test_registration_creates_one_version_per_variant_linked_to_runs(registered):
    client, _, result, _, _ = registered
    versions = client.search_model_versions("name = 'nav-test'")
    assert len(versions) == 2
    assert {version.tags["variant"] for version in versions} == {"dqn-a", "dqn-b"}
    for version in versions:
        assert result.versions[version.run_id] == version.version
        assert PACKAGED_MODEL_TAG in client.get_run(version.run_id).data.tags


def test_alias_points_to_champion_and_loads_by_uri(registered):
    client, _, result, _, _ = registered
    by_alias = client.get_model_version_by_alias("nav-test", "champion")
    assert by_alias.version == result.champion_version
    assert by_alias.run_id == result.champion.run_id
    model = mlflow.pyfunc.load_model("models:/nav-test@champion")
    row = {"q": 2, "r": 0, "goal_q": 0, "goal_r": 0, "steps_remaining": 5}
    prediction = model.predict(pd.DataFrame([row]))
    assert isinstance(prediction, pd.DataFrame)
    assert 0 <= prediction["action"].iloc[0] < 6


def test_registration_is_idempotent(registered):
    client, _, first, second, _ = registered
    assert second.versions == first.versions
    assert second.champion_version == first.champion_version
    assert len(client.search_model_versions("name = 'nav-test'")) == 2


def test_champion_decision_is_logged_to_comparison_run(registered):
    client, study, result, _, tmp_path = registered
    experiment = client.get_experiment_by_name(study.experiment)
    (comparison,) = client.search_runs(
        [experiment.experiment_id], filter_string="tags.variant = 'comparison'"
    )
    path = client.download_artifacts(
        comparison.info.run_id, "registry/champion_decision.json", str(tmp_path)
    )
    decision = json.loads(Path(path).read_text(encoding="utf-8"))
    assert decision["champion"] == {
        "version": result.champion_version,
        "run_id": result.champion.run_id,
    }
    assert len(decision["candidates"]) == 2
