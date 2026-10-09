import json
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from hexrl_platform.rl.experiment import run_study
from hexrl_platform.rl.replay import ReplayGenerationConfig, generate_replays
from hexrl_platform.rl.study import StudyConfig
from hexrl_platform.rl.training import DQNConfig

TINY_DQN = DQNConfig(
    total_timesteps=200,
    learning_starts=50,
    buffer_size=500,
    batch_size=16,
    target_update_interval=50,
    eval_freq=100,
    net_arch=(16,),
)


@pytest.fixture
def study_runs(tmp_path, monkeypatch):
    # With a sqlite backend MLflow stores artifacts in ./mlruns of the working directory
    monkeypatch.chdir(tmp_path)
    replay_path = tmp_path / "replays.jsonl"
    replay_manifest = generate_replays(
        replay_path, ReplayGenerationConfig(episodes=300, seed=5, max_steps=12)
    )
    tracking_uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    previous_uri = mlflow.get_tracking_uri()
    study = StudyConfig(
        name="test-study",
        experiment="dqn-test",
        replays=replay_path,
        seeds=(1,),
        variants=(("dqn-a", TINY_DQN), ("dqn-b", TINY_DQN)),
    )
    try:
        run_ids = run_study(
            study,
            tracking_uri=tracking_uri,
            output_dir=tmp_path / "out",
        )
    finally:
        mlflow.set_tracking_uri(previous_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    runs = {run_id: client.get_run(run_id) for run_id in run_ids}
    return client, runs, replay_manifest, tmp_path


def by_variant(runs):
    grouped = {}
    for run in runs.values():
        grouped.setdefault(run.data.tags["variant"], []).append(run)
    return grouped


def test_study_logs_baselines_variants_and_comparison(study_runs):
    _, runs, replay_manifest, _ = study_runs
    grouped = by_variant(runs)
    assert set(grouped) == {"shortest-path", "random", "dqn-a", "dqn-b", "comparison"}
    assert all(run.info.status == "FINISHED" for run in runs.values())

    for variant in ("shortest-path", "random", "dqn-a", "dqn-b"):
        (run,) = grouped[variant]
        assert run.data.tags["study"] == "test-study"
        assert run.data.tags["dataset.parent_sha256"] == replay_manifest["sha256"]
        assert run.data.params["encoder_version"] == "nav-obs-v1"
        for metric in ("success_rate", "mean_return", "blocked_move_rate"):
            assert f"test/{metric}" in run.data.metrics
            assert f"val/{metric}" in run.data.metrics

    (oracle,) = grouped["shortest-path"]
    assert oracle.data.metrics["test/success_rate"] == 1.0
    assert oracle.data.metrics["test/path_ratio"] == 1.0
    assert grouped["random"][0].data.params["seed"] == "1"


def test_dqn_run_links_split_datasets_params_curve_and_policy(study_runs):
    client, runs, _, _ = study_runs
    (run,) = by_variant(runs)["dqn-a"]
    assert run.data.params["exploration_fraction"] == str(TINY_DQN.exploration_fraction)
    assert run.data.params["total_timesteps"] == "200"
    assert run.data.tags["split.sha256"]

    contexts = {
        dataset_input.tags[0].value: dataset_input.dataset
        for dataset_input in run.inputs.dataset_inputs
    }
    assert set(contexts) == {"training", "validation", "test"}
    assert contexts["training"].name == "navigation-tasks-train"
    assert contexts["test"].source_type == "code"
    assert (
        json.loads(contexts["test"].source)["tags"]["parent_sha256"]
        == run.data.tags["dataset.parent_sha256"]
    )

    history = client.get_metric_history(run.info.run_id, "val/success_rate")
    assert [point.step for point in history] == [100, 200]

    artifacts = {item.path for item in client.list_artifacts(run.info.run_id, "policy/best")}
    assert artifacts == {"policy/best/model.zip", "policy/best/policy.json"}
    diagnostics = {item.path for item in client.list_artifacts(run.info.run_id, "diagnostics")}
    assert "diagnostics/test_episode_lengths.png" in diagnostics


def test_baselines_do_not_claim_training_data(study_runs):
    _, runs, _, _ = study_runs
    (run,) = by_variant(runs)["random"]
    contexts = {item.tags[0].value for item in run.inputs.dataset_inputs}
    assert contexts == {"validation", "test"}


def test_comparison_run_summarizes_variants(study_runs, tmp_path):
    client, runs, _, _ = study_runs
    (run,) = by_variant(runs)["comparison"]
    artifacts = {item.path for item in client.list_artifacts(run.info.run_id, "comparison")}
    assert artifacts == {
        "comparison/summary.json",
        "comparison/learning_curves.png",
        "comparison/test_success_rate.png",
        "comparison/test_episode_lengths.png",
    }
    summary_path = client.download_artifacts(
        run.info.run_id, "comparison/summary.json", str(tmp_path)
    )
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    assert set(summary["dqn_beats_random"]) == {"dqn-a", "dqn-b"}
    assert summary["shortest-path"]["success_rate"]["mean"] == 1.0
