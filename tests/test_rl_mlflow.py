import json
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from hexrl_platform.rl.eda import log_eda
from hexrl_platform.rl.replay import ReplayGenerationConfig, generate_replays
from hexrl_platform.rl.values import Probability


def test_eda_run_keeps_dataset_metrics_and_visualizations_together(tmp_path, monkeypatch):
    path = tmp_path / "replays.jsonl"
    config = ReplayGenerationConfig(
        episodes=8,
        seed=31,
        max_steps=15,
        random_fraction=Probability(0.25),
        exploration_probability=Probability(0.4),
    )
    manifest = generate_replays(path, config)
    monkeypatch.chdir(tmp_path)
    tracking_uri = "sqlite:///tracking.db"
    previous_uri = mlflow.get_tracking_uri()
    try:
        run_id = log_eda(path, tracking_uri, "replay-eda-test")
        client = MlflowClient(tracking_uri=tracking_uri)
        run = client.get_run(run_id)
        assert run.info.status == "FINISHED"
        assert run.data.params["seed"] == str(manifest["seed"])
        assert run.data.params["max_steps"] == str(manifest["max_steps"])
        assert run.data.params["map_radius"] == str(manifest["map"]["radius"])
        assert float(run.data.params["random_fraction"]) == pytest.approx(0.25)
        assert float(run.data.params["exploration_probability"]) == pytest.approx(0.4)
        assert run.data.tags["dataset.sha256"] == manifest["sha256"]
        assert run.data.metrics["episodes"] == manifest["episodes"]
        assert run.data.metrics["transitions"] == manifest["transitions"]
        assert run.data.metrics["success_rate"] == pytest.approx(
            manifest["successful_episodes"] / manifest["episodes"]
        )

        assert run.inputs is not None
        (dataset_input,) = run.inputs.dataset_inputs
        assert dataset_input.tags[0].value == "analysis"
        assert dataset_input.dataset.name == "hex-navigation-replays"
        assert dataset_input.dataset.digest == manifest["sha256"][:32]

        dataset_artifacts = {item.path for item in client.list_artifacts(run_id, "dataset")}
        assert dataset_artifacts >= {"dataset/replays.jsonl", "dataset/replays.manifest.json"}
        replay_artifact = client.download_artifacts(run_id, "dataset/replays.jsonl", str(tmp_path))
        assert Path(replay_artifact).read_bytes() == path.read_bytes()
        summary_path = client.download_artifacts(run_id, "eda/summary.json", str(tmp_path))
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        assert summary["overall"]["success_rate"] == pytest.approx(run.data.metrics["success_rate"])
        assert sum(group["episodes"] for group in summary["by_policy"].values()) == 8
        assert {item.path for item in client.list_artifacts(run_id, "eda")} >= {
            "eda/summary.json",
            "eda/episode_lengths.png",
            "eda/success_by_distance.png",
            "eda/visited_cells.png",
        }
    finally:
        mlflow.set_tracking_uri(previous_uri)


def test_eda_does_not_log_a_run_for_a_corrupt_replay(tmp_path):
    path = tmp_path / "replays.jsonl"
    generate_replays(path, ReplayGenerationConfig(episodes=2, seed=31))
    path.write_bytes(path.read_bytes() + b"\n")
    tracking_uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"

    with pytest.raises(ValueError, match="digest"):
        log_eda(path, tracking_uri, "invalid-replay-test")

    assert not (tmp_path / "tracking.db").exists()
