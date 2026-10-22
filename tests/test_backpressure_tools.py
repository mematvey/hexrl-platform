import json

import pandas as pd
import pytest

from hexrl_platform.rl.hex_grid import HexCoord
from hexrl_platform.rl.observation import ObservationEncoder
from hexrl_platform.rl.replay import default_map
from scripts.backpressure.fault_worker import FaultModel
from scripts.backpressure.make_dataset import make_dataset


def test_dataset_is_reproducible_and_matches_the_model_map(tmp_path):
    encoder = ObservationEncoder(default_map(), 20)
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    one = make_dataset(encoder, first, size=20, seed=7, model_uri="models:/test/1")
    two = make_dataset(encoder, second, size=20, seed=7, model_uri="models:/test/1")
    assert first.read_bytes() == second.read_bytes()
    assert one["sha256"] == two["sha256"]
    records = [json.loads(line) for line in first.read_text().splitlines()]
    assert len(records) == 20
    assert all(1 <= row["steps_remaining"] <= 20 for row in records)
    for row in records:
        start, goal = HexCoord(row["q"], row["r"]), HexCoord(row["goal_q"], row["goal_r"])
        assert start != goal
        assert encoder.hex_map.is_walkable(start)
        assert encoder.hex_map.is_walkable(goal)


def test_failure_dataset_contains_schema_valid_domain_errors_only(tmp_path):
    encoder = ObservationEncoder(default_map(), 20)
    normal, failure = tmp_path / "normal.jsonl", tmp_path / "failure.jsonl"
    make_dataset(encoder, normal, size=20, seed=7, model_uri="models:/test/1")
    manifest = make_dataset(
        encoder, failure, size=20, seed=7, invalid_fraction=0.2, model_uri="models:/test/1"
    )
    original = normal.read_text().splitlines()
    changed = failure.read_text().splitlines()
    assert sum(a != b for a, b in zip(original, changed, strict=True)) == 4
    assert manifest["invalid_rows"] == 4


def test_fault_injection_expires_and_preserves_model_predictions():
    from hexrl_platform.inference.predict import PredictModel

    class Model(PredictModel):
        def predict(self, data):
            return pd.DataFrame({"action": [4]})

    model = FaultModel(Model(), 1)
    with pytest.raises(RuntimeError, match="injected_transient"):
        model.predict(pd.DataFrame())
    assert model.predict(pd.DataFrame())["action"].iloc[0] == 4


@pytest.mark.parametrize("arguments", [[], ["--unbounded"], ["--queue-prefix", "isolated"]])
def test_unconfirmed_baseline_cannot_accidentally_use_the_default_queue(monkeypatch, arguments):
    from hexrl_platform.task_pipeline.batch_source import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "hexrl-batch-source",
            "--file",
            "input.jsonl",
            "--batch-id",
            "test",
            "--no-backpressure",
            *arguments,
        ],
    )
    with pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2


def test_plotting_reconciles_ids_and_uses_observer_clock(tmp_path):
    from scripts.backpressure.collect_metrics import append_csv
    from scripts.backpressure.plot_results import plot_results

    directory = tmp_path / "A"
    directory.mkdir()
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": {"rows": 1},
                "completed": 1,
                "failed": 0,
                "reconciled_unique_tasks": 1,
                "workers": 1,
                "duration_s": 1.2,
            }
        )
    )
    append_csv(
        directory / "metrics.csv",
        {
            "elapsed_s": 1,
            "queue_ready": 0,
            "queue_unacked": 0,
            "broker_memory_bytes": 1000000,
            "worker_rss_bytes": 1000000,
            "completed": 1,
            "redelivered_total": 0,
        },
    )
    append_csv(
        directory / "source.csv",
        {
            "t_s": 0.1,
            "published_ok": 1,
            "rss_mb": 50,
            "rejected": 0,
        },
    )
    append_csv(
        directory / "tasks.csv",
        {
            "task_id": "one",
            "status": "COMPLETED",
            "end_to_end_ms": -5000,
            "submitted_at": "2026-01-01T00:00:00Z",
            "inference_ms": 2,
        },
    )
    append_csv(
        directory / "observations.csv",
        {
            "task_id": "one",
            "observed_at": "2026-01-01T00:00:01Z",
            "elapsed_s": 1,
            "poll_gap_s": 1,
        },
    )
    summary = plot_results(tmp_path)["A"]
    assert summary["mean_observed_latency_s"] == 1
    assert summary["negative_raw_latency_rows"] == 1
    assert summary["terminal_rate_s"] == 1
    assert (tmp_path / "overview.png").is_file()
    append_csv(
        directory / "observations.csv",
        {
            "task_id": "unexpected",
            "observed_at": "2026-01-01T00:00:01Z",
            "elapsed_s": 1,
            "poll_gap_s": 1,
        },
    )
    with pytest.raises(ValueError, match="IDs do not match"):
        plot_results(tmp_path)
