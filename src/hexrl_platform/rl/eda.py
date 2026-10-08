import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hexrl_platform.rl.hex_grid import HexCoord
from hexrl_platform.rl.replay import SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class EpisodeSummary:
    episode_id: int
    behavior_policy: str
    length: int
    total_reward: float
    success: bool
    blocked_moves: int
    optimal_steps: int


def load_replays(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported replay schema version")
    if manifest["dataset_file"] != path.name:
        raise ValueError("Manifest does not describe this replay file")

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != manifest["sha256"]:
        raise ValueError("Replay digest does not match the manifest")
    with path.open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source]
    if len(records) != manifest["transitions"]:
        raise ValueError("Replay length does not match the manifest")
    return records, manifest


def summarize_replays(records: list[dict[str, Any]]) -> list[EpisodeSummary]:
    episodes: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        episodes[record["episode_id"]].append(record)

    summaries = []
    for episode_id, transitions in sorted(episodes.items()):
        if [record["step_id"] for record in transitions] != list(range(len(transitions))):
            raise ValueError(f"Episode {episode_id} has missing or reordered steps")
        if any(record["terminated"] or record["truncated"] for record in transitions[:-1]):
            raise ValueError(f"Episode {episode_id} contains transitions after completion")
        if not (transitions[-1]["terminated"] or transitions[-1]["truncated"]):
            raise ValueError(f"Episode {episode_id} is incomplete")
        if any(
            record["behavior_policy"] != transitions[0]["behavior_policy"] for record in transitions
        ):
            raise ValueError(f"Episode {episode_id} changes behavior policy")
        summaries.append(
            EpisodeSummary(
                episode_id=episode_id,
                behavior_policy=transitions[0]["behavior_policy"],
                length=len(transitions),
                total_reward=sum(record["reward"] for record in transitions),
                success=transitions[-1]["terminated"],
                blocked_moves=sum(record["blocked_move"] for record in transitions),
                optimal_steps=transitions[0]["optimal_steps"],
            )
        )
    return summaries


def eda_metrics(summaries: list[EpisodeSummary]) -> dict[str, float]:
    if not summaries:
        raise ValueError("Replay dataset is empty")
    episodes = len(summaries)
    transitions = sum(summary.length for summary in summaries)
    return {
        "episodes": float(episodes),
        "transitions": float(transitions),
        "success_rate": sum(summary.success for summary in summaries) / episodes,
        "mean_return": sum(summary.total_reward for summary in summaries) / episodes,
        "mean_episode_length": transitions / episodes,
        "blocked_move_rate": sum(summary.blocked_moves for summary in summaries) / transitions,
    }


def log_eda(path: Path, tracking_uri: str, experiment_name: str) -> str:
    import matplotlib.pyplot as plt
    import mlflow
    import pandas as pd
    from mlflow.data.code_dataset_source import CodeDatasetSource
    from mlflow.data.pandas_dataset import from_pandas

    plt.switch_backend("Agg")
    records, manifest = load_replays(path)
    summaries = summarize_replays(records)
    if len(summaries) != manifest["episodes"]:
        raise ValueError("Episode count does not match the manifest")
    metrics = eda_metrics(summaries)
    frame = pd.json_normalize(records)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"eda-seed-{manifest['seed']}") as run:
        dataset = from_pandas(
            frame,
            # The dataset is synthetic, so its origin is the generator rather than a file path
            source=CodeDatasetSource(
                tags={
                    "uri": manifest["source"],
                    "generator": manifest["generator"],
                    "simulator_version": manifest["simulator_version"],
                    "seed": str(manifest["seed"]),
                }
            ),
            name="hex-navigation-replays",
            digest=manifest["sha256"][:32],
        )
        mlflow.log_input(dataset, context="analysis")
        mlflow.log_artifact(str(path), artifact_path="dataset")
        mlflow.log_artifact(str(path.with_suffix(".manifest.json")), artifact_path="dataset")
        mlflow.set_tags(
            {
                "dataset.source": manifest["source"],
                "dataset.generator": manifest["generator"],
                "dataset.sha256": manifest["sha256"],
                "simulator.version": manifest["simulator_version"],
            }
        )
        mlflow.log_params(
            {
                "seed": manifest["seed"],
                "map_radius": manifest["map"]["radius"],
                "max_steps": manifest["max_steps"],
                "random_fraction": manifest["behavior_policy"]["random_fraction"],
                "exploration_probability": manifest["behavior_policy"]["exploration_probability"],
            }
        )
        mlflow.log_metrics(metrics)

        by_policy: dict[str, list[EpisodeSummary]] = defaultdict(list)
        for summary in summaries:
            by_policy[summary.behavior_policy].append(summary)
        policy_metrics = {
            policy: {
                "episodes": len(group),
                "success_rate": sum(item.success for item in group) / len(group),
                "mean_return": sum(item.total_reward for item in group) / len(group),
                "mean_length": sum(item.length for item in group) / len(group),
            }
            for policy, group in by_policy.items()
        }
        mlflow.log_dict({"overall": metrics, "by_policy": policy_metrics}, "eda/summary.json")

        figure, axis = plt.subplots()
        for policy, group in sorted(by_policy.items()):
            axis.hist([item.length for item in group], bins=20, alpha=0.6, label=policy)
        axis.set(title="Episode lengths by behavior policy", xlabel="Steps", ylabel="Episodes")
        axis.legend()
        figure.tight_layout()
        mlflow.log_figure(figure, "eda/episode_lengths.png")
        plt.close(figure)

        distances = sorted({item.optimal_steps for item in summaries})
        figure, axis = plt.subplots()
        for policy, group in sorted(by_policy.items()):
            values = [
                sum(item.success for item in group if item.optimal_steps == distance)
                / sum(item.optimal_steps == distance for item in group)
                if any(item.optimal_steps == distance for item in group)
                else float("nan")
                for distance in distances
            ]
            axis.plot(distances, values, marker="o", label=policy)
        axis.set(
            title="Success rate by shortest path length",
            xlabel="Shortest path length",
            ylabel="Success rate",
            ylim=(0, 1.05),
        )
        axis.legend()
        figure.tight_layout()
        mlflow.log_figure(figure, "eda/success_by_distance.png")
        plt.close(figure)

        visits: dict[HexCoord, int] = defaultdict(int)
        for record in records:
            state = record["state"]
            visits[HexCoord(state["q"], state["r"])] += 1
        figure, axis = plt.subplots()
        cells = sorted(visits)
        scatter = axis.scatter(
            [cell.q + cell.r / 2 for cell in cells],
            [cell.r * 3**0.5 / 2 for cell in cells],
            c=[visits[cell] for cell in cells],
            s=180,
            cmap="viridis",
        )
        for blocked in manifest["map"]["blocked"]:
            axis.scatter(
                blocked["q"] + blocked["r"] / 2,
                blocked["r"] * 3**0.5 / 2,
                marker="x",
                c="red",
            )
        figure.colorbar(scatter, ax=axis, label="Visits")
        axis.set(title="Visited hex cells", aspect="equal")
        figure.tight_layout()
        mlflow.log_figure(figure, "eda/visited_cells.png")
        plt.close(figure)
        return run.info.run_id


def main() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is not None:
                reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Analyze replay data and log EDA to MLflow")
    parser.add_argument("--dataset", type=Path, default=Path("data/replays/navigation.jsonl"))
    parser.add_argument(
        "--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    parser.add_argument("--experiment", default="hex-navigation")
    args = parser.parse_args()
    run_id = log_eda(args.dataset, args.tracking_uri, args.experiment)
    print(f"EDA logged to MLflow run {run_id}")


if __name__ == "__main__":
    main()
