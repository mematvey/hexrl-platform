import subprocess
from collections.abc import Sequence
from typing import Any

import mlflow
import pandas as pd
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.pandas_dataset import from_pandas

from hexrl_platform.rl import plots
from hexrl_platform.rl.evaluation import EpisodeResult, evaluate_policy, summarize_results
from hexrl_platform.rl.hex_grid import DIRECTIONS
from hexrl_platform.rl.navigation import NavigationTask
from hexrl_platform.rl.policies import Policy
from hexrl_platform.rl.study import StudyContext
from hexrl_platform.rl.tasks import SPLIT_RULE, SPLIT_VERSION, Partition, pairs_to_records

SPLIT_SOURCE_URI = "derived://hexrl/navigation-task-split"
DATASET_CONTEXTS: dict[Partition, str] = {
    "train": "training",
    "validation": "validation",
    "test": "test",
}
FAILED_TRAJECTORIES = 3


def code_tags() -> dict[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"code.git_commit": "unknown", "code.git_dirty": "unknown"}
    return {"code.git_commit": commit, "code.git_dirty": str(bool(status.strip())).lower()}


def prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{name}": value for name, value in metrics.items()}


def log_task_inputs(context: StudyContext, partitions: Sequence[Partition]) -> None:
    for name in partitions:
        dataset = from_pandas(
            pd.DataFrame(pairs_to_records(context.split.partition(name))),
            source=CodeDatasetSource(
                tags={
                    "uri": SPLIT_SOURCE_URI,
                    "parent_sha256": context.split.parent_sha256,
                    "split_version": str(SPLIT_VERSION),
                    "rule": SPLIT_RULE,
                    "partition": name,
                }
            ),
            name=f"navigation-tasks-{name}",
            digest=context.split_manifest["partition_sha256"][name][:32],
        )
        mlflow.log_input(dataset, context=DATASET_CONTEXTS[name])
    mlflow.log_artifact(str(context.split_path), artifact_path="dataset")
    mlflow.log_artifact(str(context.split_path.with_suffix(".manifest.json")), "dataset")


def log_run_setup(
    context: StudyContext, label: str, seed: int | None, partitions: Sequence[Partition]
) -> None:
    manifest = context.replay_manifest
    mlflow.set_tags(
        {
            "study": context.study.name,
            "variant": label,
            "dataset.parent_sha256": context.split.parent_sha256,
            "split.sha256": context.split_manifest["sha256"],
            "simulator.version": manifest["simulator_version"],
            "encoder.version": context.encoder.version,
            **context.code_tags,
        }
    )
    params: dict[str, Any] = {
        "map_radius": manifest["map"]["radius"],
        "max_steps": manifest["max_steps"],
        "encoder_version": context.encoder.version,
        "split_version": SPLIT_VERSION,
        **{f"reward.{name}": value for name, value in manifest["rewards"].items()},
    }
    if seed is not None:
        params["seed"] = seed
    mlflow.log_params(params)
    mlflow.log_dict(context.study.to_dict(), "config/study.json")
    mlflow.log_dict(context.encoder.to_config(), "config/encoder.json")
    mlflow.log_dict({"actions": [{"q": d.q, "r": d.r} for d in DIRECTIONS]}, "config/actions.json")
    log_task_inputs(context, partitions)


def log_test_diagnostics(
    context: StudyContext, label: str, results: Sequence[EpisodeResult]
) -> None:
    mlflow.log_figure(
        plots.episode_lengths({label: [r.length for r in results]}, context.encoder.max_steps),
        "diagnostics/test_episode_lengths.png",
    )
    failures = sorted(
        (result for result in results if not result.success),
        key=lambda result: result.optimal_steps,
    )[:FAILED_TRAJECTORIES]
    for index, result in enumerate(failures):
        title = (
            f"{label}: failed, optimal {result.optimal_steps} steps, "
            f"{result.blocked_moves} wall hits"
        )
        mlflow.log_figure(
            plots.trajectory(
                context.encoder.hex_map, result.start, result.goal, result.trajectory, title
            ),
            f"diagnostics/failed_trajectory_{index}.png",
        )
    mlflow.log_dict(
        {
            "failed_episodes": [
                {
                    "start": [result.start.q, result.start.r],
                    "goal": [result.goal.q, result.goal.r],
                    "optimal_steps": result.optimal_steps,
                    "length": result.length,
                    "blocked_moves": result.blocked_moves,
                    "trajectory": [[cell.q, cell.r] for cell in result.trajectory],
                }
                for result in failures
            ]
        },
        "diagnostics/failed_trajectories.json",
    )


def evaluate_and_log(
    prefix: str, policy: Policy, tasks: Sequence[NavigationTask], step: int | None = None
) -> tuple[dict[str, float], list[EpisodeResult]]:
    results = evaluate_policy(policy, tasks)
    metrics = summarize_results(results)
    mlflow.log_metrics(prefixed(prefix, metrics), step=step)
    return metrics, results
