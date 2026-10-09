import argparse
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import mlflow

from hexrl_platform.rl import plots
from hexrl_platform.rl.eda import load_replays
from hexrl_platform.rl.evaluation import evaluate_policy, summarize_results
from hexrl_platform.rl.observation import ObservationEncoder
from hexrl_platform.rl.policies import Policy, RandomPolicy, ShortestPathPolicy
from hexrl_platform.rl.replay import map_from_manifest
from hexrl_platform.rl.study import (
    COMPARISON_LABEL,
    RANDOM_LABEL,
    SHORTEST_PATH_LABEL,
    StudyConfig,
    StudyContext,
    load_study_config,
)
from hexrl_platform.rl.tasks import PARTITIONS, navigation_tasks, save_task_split, split_tasks
from hexrl_platform.rl.tracking import (
    code_tags,
    evaluate_and_log,
    log_run_setup,
    log_test_diagnostics,
    prefixed,
)
from hexrl_platform.rl.training import DQNConfig, load_policy, train_dqn

DEFAULT_CONFIG = Path("configs/navigation-dqn.toml")

Curve = tuple[tuple[int, dict[str, float]], ...]


@dataclass(frozen=True, slots=True)
class RunOutcome:
    label: str
    run_id: str
    seed: int | None
    validation: dict[str, float]
    test: dict[str, float]
    test_lengths: tuple[int, ...]
    curve: Curve = ()


def _run_baseline(
    context: StudyContext, label: str, policy: Policy, seed: int | None
) -> RunOutcome:
    run_name = label if seed is None else f"{label}-seed-{seed}"
    with mlflow.start_run(run_name=run_name) as run:
        log_run_setup(context, label, seed, ("validation", "test"))
        validation, _ = evaluate_and_log("val", policy, context.tasks["validation"], step=0)
        test, results = evaluate_and_log("test", policy, context.tasks["test"])
        log_test_diagnostics(context, label, results)
        return RunOutcome(
            label, run.info.run_id, seed, validation, test, tuple(r.length for r in results)
        )


def _run_dqn(context: StudyContext, label: str, config: DQNConfig, seed: int) -> RunOutcome:
    validation_tasks = context.tasks["validation"]
    with mlflow.start_run(run_name=f"{label}-seed-{seed}") as run:
        log_run_setup(context, label, seed, PARTITIONS)
        mlflow.set_tag("algorithm", "DQN")
        mlflow.log_params(config.to_params())
        result = train_dqn(
            config,
            context.encoder,
            context.tasks["train"],
            lambda policy: summarize_results(evaluate_policy(policy, validation_tasks)),
            seed=seed,
            output_dir=context.output_dir / label / f"seed-{seed}",
            on_evaluation=lambda step, metrics: mlflow.log_metrics(
                prefixed("val", metrics), step=step
            ),
        )
        mlflow.log_param("device_resolved", result.device)
        mlflow.log_metric("best_step", result.best_step)
        mlflow.log_artifacts(str(result.best_policy_dir), "policy/best")
        mlflow.log_artifacts(str(result.final_policy_dir), "policy/final")
        # Test tasks are touched once, after the checkpoint was chosen on validation
        test, results = evaluate_and_log(
            "test", load_policy(result.best_policy_dir), context.tasks["test"]
        )
        log_test_diagnostics(context, label, results)
        return RunOutcome(
            label,
            run.info.run_id,
            seed,
            result.best_metrics,
            test,
            tuple(r.length for r in results),
            result.curve,
        )


def _aggregate(values: Sequence[float]) -> dict[str, float | None]:
    finite = [value for value in values if not math.isnan(value)]
    if not finite:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": mean(finite), "std": pstdev(finite), "n": len(finite)}


def _log_comparison(context: StudyContext, outcomes: Sequence[RunOutcome]) -> str:
    by_label: dict[str, list[RunOutcome]] = {}
    for outcome in outcomes:
        by_label.setdefault(outcome.label, []).append(outcome)
    dqn_labels = [label for label, _ in context.study.variants]

    with mlflow.start_run(run_name=COMPARISON_LABEL) as run:
        mlflow.set_tags(
            {"study": context.study.name, "variant": COMPARISON_LABEL, **context.code_tags}
        )
        mlflow.log_dict(context.study.to_dict(), "config/study.json")
        summary: dict[str, Any] = {}
        for label, group in by_label.items():
            summary[label] = {
                metric: _aggregate([outcome.test[metric] for outcome in group])
                for metric in group[0].test
            }
            summary[label]["run_ids"] = [outcome.run_id for outcome in group]
            success = summary[label]["success_rate"]["mean"]
            mlflow.log_metric(f"{label}/test_success_rate_mean", success)

        random_success = summary[RANDOM_LABEL]["success_rate"]["mean"]
        summary["dqn_beats_random"] = {
            label: summary[label]["success_rate"]["mean"] > random_success for label in dqn_labels
        }
        mlflow.log_dict(summary, "comparison/summary.json")

        validation_references = {
            f"{RANDOM_LABEL} (mean)": mean(
                outcome.validation["success_rate"] for outcome in by_label[RANDOM_LABEL]
            ),
            SHORTEST_PATH_LABEL: by_label[SHORTEST_PATH_LABEL][0].validation["success_rate"],
        }
        curves = {
            label: [
                [(step, metrics["success_rate"]) for step, metrics in outcome.curve]
                for outcome in by_label[label]
            ]
            for label in dqn_labels
        }
        mlflow.log_figure(
            plots.learning_curves(curves, "success_rate", validation_references),
            "comparison/learning_curves.png",
        )
        mlflow.log_figure(
            plots.variant_comparison(
                {
                    label: [outcome.test["success_rate"] for outcome in group]
                    for label, group in by_label.items()
                },
                "success_rate",
            ),
            "comparison/test_success_rate.png",
        )
        mlflow.log_figure(
            plots.episode_lengths(
                {
                    label: [length for outcome in group for length in outcome.test_lengths]
                    for label, group in by_label.items()
                },
                context.encoder.max_steps,
            ),
            "comparison/test_episode_lengths.png",
        )
        return run.info.run_id


def run_study(study: StudyConfig, *, tracking_uri: str, output_dir: Path) -> list[str]:
    records, replay_manifest = load_replays(study.replays)
    split = split_tasks(records, replay_manifest["sha256"])
    split_path = output_dir / "task_split.json"
    split_manifest = save_task_split(split, split_path)
    hex_map = map_from_manifest(replay_manifest)
    max_steps = replay_manifest["max_steps"]
    context = StudyContext(
        study=study,
        replay_manifest=replay_manifest,
        split=split,
        split_manifest=split_manifest,
        split_path=split_path,
        encoder=ObservationEncoder(hex_map, max_steps),
        tasks={
            name: navigation_tasks(split.partition(name), hex_map, max_steps) for name in PARTITIONS
        },
        output_dir=output_dir,
        code_tags=code_tags(),
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(study.experiment)
    outcomes = [_run_baseline(context, SHORTEST_PATH_LABEL, ShortestPathPolicy(hex_map), None)]
    outcomes += [
        _run_baseline(context, RANDOM_LABEL, RandomPolicy(seed), seed) for seed in study.seeds
    ]
    outcomes += [
        _run_dqn(context, label, config, seed)
        for label, config in study.variants
        for seed in study.seeds
    ]
    comparison_id = _log_comparison(context, outcomes)
    return [outcome.run_id for outcome in outcomes] + [comparison_id]


def main() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is not None:
                reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Train DQN variants and compare them to baselines")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiments"))
    parser.add_argument(
        "--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    args = parser.parse_args()

    study = load_study_config(args.config)
    run_ids = run_study(
        study,
        tracking_uri=args.tracking_uri,
        output_dir=args.output_dir / study.name,
    )
    print(f"Logged {len(run_ids)} runs to MLflow experiment '{study.experiment}'")


if __name__ == "__main__":
    main()
