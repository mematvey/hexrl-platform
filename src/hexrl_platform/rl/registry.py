import argparse
import math
import os
import sys
import tempfile
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

from hexrl_platform.rl.hex_grid import DIRECTIONS
from hexrl_platform.rl.packaging import (
    OUTPUT_COLUMN,
    NavigationPolicyModel,
    example_input,
    log_policy_model,
)
from hexrl_platform.rl.study import COMPARISON_LABEL, StudyConfig, load_study_config

PACKAGED_MODEL_TAG = "packaging.model_uri"
CHAMPION_RULE = (
    "highest test/success_rate; candidates within tie_tolerance of the best "
    "are ordered by lower test/path_ratio"
)


@dataclass(frozen=True, slots=True)
class RegistryConfig:
    model_name: str
    alias: str = "champion"
    tie_tolerance: float = 0.01

    def __post_init__(self) -> None:
        if not self.model_name or not self.alias:
            raise ValueError("model_name and alias must not be empty")
        if not 0 <= self.tie_tolerance < 1:
            raise ValueError("tie_tolerance must be in [0, 1)")


def load_registry_config(path: Path) -> RegistryConfig:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if "registry" not in data:
        raise ValueError(f"Study config {path} has no [registry] section")
    return RegistryConfig(**data["registry"])


@dataclass(frozen=True, slots=True)
class Candidate:
    run_id: str
    variant: str
    seed: int
    success_rate: float
    path_ratio: float


def choose_champion(candidates: Sequence[Candidate], tie_tolerance: float) -> Candidate:
    """Best success rate; near-ties are broken by the shorter routes, then deterministically."""
    if not candidates:
        raise ValueError("No candidates to choose from")
    best_success = max(candidate.success_rate for candidate in candidates)
    contenders = [c for c in candidates if best_success - c.success_rate <= tie_tolerance]
    return min(
        contenders,
        key=lambda c: (
            math.inf if math.isnan(c.path_ratio) else c.path_ratio,
            -c.success_rate,
            c.variant,
            c.seed,
        ),
    )


def best_per_variant(candidates: Sequence[Candidate], tie_tolerance: float) -> list[Candidate]:
    variants = sorted({candidate.variant for candidate in candidates})
    return [
        choose_champion([c for c in candidates if c.variant == variant], tie_tolerance)
        for variant in variants
    ]


def find_candidates(client: MlflowClient, study: StudyConfig) -> list[Candidate]:
    experiment = client.get_experiment_by_name(study.experiment)
    if experiment is None:
        raise ValueError(f"MLflow experiment '{study.experiment}' does not exist")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=(
            f"tags.study = '{study.name}' and tags.algorithm = 'DQN' "
            "and attributes.status = 'FINISHED'"
        ),
    )
    return [
        Candidate(
            run_id=run.info.run_id,
            variant=run.data.tags["variant"],
            seed=int(run.data.params["seed"]),
            success_rate=run.data.metrics["test/success_rate"],
            path_ratio=run.data.metrics.get("test/path_ratio", math.nan),
        )
        for run in runs
    ]


def ensure_packaged(client: MlflowClient, candidate: Candidate) -> str:
    """Logs the best checkpoint of a finished run as a pyfunc model, once."""
    run = client.get_run(candidate.run_id)
    if PACKAGED_MODEL_TAG in run.data.tags:
        return run.data.tags[PACKAGED_MODEL_TAG]
    with tempfile.TemporaryDirectory() as directory:
        policy_dir = mlflow.artifacts.download_artifacts(
            run_id=candidate.run_id, artifact_path="policy/best", dst_path=directory
        )
        with mlflow.start_run(run_id=candidate.run_id):
            model_info = log_policy_model(Path(policy_dir))
            mlflow.set_tag(PACKAGED_MODEL_TAG, model_info.model_uri)
    return model_info.model_uri


def ensure_registered(
    client: MlflowClient, model_name: str, candidate: Candidate, model_uri: str, study: str
) -> ModelVersion:
    existing = client.search_model_versions(
        f"name = '{model_name}' and run_id = '{candidate.run_id}'"
    )
    if existing:
        return existing[0]
    return mlflow.register_model(
        model_uri,
        model_name,
        tags={
            "study": study,
            "variant": candidate.variant,
            "seed": str(candidate.seed),
            "test_success_rate": f"{candidate.success_rate:.4f}",
        },
    )


def verify_version(model_name: str, version: str) -> None:
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")
    python_model = model.unwrap_python_model()
    if not isinstance(python_model, NavigationPolicyModel):
        raise TypeError(f"{model_name}/{version} is not a navigation policy model")
    prediction = model.predict(example_input(python_model.policy.encoder))
    if not isinstance(prediction, pd.DataFrame) or OUTPUT_COLUMN not in prediction:
        raise TypeError(f"{model_name}/{version} must return a DataFrame with '{OUTPUT_COLUMN}'")
    if not all(0 <= action < len(DIRECTIONS) for action in prediction[OUTPUT_COLUMN]):
        raise ValueError(f"{model_name}/{version} returned an invalid action")


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    model_name: str
    alias: str
    champion: Candidate
    champion_version: str
    versions: dict[str, str]


def _log_decision(client: MlflowClient, study: StudyConfig, decision: dict[str, Any]) -> None:
    experiment = client.get_experiment_by_name(study.experiment)
    if experiment is None:
        return
    comparison = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.study = '{study.name}' and tags.variant = '{COMPARISON_LABEL}'",
        max_results=1,
    )
    if comparison:
        with mlflow.start_run(run_id=comparison[0].info.run_id):
            mlflow.log_dict(decision, "registry/champion_decision.json")


def register_study(
    study: StudyConfig, registry: RegistryConfig, *, tracking_uri: str
) -> RegistrationResult:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    candidates = find_candidates(client, study)
    if not candidates:
        raise ValueError(f"No finished DQN runs for study '{study.name}'")

    selected = best_per_variant(candidates, registry.tie_tolerance)
    versions: dict[str, str] = {}
    for candidate in selected:
        model_uri = ensure_packaged(client, candidate)
        model_version = ensure_registered(
            client, registry.model_name, candidate, model_uri, study.name
        )
        # MLflow returns version numbers as int or str depending on the call
        version = str(model_version.version)
        verify_version(registry.model_name, version)
        versions[candidate.run_id] = version

    champion = choose_champion(selected, registry.tie_tolerance)
    champion_version = versions[champion.run_id]
    client.set_registered_model_alias(registry.model_name, registry.alias, champion_version)
    client.update_registered_model(
        registry.model_name,
        description=(
            f"Navigation policy for study '{study.name}'. {registry.alias}: {CHAMPION_RULE}"
        ),
    )
    _log_decision(
        client,
        study,
        {
            "rule": CHAMPION_RULE,
            "tie_tolerance": registry.tie_tolerance,
            "model_name": registry.model_name,
            "alias": registry.alias,
            "champion": {"version": champion_version, "run_id": champion.run_id},
            "candidates": [
                {
                    "version": versions[c.run_id],
                    "run_id": c.run_id,
                    "variant": c.variant,
                    "seed": c.seed,
                    "test_success_rate": c.success_rate,
                    "test_path_ratio": None if math.isnan(c.path_ratio) else c.path_ratio,
                }
                for c in selected
            ],
        },
    )
    return RegistrationResult(
        registry.model_name, registry.alias, champion, champion_version, versions
    )


def main() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is not None:
                reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Register the best DQN policies of a study")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    args = parser.parse_args()
    result = register_study(
        load_study_config(args.config),
        load_registry_config(args.config),
        tracking_uri=args.tracking_uri,
    )
    for run_id, model_version in sorted(result.versions.items(), key=lambda item: item[1]):
        print(f"{result.model_name} v{model_version} <- run {run_id}")
    print(
        f"{result.model_name}@{result.alias} -> v{result.champion_version} "
        f"({result.champion.variant}, seed {result.champion.seed})"
    )


if __name__ == "__main__":
    main()
