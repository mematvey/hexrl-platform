import argparse
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from hexrl_platform.core.logging import configure_logging
from hexrl_platform.rl.experiment import run_study
from hexrl_platform.rl.inputs import ensure_inputs
from hexrl_platform.rl.observation import ObservationEncoder
from hexrl_platform.rl.registry import (
    RegistryConfig,
    load_registry_config,
    register_study,
    verify_version,
)
from hexrl_platform.rl.replay import ReplayGenerationConfig, generate_replays
from hexrl_platform.rl.study import StudyConfig, load_study_config

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/navigation-dqn-smoke.toml")


class BootstrapError(Exception):
    """Модель для стенда нельзя ни найти, ни честно подготовить."""


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    model_name: str
    alias: str
    version: str
    trained: bool

    @property
    def model_uri(self) -> str:
        return f"models:/{self.model_name}/{self.version}"


def champion_version(client: MlflowClient, model_name: str, alias: str) -> str | None:
    try:
        model = client.get_registered_model(model_name)
    except MlflowException as error:
        if error.error_code == "RESOURCE_DOES_NOT_EXIST":
            return None
        raise
    version = model.aliases.get(alias)
    return None if version is None else str(version)


def ensure_champion(
    study: StudyConfig,
    registry: RegistryConfig,
    *,
    tracking_uri: str,
    work_dir: Path,
    replays: ReplayGenerationConfig | None = None,
) -> BootstrapResult:
    """Повторный запуск не создаёт новую версию: обучение только при отсутствии alias."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    version = champion_version(client, registry.model_name, registry.alias)
    trained = version is None
    if version is None:
        replay_path = work_dir / "replays" / "navigation.jsonl"
        generate_replays(replay_path, replays or ReplayGenerationConfig())
        study = replace(study, replays=replay_path)
        run_study(study, tracking_uri=tracking_uri, output_dir=work_dir / "experiments")
        version = register_study(study, registry, tracking_uri=tracking_uri).champion_version
    # Загрузка с predict: alias без доступного артефакта - ошибка, а не готовность
    verify_version(registry.model_name, version)
    return BootstrapResult(registry.model_name, registry.alias, version, trained)


def bootstrap(
    config: Path,
    *,
    model_name: str,
    alias: str,
    tracking_uri: str,
    work_dir: Path,
    inputs_dir: Path | None,
) -> BootstrapResult:
    study = load_study_config(config)
    registry = load_registry_config(config)
    if (registry.model_name, registry.alias) != (model_name, alias):
        # Иначе воркер ждал бы одну модель, а обучилась бы другая
        raise BootstrapError(
            f"{config.name} registers {registry.model_name}@{registry.alias}, "
            f"but the worker loads {model_name}@{alias}"
        )
    result = ensure_champion(study, registry, tracking_uri=tracking_uri, work_dir=work_dir)
    logger.info(
        "bootstrap_model_ready",
        extra={
            "component": result.model_name,
            "model_alias": result.alias,
            "model_version": result.version,
            "reason": "trained" if result.trained else "existing",
        },
    )
    if inputs_dir is not None:
        info = mlflow.models.get_model_info(result.model_uri)
        if info.metadata is None or "encoder" not in info.metadata:
            raise BootstrapError(f"{result.model_uri} has no encoder metadata")
        encoder = ObservationEncoder.from_config(info.metadata["encoder"])
        for name, generated in ensure_inputs(encoder, inputs_dir, result.model_uri).items():
            logger.info(
                "bootstrap_inputs_ready",
                extra={"component": name, "reason": "generated" if generated else "current"},
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the champion model for a clean stand")
    parser.add_argument(
        "--config", type=Path, default=Path(os.getenv("BOOTSTRAP_CONFIG", str(DEFAULT_CONFIG)))
    )
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/hexrl-bootstrap"))
    parser.add_argument("--inputs-dir", type=Path)
    args = parser.parse_args()
    configure_logging(os.getenv("LOG_LEVEL", "INFO"), "hexrl-bootstrap")
    try:
        bootstrap(
            args.config,
            model_name=os.environ["MODEL_NAME"],
            alias=os.getenv("MODEL_ALIAS", "champion"),
            tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
            work_dir=args.work_dir,
            inputs_dir=args.inputs_dir,
        )
    except Exception:
        # Ненулевой код останавливает воркер: depends_on ждёт успешного завершения
        logger.exception("bootstrap_failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
