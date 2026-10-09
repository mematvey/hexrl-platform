import logging
from dataclasses import dataclass
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from hexrl_platform.inference.config import InferenceSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LoadedModel:
    model: Any
    name: str
    alias: str
    version: str
    run_id: str | None
    max_steps: int


def load_champion(settings: InferenceSettings) -> LoadedModel:
    """Резолвит alias в номер версии и грузит именно её.

    Если между резолвом и загрузкой alias переставят, в метаданных и в памяти
    окажутся разные версии — поэтому грузим по номеру.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

    model_version = client.get_model_version_by_alias(settings.model_name, settings.model_alias)
    model = mlflow.pyfunc.load_model(f"models:/{settings.model_name}/{model_version.version}")
    encoder = model.unwrap_python_model().policy.encoder

    return LoadedModel(
        model=model,
        name=settings.model_name,
        alias=settings.model_alias,
        version=str(model_version.version),
        run_id=model_version.run_id,
        max_steps=encoder.max_steps,
    )
