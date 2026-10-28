import os
import threading
from dataclasses import replace

import pandas as pd
import uvicorn

from hexrl_platform.inference.config import InferenceSettings, get_inference_settings
from hexrl_platform.inference.model_loader import LoadedModel, load_champion
from hexrl_platform.inference.predict import PredictModel
from hexrl_platform.task_pipeline.worker import create_worker_app


class FaultModel(PredictModel):
    def __init__(self, model: PredictModel, failures: int) -> None:
        self._model = model
        self._remaining = failures
        self._lock = threading.Lock()

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        with self._lock:
            if self._remaining > 0:
                self._remaining -= 1
                raise RuntimeError("benchmark_injected_transient_failure")
        return self._model.predict(data)


def load(settings: InferenceSettings) -> LoadedModel:
    loaded = load_champion(settings)
    failures = int(os.getenv("BENCHMARK_TRANSIENT_FAILURES", "0"))
    if failures < 0:
        raise ValueError("BENCHMARK_TRANSIENT_FAILURES must be non-negative")
    return replace(loaded, model=FaultModel(loaded.model, failures)) if failures else loaded


if __name__ == "__main__":
    settings = get_inference_settings()
    uvicorn.run(
        create_worker_app(loader=load),
        host=settings.inference_host,
        port=settings.inference_port,
        access_log=False,
    )
