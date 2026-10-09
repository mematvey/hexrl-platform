import asyncio
import logging

import pandas as pd
from mlflow.exceptions import MlflowException

from hexrl_platform.inference.model_loader import LoadedModel
from hexrl_platform.inference.schemas import Direction, ModelInfo, ProcessRequest, ProcessResponse
from hexrl_platform.rl.hex_grid import DIRECTIONS
from hexrl_platform.rl.packaging import INPUT_COLUMNS, OUTPUT_COLUMN

logger = logging.getLogger(__name__)


class InvalidStateError(Exception):
    """Состояние не подходит модели: стена, нецелевая клетка или шаги вне диапазона."""


class PolicyService:
    """Знает про модель и домен, ничего не знает про HTTP."""

    def __init__(self, loaded: LoadedModel) -> None:
        self._loaded = loaded

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name=self._loaded.name,
            version=self._loaded.version,
            alias=self._loaded.alias,
            run_id=self._loaded.run_id,
            max_steps=self._loaded.max_steps,
        )

    def _predict(self, request: ProcessRequest) -> int:
        frame = pd.DataFrame([{column: getattr(request, column) for column in INPUT_COLUMNS}])
        return int(self._loaded.model.predict(frame)[OUTPUT_COLUMN].iloc[0])

    async def process(self, request: ProcessRequest) -> ProcessResponse:
        try:
            # torch держит GIL, поэтому уводим инференс из event loop
            action = await asyncio.to_thread(self._predict, request)
        except (ValueError, MlflowException) as exc:
            logger.warning("invalid_state", extra={"error_type": type(exc).__name__})
            raise InvalidStateError(str(exc)) from exc
        step = DIRECTIONS[action]
        return ProcessResponse(
            action=action,
            direction=Direction(dq=step.q, dr=step.r),
            model=self.info,
        )
