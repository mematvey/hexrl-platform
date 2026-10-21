import logging
from typing import Protocol

import pandas as pd
from mlflow.exceptions import MlflowException

from hexrl_platform.rl.packaging import INPUT_COLUMNS, OUTPUT_COLUMN
from hexrl_platform.task_pipeline.schemas import HexState

logger = logging.getLogger(__name__)


class SupportsPredict(Protocol):
    """Структурный тип: подходят и pyfunc-модель MLflow, и фейк в тестах."""

    def predict(self, data: pd.DataFrame) -> pd.DataFrame: ...


class InvalidStateError(Exception):
    """Состояние не подходит модели: стена, нецелевая клетка или шаги вне диапазона."""


def predict_action(model: SupportsPredict, payload: HexState) -> int:
    """Синхронный вызов модели: его зовут из потока runner'а, не из event loop.

    pyfunc проверяет сигнатуру входа до predict, поэтому несовпадение приходит
    MlflowException, а доменная ошибка модели - ValueError. Снаружи обе означают
    одно: это состояние модели не подходит, повторять бессмысленно.
    """
    frame = pd.DataFrame([{column: getattr(payload, column) for column in INPUT_COLUMNS}])
    try:
        return int(model.predict(frame)[OUTPUT_COLUMN].iloc[0])
    except (ValueError, MlflowException) as exc:
        logger.warning("invalid_state", extra={"error_type": type(exc).__name__})
        raise InvalidStateError(str(exc)) from exc
