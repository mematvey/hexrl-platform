import logging
from abc import ABC, abstractmethod

import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PyFuncModel

from hexrl_platform.rl.packaging import INPUT_COLUMNS, OUTPUT_COLUMN
from hexrl_platform.task_pipeline.schemas import HexState

logger = logging.getLogger(__name__)


class PredictModel(ABC):
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.DataFrame: ...


class PyfuncModel(PredictModel):
    def __init__(self, model: PyFuncModel) -> None:
        self._model = model

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        result = self._model.predict(data)
        if not isinstance(result, pd.DataFrame):
            raise TypeError(f"model returned {type(result).__name__}, expected DataFrame")
        return result


class InvalidStateError(Exception):
    """Состояние не подходит модели: стена, нецелевая клетка или шаги вне диапазона"""


def predict_action(model: PredictModel, payload: HexState) -> int:
    frame = pd.DataFrame([{column: getattr(payload, column) for column in INPUT_COLUMNS}])
    try:
        return int(model.predict(frame)[OUTPUT_COLUMN].iloc[0])
    except (ValueError, MlflowException) as exc:
        # неподходящий вход pyfunc отвергает как MlflowException, сама модель - как ValueError
        logger.warning("invalid_state", extra={"error_type": type(exc).__name__})
        raise InvalidStateError(str(exc)) from exc
