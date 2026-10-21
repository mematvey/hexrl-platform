import asyncio
import logging

from hexrl_platform.inference.model_loader import LoadedModel
from hexrl_platform.inference.predict import InvalidStateError, predict_action
from hexrl_platform.inference.schemas import Direction, ModelInfo, ProcessRequest, ProcessResponse
from hexrl_platform.rl.hex_grid import DIRECTIONS
from hexrl_platform.task_pipeline.schemas import HexState

logger = logging.getLogger(__name__)

__all__ = ["InvalidStateError", "PolicyService"]


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

    async def process(self, request: ProcessRequest) -> ProcessResponse:
        state = HexState(
            q=request.q,
            r=request.r,
            goal_q=request.goal_q,
            goal_r=request.goal_r,
            steps_remaining=request.steps_remaining,
        )
        # torch держит GIL, поэтому уводим инференс из event loop
        action = await asyncio.to_thread(predict_action, self._loaded.model, state)
        step = DIRECTIONS[action]
        return ProcessResponse(
            action=action,
            direction=Direction(dq=step.q, dr=step.r),
            model=self.info,
        )
