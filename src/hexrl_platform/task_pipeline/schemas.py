from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class HexState(BaseModel):
    q: int
    r: int
    goal_q: int
    goal_r: int
    steps_remaining: int = Field(ge=1)


class TaskStatus(StrEnum):
    # живёт только между INSERT и ответом брокера; при отказе строка удаляется
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TaskMessage(BaseModel):
    task_id: UUID
    batch_id: str | None = None
    payload: HexState
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class Prediction:
    action: int
    # раздельно: иначе в отчёте смешаются скорость модели и искусственная нагрузка
    inference_ms: float
    delay_ms: float


class TaskResultView(BaseModel):
    action: int
    model_name: str | None
    model_version: str
    inference_ms: float
    delay_ms: float
    completed_at: datetime


class TaskView(BaseModel):
    task_id: UUID
    batch_id: str | None
    status: TaskStatus
    payload: HexState
    attempts: int
    submitted_at: datetime
    created_at: datetime
    completed_at: datetime | None
    last_error: str | None
    result: TaskResultView | None
