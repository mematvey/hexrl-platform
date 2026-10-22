from uuid import UUID

from pydantic import BaseModel

from hexrl_platform.task_pipeline.schemas import HexState, TaskStatus, TaskView

TaskRequest = HexState
TaskResponse = TaskView


class TaskAccepted(BaseModel):
    task_id: UUID
    status: TaskStatus = TaskStatus.ACCEPTED
    status_url: str


class TaskError(BaseModel):
    error: str
    message: str
