from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from hexrl_platform.core.config import Settings, get_settings
from hexrl_platform.db.session import get_session
from hexrl_platform.schemas.tasks import TaskAccepted, TaskError, TaskRequest, TaskResponse
from hexrl_platform.service.tasks import TaskUnavailableError, retry_after, submit_task
from hexrl_platform.task_pipeline.config import PipelineSettings, get_pipeline_settings
from hexrl_platform.task_pipeline.publisher import TaskPublisher
from hexrl_platform.task_pipeline.repository import TaskRepository

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])


def get_publisher(request: Request) -> TaskPublisher:
    return request.app.state.task_publisher


@router.post(
    "", status_code=202, response_model=TaskAccepted, responses={503: {"model": TaskError}}
)
async def create_task(
    payload: TaskRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    publisher: Annotated[TaskPublisher, Depends(get_publisher)],
    settings: Annotated[Settings, Depends(get_settings)],
    pipeline: Annotated[PipelineSettings, Depends(get_pipeline_settings)],
) -> TaskAccepted | JSONResponse:
    try:
        message = await submit_task(payload, session, publisher, settings)
    except TaskUnavailableError as error:
        body = TaskError(error=error.reason, message="Task could not be accepted; retry later")
        return JSONResponse(
            status_code=503,
            content=body.model_dump(),
            headers={"Retry-After": str(retry_after(settings, pipeline))},
        )
    location = f"/api/v1/tasks/{message.task_id}"
    return TaskAccepted(task_id=message.task_id, status_url=location)


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: UUID, session: Annotated[AsyncSession, Depends(get_session)]
) -> TaskResponse:
    task = await TaskRepository(session).get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task
