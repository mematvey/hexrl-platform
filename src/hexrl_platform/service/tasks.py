import asyncio
import logging
import math
from datetime import timedelta
from uuid import uuid4

import anyio
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hexrl_platform.core.config import Settings
from hexrl_platform.core.logging import task_context
from hexrl_platform.core.telemetry import operation, record_error
from hexrl_platform.task_pipeline.config import PipelineSettings
from hexrl_platform.task_pipeline.errors import QueueOverloadedError, TopologyError
from hexrl_platform.task_pipeline.publisher import TaskPublisher
from hexrl_platform.task_pipeline.repository import TaskRepository
from hexrl_platform.task_pipeline.schemas import HexState, TaskMessage

logger = logging.getLogger(__name__)


class TaskUnavailableError(Exception):
    def __init__(self, reason: str, task_id: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.task_id = task_id


def retry_after(settings: Settings, pipeline: PipelineSettings) -> int:
    # Число реплик — оценка из конфига; Management API не участвует в управлении нагрузкой.
    return max(
        1,
        math.ceil(
            pipeline.inference_delay_s
            / (settings.estimated_worker_count * pipeline.worker_concurrency)
        ),
    )


async def _remove_pending(session: AsyncSession, message: TaskMessage, timeout_s: float) -> None:
    try:
        async with asyncio.timeout(timeout_s):
            await session.rollback()
            await TaskRepository(session).delete_pending(message.task_id)
            await session.commit()
    except (SQLAlchemyError, TimeoutError):
        logger.error("pending_cleanup_failed", extra={"task_id": str(message.task_id)})


async def submit_task(
    payload: HexState, session: AsyncSession, publisher: TaskPublisher, settings: Settings
) -> TaskMessage:
    message = TaskMessage(task_id=uuid4(), payload=payload)
    with task_context(str(message.task_id)):
        return await _submit(message, session, publisher, settings)


async def _submit(
    message: TaskMessage, session: AsyncSession, publisher: TaskPublisher, settings: Settings
) -> TaskMessage:
    repository = TaskRepository(session)
    try:
        async with asyncio.timeout(settings.task_submission_timeout_s):
            with operation("db.create_task"):
                await repository.create_pending(message)
                await session.commit()
            await publisher.publish(message)
            await repository.mark_accepted(message.task_id)
            await session.commit()
            logger.info("task_accepted")
    except asyncio.CancelledError:
        # ASGI отменяет через cancel scope: одного asyncio.shield недостаточно,
        # cleanup должен закончиться до закрытия request-scoped сессии.
        with anyio.CancelScope(shield=True):
            await _remove_pending(session, message, settings.task_cleanup_timeout_s)
        raise
    except TopologyError as error:
        record_error(error)
        await _remove_pending(session, message, settings.task_cleanup_timeout_s)
        logger.error("task_topology_error", extra={"task_id": str(message.task_id)})
        raise TaskUnavailableError("pipeline_unavailable", str(message.task_id)) from error
    except (QueueOverloadedError, TimeoutError, SQLAlchemyError) as error:
        record_error(error)
        await _remove_pending(session, message, settings.task_cleanup_timeout_s)
        logger.warning("task_not_accepted", extra={"task_id": str(message.task_id)})
        raise TaskUnavailableError("queue_overloaded", str(message.task_id)) from error
    return message


async def cleanup_pending(sessions: async_sessionmaker[AsyncSession], older_than_s: float) -> int:
    async with sessions() as session:
        removed = await TaskRepository(session).delete_stale_pending(
            timedelta(seconds=older_than_s)
        )
        await session.commit()
        return removed


async def pending_cleanup_loop(
    sessions: async_sessionmaker[AsyncSession], settings: Settings
) -> None:
    while True:
        await asyncio.sleep(settings.pending_cleanup_interval_s)
        try:
            async with asyncio.timeout(settings.task_cleanup_timeout_s):
                await cleanup_pending(sessions, settings.pending_task_ttl_s)
        except (SQLAlchemyError, TimeoutError):
            logger.warning("stale_pending_cleanup_failed")
