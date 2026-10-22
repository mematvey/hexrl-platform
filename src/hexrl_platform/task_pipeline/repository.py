import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from hexrl_platform.db.models import InferenceResult, InferenceTask
from hexrl_platform.task_pipeline.schemas import (
    HexState,
    Prediction,
    TaskMessage,
    TaskResultView,
    TaskStatus,
    TaskView,
)

logger = logging.getLogger(__name__)

ERROR_LIMIT = 1000

# postgres в проде, sqlite в тестах: ON CONFLICT DO NOTHING у них одинаков по смыслу
_DIALECT_INSERTS: dict[str, Callable[..., Any]] = {
    "postgresql": postgresql_insert,
    "sqlite": sqlite_insert,
}


class TaskRepository:
    """Транзакцией управляет вызывающий: commit делает воркер перед ack"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        dialect = session.get_bind().dialect.name
        insert = _DIALECT_INSERTS.get(dialect)
        if insert is None:
            raise RuntimeError(f"dialect {dialect} has no ON CONFLICT support here")
        self._insert = insert

    async def get_status(self, task_id: UUID) -> TaskStatus | None:
        return await self._session.scalar(
            select(InferenceTask.status).where(InferenceTask.task_id == task_id)
        )

    async def get(self, task_id: UUID) -> TaskView | None:
        row = (
            await self._session.execute(
                select(InferenceTask, InferenceResult)
                .outerjoin(InferenceResult, InferenceResult.task_id == InferenceTask.task_id)
                .where(InferenceTask.task_id == task_id)
            )
        ).one_or_none()
        if row is None:
            return None
        task, result = row
        completed_at = result.created_at if result is not None else None
        if task.status is TaskStatus.FAILED:
            completed_at = task.updated_at
        return TaskView(
            task_id=task.task_id,
            batch_id=task.batch_id,
            status=task.status,
            payload=HexState.model_validate(task.payload),
            attempts=task.attempts,
            submitted_at=task.submitted_at,
            created_at=task.created_at,
            completed_at=completed_at,
            last_error=task.last_error,
            result=TaskResultView(
                action=result.action,
                model_name=result.model_name,
                model_version=result.model_version,
                inference_ms=result.inference_ms,
                delay_ms=result.delay_ms,
                completed_at=result.created_at,
            )
            if result is not None
            else None,
        )

    async def delete_stale_pending(self, older_than: timedelta) -> int:
        if older_than <= timedelta(0):
            raise ValueError("older_than must be positive")
        result = await self._session.execute(
            delete(InferenceTask)
            .where(
                InferenceTask.status == TaskStatus.PENDING,
                InferenceTask.created_at < datetime.now(UTC) - older_than,
            )
            .returning(InferenceTask.task_id)
        )
        return len(result.all())

    async def ensure_task(self, message: TaskMessage) -> None:
        # API мог ответить 503 с неопределённым исходом, а сообщение всё же доехало
        await self._insert_task(message, TaskStatus.ACCEPTED)

    async def create_pending(self, message: TaskMessage) -> None:
        await self._insert_task(message, TaskStatus.PENDING)

    async def mark_processing(self, task_id: UUID) -> None:
        await self._update_unless_completed(task_id, {"status": TaskStatus.PROCESSING})

    async def save_result(
        self,
        task_id: UUID,
        prediction: Prediction,
        model_version: str,
        *,
        model_name: str | None = None,
    ) -> bool:
        """False, если результат уже был: повтор не должен переписывать чужой результат"""
        statement = (
            self._insert(InferenceResult)
            .values(
                task_id=task_id,
                action=prediction.action,
                model_version=model_version,
                model_name=model_name,
                inference_ms=prediction.inference_ms,
                delay_ms=prediction.delay_ms,
            )
            .on_conflict_do_nothing(index_elements=[InferenceResult.task_id])
            # RETURNING отличает вставку от пропуска надёжнее, чем rowcount
            .returning(InferenceResult.task_id)
        )
        inserted = (await self._session.execute(statement)).scalar_one_or_none() is not None
        await self._session.execute(
            update(InferenceTask)
            .where(InferenceTask.task_id == task_id)
            .values(status=TaskStatus.COMPLETED, last_error=None, updated_at=datetime.now(UTC))
        )
        if not inserted:
            logger.info("duplicate_delivery_ignored", extra={"task_id": str(task_id)})
        return inserted

    async def mark_failed(self, task_id: UUID, error: str, attempts: int) -> None:
        await self._update_unless_completed(
            task_id,
            {"status": TaskStatus.FAILED, "last_error": error[:ERROR_LIMIT], "attempts": attempts},
        )

    async def record_attempt(self, task_id: UUID, attempts: int, error: str) -> None:
        # промежуточная неудача: задача вернётся из retry-очереди, статус не финальный
        await self._update_unless_completed(
            task_id,
            {
                "status": TaskStatus.ACCEPTED,
                "last_error": error[:ERROR_LIMIT],
                "attempts": attempts,
            },
        )

    async def mark_accepted(self, task_id: UUID) -> None:
        await self._session.execute(
            update(InferenceTask)
            .where(InferenceTask.task_id == task_id, InferenceTask.status == TaskStatus.PENDING)
            .values(status=TaskStatus.ACCEPTED, updated_at=datetime.now(UTC))
        )

    async def delete_pending(self, task_id: UUID) -> None:
        await self._session.execute(
            delete(InferenceTask).where(
                InferenceTask.task_id == task_id,
                InferenceTask.status == TaskStatus.PENDING,
            )
        )

    async def _insert_task(self, message: TaskMessage, status: TaskStatus) -> None:
        statement = (
            self._insert(InferenceTask)
            .values(
                task_id=message.task_id,
                batch_id=message.batch_id,
                status=status,
                payload=message.payload.model_dump(),
                submitted_at=message.submitted_at,
            )
            .on_conflict_do_nothing(index_elements=[InferenceTask.task_id])
        )
        await self._session.execute(statement)

    async def _update_unless_completed(self, task_id: UUID, values: dict[str, Any]) -> None:
        # успевшая завершиться доставка важнее: её результат уже записан
        await self._session.execute(
            update(InferenceTask)
            .where(
                InferenceTask.task_id == task_id,
                InferenceTask.status != TaskStatus.COMPLETED,
            )
            .values(updated_at=datetime.now(UTC), **values)
        )
