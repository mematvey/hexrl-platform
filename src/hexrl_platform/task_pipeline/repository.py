import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from hexrl_platform.db.models import InferenceResult, InferenceTask
from hexrl_platform.task_pipeline.schemas import Prediction, TaskMessage, TaskStatus

logger = logging.getLogger(__name__)

ERROR_LIMIT = 1000

# postgres в проде, sqlite в тестах: ON CONFLICT DO NOTHING у них одинаков по смыслу
_DIALECT_INSERTS: dict[str, Callable[..., Any]] = {
    "postgresql": postgresql_insert,
    "sqlite": sqlite_insert,
}


class TaskRepository:
    """Операции с БД. Транзакцией управляет вызывающий: commit делает воркер перед ack."""

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

    async def ensure_task(self, message: TaskMessage) -> None:
        """Создаёт строку, если её нет.

        API мог ответить 503 с неопределённым исходом (publish прошёл, запись - нет),
        а сообщение всё же доехало. Падать в этом случае нельзя: задача настоящая.
        """
        statement = (
            self._insert(InferenceTask)
            .values(
                task_id=message.task_id,
                batch_id=message.batch_id,
                status=TaskStatus.ACCEPTED,
                payload=message.payload.model_dump(),
                submitted_at=message.submitted_at,
            )
            .on_conflict_do_nothing(index_elements=[InferenceTask.task_id])
        )
        await self._session.execute(statement)

    async def mark_processing(self, task_id: UUID) -> None:
        await self._session.execute(
            update(InferenceTask)
            .where(
                InferenceTask.task_id == task_id,
                InferenceTask.status != TaskStatus.COMPLETED,
            )
            .values(status=TaskStatus.PROCESSING, updated_at=datetime.now(UTC))
        )

    async def save_result(self, task_id: UUID, prediction: Prediction, model_version: str) -> bool:
        """Атомарно пишет результат и переводит задачу в COMPLETED.

        Возвращает False, если результат уже существовал: повторная доставка не должна
        переписывать чужой результат, в том числе посчитанный другой версией модели.
        Защита - уникальный ключ в БД, а не предварительная проверка: две конкурентные
        доставки обе увидели бы "результата нет".
        """
        statement = (
            self._insert(InferenceResult)
            .values(
                task_id=task_id,
                action=prediction.action,
                model_version=model_version,
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
        await self._session.execute(
            update(InferenceTask)
            .where(InferenceTask.task_id == task_id)
            .values(
                status=TaskStatus.FAILED,
                last_error=error[:ERROR_LIMIT],
                attempts=attempts,
                updated_at=datetime.now(UTC),
            )
        )

    async def record_attempt(self, task_id: UUID, attempts: int, error: str) -> None:
        """Промежуточная неудача: задача вернётся из retry-очереди, статус не финальный."""
        await self._session.execute(
            update(InferenceTask)
            .where(InferenceTask.task_id == task_id)
            .values(
                status=TaskStatus.ACCEPTED,
                last_error=error[:ERROR_LIMIT],
                attempts=attempts,
                updated_at=datetime.now(UTC),
            )
        )

    # --- для Части 3 (API) -------------------------------------------------

    async def create_pending(self, message: TaskMessage) -> None:
        """PENDING живёт только между INSERT и ответом брокера."""
        statement = (
            self._insert(InferenceTask)
            .values(
                task_id=message.task_id,
                batch_id=message.batch_id,
                status=TaskStatus.PENDING,
                payload=message.payload.model_dump(),
                submitted_at=message.submitted_at,
            )
            .on_conflict_do_nothing(index_elements=[InferenceTask.task_id])
        )
        await self._session.execute(statement)

    async def mark_accepted(self, task_id: UUID) -> None:
        await self._session.execute(
            update(InferenceTask)
            .where(InferenceTask.task_id == task_id, InferenceTask.status == TaskStatus.PENDING)
            .values(status=TaskStatus.ACCEPTED, updated_at=datetime.now(UTC))
        )

    async def delete_pending(self, task_id: UUID) -> None:
        """Брокер отказал в публикации: строка PENDING не должна остаться висеть."""
        await self._session.execute(
            delete(InferenceTask).where(
                InferenceTask.task_id == task_id,
                InferenceTask.status == TaskStatus.PENDING,
            )
        )
