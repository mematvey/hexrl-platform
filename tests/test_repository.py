import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from hexrl_platform.db.models import InferenceResult, InferenceTask
from hexrl_platform.db.schema import create_schema
from hexrl_platform.task_pipeline.repository import TaskRepository
from hexrl_platform.task_pipeline.schemas import HexState, Prediction, TaskMessage, TaskStatus

PREDICTION = Prediction(action=3, inference_ms=1.5, delay_ms=500.0)


def make_message(task_id=None) -> TaskMessage:
    return TaskMessage(
        task_id=task_id or uuid4(),
        batch_id="batch-1",
        payload=HexState(q=0, r=0, goal_q=2, goal_r=-1, steps_remaining=20),
        submitted_at=datetime.now(UTC),
    )


@pytest.fixture
async def sessions(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'tasks.db').as_posix()}")
    await create_schema(engine)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


async def count_results(sessions, task_id) -> int:
    async with sessions() as session:
        return await session.scalar(
            select(func.count())
            .select_from(InferenceResult)
            .where(InferenceResult.task_id == task_id)
        )


async def test_ensure_task_creates_a_missing_row(sessions):
    message = make_message()
    async with sessions() as session:
        await TaskRepository(session).ensure_task(message)
        await session.commit()

    async with sessions() as session:
        assert await TaskRepository(session).get_status(message.task_id) is TaskStatus.ACCEPTED


async def test_ensure_task_does_not_overwrite_an_existing_row(sessions):
    message = make_message()
    async with sessions() as session:
        repository = TaskRepository(session)
        await repository.ensure_task(message)
        await repository.save_result(message.task_id, PREDICTION, "7")
        await session.commit()

    # вторая доставка того же сообщения не должна откатить статус в ACCEPTED
    async with sessions() as session:
        repository = TaskRepository(session)
        await repository.ensure_task(message)
        await session.commit()
        assert await repository.get_status(message.task_id) is TaskStatus.COMPLETED


async def test_second_delivery_does_not_write_a_second_result(sessions):
    message = make_message()
    async with sessions() as session:
        repository = TaskRepository(session)
        await repository.ensure_task(message)
        assert await repository.save_result(message.task_id, PREDICTION, "7") is True
        await session.commit()

    async with sessions() as session:
        repository = TaskRepository(session)
        other = Prediction(action=5, inference_ms=2.0, delay_ms=0.0)
        assert await repository.save_result(message.task_id, other, "8") is False
        await session.commit()

    assert await count_results(sessions, message.task_id) == 1
    async with sessions() as session:
        stored = await session.get(InferenceResult, message.task_id)
        # результат принадлежит той доставке, что успела первой, вместе с её версией модели
        assert stored.action == PREDICTION.action
        assert stored.model_version == "7"


async def test_concurrent_deliveries_write_one_result(sessions):
    message = make_message()
    async with sessions() as session:
        await TaskRepository(session).ensure_task(message)
        await session.commit()

    async def deliver(action: int) -> bool:
        async with sessions() as session:
            repository = TaskRepository(session)
            saved = await repository.save_result(
                message.task_id, Prediction(action=action, inference_ms=1.0, delay_ms=0.0), "7"
            )
            await session.commit()
            return saved

    outcomes = await asyncio.gather(deliver(1), deliver(2))

    assert sorted(outcomes) == [False, True]
    assert await count_results(sessions, message.task_id) == 1


async def test_failure_keeps_the_error_and_attempts(sessions):
    message = make_message()
    async with sessions() as session:
        repository = TaskRepository(session)
        await repository.ensure_task(message)
        await repository.mark_failed(message.task_id, "wall", attempts=3)
        await session.commit()

    async with sessions() as session:
        task = await session.get(InferenceTask, message.task_id)
        assert task.status is TaskStatus.FAILED
        assert task.last_error == "wall"
        assert task.attempts == 3


async def test_api_flow_creates_and_removes_a_pending_task(sessions):
    message = make_message()
    async with sessions() as session:
        repository = TaskRepository(session)
        await repository.create_pending(message)
        await session.commit()
        assert await repository.get_status(message.task_id) is TaskStatus.PENDING

        await repository.delete_pending(message.task_id)
        await session.commit()
        assert await repository.get_status(message.task_id) is None
