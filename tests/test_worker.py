from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pandas as pd
import pytest
from faststream import TestApp
from faststream.rabbit import TestRabbitBroker
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from hexrl_platform.db.models import InferenceResult, InferenceTask
from hexrl_platform.inference.model_loader import LoadedModel
from hexrl_platform.task_pipeline.config import get_pipeline_settings
from hexrl_platform.task_pipeline.schemas import HexState, TaskMessage, TaskStatus
from hexrl_platform.task_pipeline.topology import build_topology
from hexrl_platform.task_pipeline.worker import create_worker_app

MODEL_VERSION = "7"
STATE = HexState(q=0, r=0, goal_q=2, goal_r=-1, steps_remaining=20)


class FakeModel:
    """Подменяет pyfunc-модель: отдаёт действие или падает нужной ошибкой."""

    def __init__(self, action: int = 2, error: Exception | None = None) -> None:
        self.action = action
        self.error = error
        self.calls = 0

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return pd.DataFrame({"action": [self.action] * len(data)})


class CountingLoader:
    """Считает, сколько раз воркер ходил за моделью."""

    def __init__(self, model: FakeModel, version: str = MODEL_VERSION) -> None:
        self.model = model
        self.version = version
        self.calls = 0

    def __call__(self, settings: Any) -> LoadedModel:
        self.calls += 1
        return LoadedModel(
            model=self.model,
            name="hex-navigation-default",
            alias="champion",
            version=self.version,
            run_id="run-1",
            max_steps=50,
        )


@pytest.fixture(autouse=True)
def fast_pipeline(monkeypatch):
    # без задержки тесты не ждут искусственной нагрузки
    monkeypatch.setenv("INFERENCE_DELAY_S", "0")
    monkeypatch.setenv("MAX_DELIVERIES", "3")
    get_pipeline_settings.cache_clear()
    yield
    get_pipeline_settings.cache_clear()


@pytest.fixture
def database(tmp_path):
    return f"sqlite+aiosqlite:///{(tmp_path / 'worker.db').as_posix()}"


def make_message(**overrides: Any) -> TaskMessage:
    fields: dict[str, Any] = {
        "task_id": uuid4(),
        "batch_id": "batch-1",
        "payload": STATE,
        "submitted_at": datetime.now(UTC),
    }
    return TaskMessage(**(fields | overrides))


def build_app(database: str, model: FakeModel):
    loader = CountingLoader(model)
    app = create_worker_app(loader=loader, engine_factory=lambda: create_async_engine(database))
    return app, loader


def topology_for():
    settings = get_pipeline_settings()
    return build_topology(settings.queue_prefix, settings.queue_max_length, settings.retry_delay_ms)


async def publish(app, body: Any, headers: dict[str, Any] | None = None) -> None:
    topology = topology_for()
    await app.broker.publish(
        body,
        exchange=topology.tasks_exchange,
        routing_key=topology.tasks_routing_key,
        headers=headers,
    )


def attach_dlq(app) -> list[Any]:
    """Подписчик на DLQ: в тест-брокере публикация без получателя падает."""
    topology = topology_for()
    received: list[Any] = []

    @app.broker.subscriber(topology.dlq, topology.dlx)
    async def collect(body: Any) -> None:
        received.append(body)

    return received


def deaths(queue_prefix: str, count: int) -> dict[str, Any]:
    """Заголовок x-death в том виде, в каком его проставляет RabbitMQ."""
    return {"x-death": [{"queue": f"{queue_prefix}.tasks", "count": count}]}


async def read_task(database: str, task_id):
    engine = create_async_engine(database)
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            task = await session.get(InferenceTask, task_id)
            results = await session.scalar(
                select(func.count())
                .select_from(InferenceResult)
                .where(InferenceResult.task_id == task_id)
            )
            stored = await session.get(InferenceResult, task_id)
            return task, results, stored
    finally:
        await engine.dispose()


async def test_successful_task_is_stored_with_the_model_version(database):
    app, loader = build_app(database, FakeModel(action=4))
    message = make_message()

    async with TestRabbitBroker(app.broker), TestApp(app):
        await publish(app, message)

    task, results, stored = await read_task(database, message.task_id)
    assert task.status is TaskStatus.COMPLETED
    assert results == 1
    assert stored.action == 4
    assert stored.model_version == MODEL_VERSION
    assert loader.calls == 1


async def test_model_is_loaded_once_for_many_messages(database):
    model = FakeModel()
    app, loader = build_app(database, model)

    async with TestRabbitBroker(app.broker), TestApp(app):
        for _ in range(3):
            await publish(app, make_message())

    assert loader.calls == 1
    assert model.calls == 3


async def test_repeated_delivery_keeps_a_single_result(database):
    app, _ = build_app(database, FakeModel(action=4))
    message = make_message()

    async with TestRabbitBroker(app.broker), TestApp(app):
        await publish(app, message)
        await publish(app, message)

    task, results, stored = await read_task(database, message.task_id)
    assert results == 1
    assert stored.action == 4
    assert task.status is TaskStatus.COMPLETED


async def test_invalid_message_goes_to_the_dlq_and_worker_survives(database):
    model = FakeModel()
    app, _ = build_app(database, model)
    dead = attach_dlq(app)

    async with TestRabbitBroker(app.broker), TestApp(app):
        await publish(app, {"not": "a task"})
        # воркер жив и обрабатывает следующее сообщение
        good = make_message()
        await publish(app, good)

    assert len(dead) == 1
    task, results, _ = await read_task(database, good.task_id)
    assert task.status is TaskStatus.COMPLETED
    assert results == 1
    assert model.calls == 1


async def test_domain_error_fails_the_task_without_retries(database):
    error = ValueError("Position and goal must be walkable cells of the model map")
    app, _ = build_app(database, FakeModel(error=error))
    dead = attach_dlq(app)
    message = make_message()

    async with TestRabbitBroker(app.broker), TestApp(app):
        await publish(app, message)

    # постоянная ошибка не станет другой от повтора: сразу в DLQ
    assert len(dead) == 1
    task, results, _ = await read_task(database, message.task_id)
    assert task.status is TaskStatus.FAILED
    assert "walkable" in task.last_error
    assert results == 0


async def test_last_attempt_goes_to_the_dlq(database):
    app, _ = build_app(database, FakeModel(error=RuntimeError("model is busy")))
    dead = attach_dlq(app)
    message = make_message()
    settings = get_pipeline_settings()

    async with TestRabbitBroker(app.broker), TestApp(app):
        # сообщение уже возвращалось max_deliveries - 1 раз, эта попытка последняя
        await publish(
            app, message, headers=deaths(settings.queue_prefix, settings.max_deliveries - 1)
        )

    assert len(dead) == 1
    task, results, _ = await read_task(database, message.task_id)
    assert task.status is TaskStatus.FAILED
    assert task.attempts == settings.max_deliveries
    assert results == 0


async def test_transient_error_is_retried_before_failing(database):
    app, _ = build_app(database, FakeModel(error=RuntimeError("model is busy")))
    message = make_message()

    async with TestRabbitBroker(app.broker), TestApp(app):
        await publish(app, message)

    task, results, _ = await read_task(database, message.task_id)
    # первая попытка не финальна: сообщение ушло в retry-очередь, статус не FAILED
    assert task.status is TaskStatus.ACCEPTED
    assert task.attempts == 1
    assert "busy" in task.last_error
    assert results == 0
