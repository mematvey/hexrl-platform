import asyncio
import threading
from urllib.parse import urlsplit
from uuid import uuid4

import pandas as pd
import pytest
from faststream import TestApp
from faststream.rabbit import RabbitBroker
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from testcontainers.community.rabbitmq import RabbitMqContainer

from hexrl_platform.db.models import InferenceTask
from hexrl_platform.inference.model_loader import LoadedModel
from hexrl_platform.inference.predict import PredictModel
from hexrl_platform.task_pipeline import worker as worker_module
from hexrl_platform.task_pipeline.config import get_pipeline_settings
from hexrl_platform.task_pipeline.errors import QueueOverloadedError
from hexrl_platform.task_pipeline.publisher import TaskPublisher, publish_confirmed
from hexrl_platform.task_pipeline.repository import TaskRepository
from hexrl_platform.task_pipeline.retries import ATTEMPT_HEADER, EXHAUSTED_REASON
from hexrl_platform.task_pipeline.schemas import HexState, TaskMessage, TaskStatus
from hexrl_platform.task_pipeline.topology import build_topology, declare_topology
from hexrl_platform.task_pipeline.worker import DEAD_REASON_HEADER, create_worker_app

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def rabbitmq_url():
    container = RabbitMqContainer(
        "rabbitmq:4.3.6-management-alpine", username="retry-test", password="retry-test"
    ).with_env(
        "RABBITMQ_SERVER_ADDITIONAL_ERL_ARGS",
        "-rabbit dead_letter_worker_publisher_confirm_timeout 1000",
    )
    with container:
        yield (
            f"amqp://retry-test:retry-test@{container.get_container_host_ip()}:"
            f"{container.get_exposed_port(5672)}/"
        )


async def test_expired_retry_survives_a_full_tasks_queue(rabbitmq_url):
    broker = RabbitBroker(rabbitmq_url)
    await broker.start()
    try:
        topology = build_topology(f"retry-test.{uuid4().hex}", 1, retry_delay_ms=100)
        await declare_topology(broker, topology)
        publisher = TaskPublisher(broker, topology, confirm_timeout_s=1.0)
        payload = HexState(q=0, r=0, goal_q=2, goal_r=-1, steps_remaining=20)
        blocker = TaskMessage(task_id=uuid4(), payload=payload)
        retried = TaskMessage(task_id=uuid4(), payload=payload)
        await publisher.publish(blocker)
        await publish_confirmed(
            broker,
            retried,
            exchange=topology.dlx,
            routing_key=topology.retry_routing_key,
            timeout_s=1.0,
        )

        # Место не освобождаем до истечения TTL и первой попытки пересылки.
        await asyncio.sleep(1.5)
        queue = await broker.declare_queue(topology.tasks_queue)
        first = await queue.get(timeout=2.0)
        assert TaskMessage.model_validate_json(first.body).task_id == blocker.task_id
        await first.ack()

        async with queue.iterator() as messages:
            recovered = await asyncio.wait_for(anext(messages), timeout=8.0)
            assert TaskMessage.model_validate_json(recovered.body).task_id == retried.task_id
            await recovered.ack()
    finally:
        await broker.stop()


@pytest.mark.parametrize("retry_publish_fails", [False, True])
async def test_worker_retries_three_times_then_confirms_the_dlq(
    rabbitmq_url, tmp_path, monkeypatch, retry_publish_fails
):
    class FailingModel(PredictModel):
        def __init__(self):
            self.calls = 0

        def predict(self, data: pd.DataFrame) -> pd.DataFrame:
            self.calls += 1
            raise RuntimeError("temporary test failure")

    address = urlsplit(rabbitmq_url)
    prefix = f"worker-retry-test.{uuid4().hex}"
    for key, value in {
        "RABBITMQ_USER": "retry-test",
        "RABBITMQ_PASSWORD": "retry-test",
        "RABBITMQ_HOST": address.hostname,
        "RABBITMQ_PORT": str(address.port),
        "QUEUE_PREFIX": prefix,
        "INFERENCE_DELAY_S": "0",
        "MAX_DELIVERIES": "3",
        "RETRY_DELAY_MS": "100",
    }.items():
        monkeypatch.setenv(key, value)
    get_pipeline_settings.cache_clear()
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'tasks.db').as_posix()}")
    model = FailingModel()
    loaded = LoadedModel(model, "test-policy", "champion", "1", "test-run", 20)
    app = create_worker_app(loader=lambda _: loaded, engine_factory=lambda: engine)
    assert isinstance(app.broker, RabbitBroker)
    broker = app.broker
    topology = build_topology(prefix, 20, 100)
    if retry_publish_fails:
        original = worker_module.publish_confirmed

        async def reject_retry(*args, **kwargs):
            if kwargs["routing_key"] == topology.retry_routing_key:
                raise QueueOverloadedError("retry publish unavailable")
            await original(*args, **kwargs)

        monkeypatch.setattr(worker_module, "publish_confirmed", reject_retry)
        monkeypatch.setattr(worker_module, "CRASH_PAUSE_S", 0.01)
    task = TaskMessage(
        task_id=uuid4(), payload=HexState(q=0, r=0, goal_q=2, goal_r=-1, steps_remaining=20)
    )
    try:
        async with TestApp(app):
            await TaskPublisher(broker, topology, 1.0).publish(task)

            async def wait_for_failure():
                sessions = async_sessionmaker(engine, expire_on_commit=False)
                while True:
                    async with sessions() as session:
                        status = await TaskRepository(session).get_status(task.task_id)
                    if status is TaskStatus.FAILED:
                        return
                    await asyncio.sleep(0.05)

            await asyncio.wait_for(wait_for_failure(), timeout=10.0)
            queue = await broker.declare_queue(topology.dlq)
            async with queue.iterator() as messages:
                dead = await asyncio.wait_for(anext(messages), timeout=5.0)
                assert TaskMessage.model_validate_json(dead.body).task_id == task.task_id
                assert dead.headers[ATTEMPT_HEADER] == 3
                assert dead.headers[DEAD_REASON_HEADER] == EXHAUSTED_REASON
                assert model.calls == 3
                await dead.ack()
    finally:
        await engine.dispose()
        get_pipeline_settings.cache_clear()


async def test_real_worker_never_buffers_more_than_its_prefetch(
    rabbitmq_url, tmp_path, monkeypatch
):
    entered = threading.Event()
    release = threading.Event()

    class BlockingModel(PredictModel):
        def predict(self, data: pd.DataFrame) -> pd.DataFrame:
            entered.set()
            if not release.wait(timeout=15):
                raise TimeoutError("test did not release inference")
            return pd.DataFrame({"action": [2] * len(data)})

    address = urlsplit(rabbitmq_url)
    prefix = f"prefetch-test.{uuid4().hex}"
    for key, value in {
        "RABBITMQ_USER": "retry-test",
        "RABBITMQ_PASSWORD": "retry-test",
        "RABBITMQ_HOST": address.hostname,
        "RABBITMQ_PORT": str(address.port),
        "QUEUE_PREFIX": prefix,
        "INFERENCE_DELAY_S": "0",
        "PREFETCH_COUNT": "2",
        "WORKER_CONCURRENCY": "2",
    }.items():
        monkeypatch.setenv(key, value)
    get_pipeline_settings.cache_clear()
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'prefetch.db').as_posix()}")
    model = LoadedModel(BlockingModel(), "test-policy", "champion", "1", "test-run", 20)
    app = create_worker_app(loader=lambda _: model, engine_factory=lambda: engine)
    assert isinstance(app.broker, RabbitBroker)
    topology = build_topology(prefix, 20, 5000)
    try:
        async with TestApp(app):
            publisher = TaskPublisher(app.broker, topology, 1)
            for _ in range(10):
                await publisher.publish(
                    TaskMessage(
                        task_id=uuid4(),
                        payload=HexState(q=0, r=0, goal_q=2, goal_r=-1, steps_remaining=20),
                    )
                )
            assert await asyncio.to_thread(entered.wait, 5)
            # Даём брокеру время доставить лишние сообщения, если QoS сломан.
            await asyncio.sleep(0.3)
            queue = await app.broker.declare_queue(topology.tasks_queue)
            await queue.declare()
            assert queue.declaration_result.message_count == 8
            sessions = async_sessionmaker(engine)
            async with sessions() as session:
                assert await session.scalar(select(func.count()).select_from(InferenceTask)) == 2
            release.set()
    finally:
        release.set()
        await engine.dispose()
        get_pipeline_settings.cache_clear()
