import asyncio
import random
import time
from uuid import uuid4

import httpx
import pytest
from asgi_lifespan import LifespanManager
from faststream.rabbit import RabbitBroker
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import create_async_engine
from testcontainers.community.rabbitmq import RabbitMqContainer
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

from hexrl_platform.core.config import get_settings
from hexrl_platform.db.models import InferenceTask
from hexrl_platform.db.schema import create_schema
from hexrl_platform.main import create_app
from hexrl_platform.task_pipeline.batch_source import Backoff, SourceStats, run_batch, task_id_for
from hexrl_platform.task_pipeline.config import get_pipeline_settings
from hexrl_platform.task_pipeline.errors import QueueOverloadedError
from hexrl_platform.task_pipeline.publisher import TaskPublisher
from hexrl_platform.task_pipeline.schemas import HexState, TaskMessage
from hexrl_platform.task_pipeline.topology import build_topology, declare_topology

pytestmark = pytest.mark.integration
STATE = HexState(q=0, r=0, goal_q=2, goal_r=-1, steps_remaining=20)


@pytest.fixture(scope="module")
def rabbit():
    with RabbitMqContainer(
        "rabbitmq:4.3.6-management-alpine", username="overflow-test", password="overflow-test"
    ).with_exposed_ports(15672) as container:
        yield container


def connection_url(container):
    return (
        f"amqp://overflow-test:overflow-test@{container.get_container_host_ip()}:"
        f"{container.get_exposed_port(5672)}/"
    )


async def test_real_queue_rejects_without_discarding_accepted_messages(rabbit):
    async with RabbitBroker(connection_url(rabbit)) as broker:
        topology = build_topology(f"overflow.{uuid4().hex}", 5, 100)
        await declare_topology(broker, topology)
        publisher = TaskPublisher(broker, topology, 1)
        accepted = []
        for index in range(20):
            message = TaskMessage(task_id=uuid4(), payload=STATE)
            if index < 5:
                await publisher.publish(message)
                accepted.append(message.task_id)
            else:
                with pytest.raises(QueueOverloadedError):
                    await publisher.publish(message)
        async with httpx.AsyncClient(auth=("overflow-test", "overflow-test")) as client:
            url = (
                f"http://{rabbit.get_container_host_ip()}:{rabbit.get_exposed_port(15672)}"
                f"/api/queues/%2F/{topology.tasks_queue.name}"
            )
            async with asyncio.timeout(15):
                while True:
                    response = await client.get(url)
                    response.raise_for_status()
                    if response.json().get("messages_ready") == 5:
                        break
                    await asyncio.sleep(0.2)
        queue = await broker.declare_queue(topology.tasks_queue)
        received = []
        for _ in range(5):
            message = await queue.get(timeout=2)
            received.append(TaskMessage.model_validate_json(message.body).task_id)
            await message.ack()
        assert received == accepted


async def test_requeue_preserves_inflight_messages_above_the_publish_limit(rabbit):
    async with RabbitBroker(connection_url(rabbit)) as broker:
        topology = build_topology(f"requeue-limit.{uuid4().hex}", 5, 100)
        await declare_topology(broker, topology)
        publisher = TaskPublisher(broker, topology, 1)
        queue = await broker.declare_queue(topology.tasks_queue)
        expected = set()
        for _ in range(2):
            task = TaskMessage(task_id=uuid4(), payload=STATE)
            expected.add(task.task_id)
            await publisher.publish(task)
        inflight = [await queue.get(timeout=2) for _ in range(2)]
        for _ in range(5):
            task = TaskMessage(task_id=uuid4(), payload=STATE)
            expected.add(task.task_id)
            await publisher.publish(task)
        for message in inflight:
            await message.nack(requeue=True)
        await queue.declare()
        assert queue.declaration_result.message_count == 7
        with pytest.raises(QueueOverloadedError):
            await publisher.publish(TaskMessage(task_id=uuid4(), payload=STATE))
        recovered = set()
        for _ in range(7):
            message = await queue.get(timeout=2)
            recovered.add(TaskMessage.model_validate_json(message.body).task_id)
            await message.ack()
        assert recovered == expected


async def test_batch_pauses_on_the_same_record_then_resumes_without_loss(rabbit):
    async with RabbitBroker(connection_url(rabbit)) as broker:
        topology = build_topology(f"resume.{uuid4().hex}", 5, 100)
        await declare_topology(broker, topology)
        batch_id = uuid4().hex
        stats = SourceStats()
        read = []
        rejected = asyncio.Event()

        async def backoff_sleep(delay):
            rejected.set()
            await asyncio.sleep(delay)

        async def lines():
            for number in range(1, 21):
                read.append(number)
                yield number, STATE.model_dump_json()

        source = asyncio.create_task(
            run_batch(
                TaskPublisher(broker, topology, 1),
                lines(),
                batch_id,
                stats,
                Backoff(0.01, 0.05, random.Random(42)),
                sleep=backoff_sleep,
            )
        )
        try:
            await asyncio.wait_for(rejected.wait(), timeout=5)
            assert stats.published_ok == 5
            assert read == list(range(1, 7))
            queue = await broker.declare_queue(topology.tasks_queue)
            received = []
            async with queue.iterator() as messages:
                async with asyncio.timeout(15):
                    async for message in messages:
                        received.append(TaskMessage.model_validate_json(message.body).task_id)
                        await message.ack()
                        if len(received) == 20:
                            break
            await asyncio.wait_for(source, 5)
            assert received == [task_id_for(batch_id, number) for number in range(1, 21)]
            assert stats.published_ok == 20
            assert stats.wait_s_total > 0
        finally:
            source.cancel()
            await asyncio.gather(source, return_exceptions=True)


@pytest.fixture(scope="module")
def postgres():
    container = (
        DockerContainer("postgres:16-alpine")
        .with_env("POSTGRES_USER", "api_test")
        .with_env("POSTGRES_PASSWORD", "api_test")
        .with_env("POSTGRES_DB", "api_test")
        .with_exposed_ports(5432)
        .waiting_for(
            LogMessageWaitStrategy("database system is ready to accept connections", times=2)
        )
    )
    with container:
        yield container


async def test_postgres_schema_initialization_is_safe_for_concurrent_replicas(postgres):
    engine = create_async_engine(
        f"postgresql+asyncpg://api_test:api_test@{postgres.get_container_host_ip()}:"
        f"{postgres.get_exposed_port(5432)}/api_test"
    )
    try:
        await asyncio.gather(*(create_schema(engine) for _ in range(4)))
    finally:
        await engine.dispose()


async def test_api_lifespan_real_postgres_and_rabbit_overload(rabbit, postgres, monkeypatch):
    for key, value in {
        "POSTGRES_USER": "api_test",
        "POSTGRES_PASSWORD": "api_test",
        "POSTGRES_DB": "api_test",
        "POSTGRES_HOST": postgres.get_container_host_ip(),
        "POSTGRES_PORT": str(postgres.get_exposed_port(5432)),
        "RABBITMQ_USER": "overflow-test",
        "RABBITMQ_PASSWORD": "overflow-test",
        "RABBITMQ_HOST": rabbit.get_container_host_ip(),
        "RABBITMQ_PORT": str(rabbit.get_exposed_port(5672)),
        "QUEUE_PREFIX": f"api-integration.{uuid4().hex}",
        "QUEUE_MAX_LENGTH": "5",
    }.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    get_pipeline_settings.cache_clear()
    try:
        app = create_app()
        async with LifespanManager(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                accepted = []
                for _ in range(5):
                    response = await client.post("/api/v1/tasks", json=STATE.model_dump())
                    assert response.status_code == 202
                    accepted.append(response.json()["status_url"])
                for _ in range(3):
                    started = time.perf_counter()
                    response = await client.post("/api/v1/tasks", json=STATE.model_dump())
                    assert time.perf_counter() - started < 1
                    assert response.status_code == 503
                    assert response.headers["Retry-After"] == "1"
                async with app.state.session_factory() as session:
                    assert (
                        await session.scalar(select(func.count()).select_from(InferenceTask)) == 5
                    )
                    pending = (
                        select(func.count())
                        .select_from(InferenceTask)
                        .where(InferenceTask.status == "PENDING")
                    )
                    assert await session.scalar(pending) == 0
                for url in accepted:
                    assert (await client.get(url)).json()["status"] == "ACCEPTED"
    finally:
        get_settings.cache_clear()
        get_pipeline_settings.cache_clear()
