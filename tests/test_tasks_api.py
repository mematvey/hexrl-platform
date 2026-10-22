import asyncio
import time
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from hexrl_platform.api.v1.tasks import get_publisher
from hexrl_platform.core.config import get_settings
from hexrl_platform.db.models import InferenceTask
from hexrl_platform.db.schema import create_schema
from hexrl_platform.db.session import get_session
from hexrl_platform.main import create_app
from hexrl_platform.service.tasks import retry_after
from hexrl_platform.task_pipeline.config import PipelineSettings, get_pipeline_settings
from hexrl_platform.task_pipeline.errors import QueueOverloadedError, TopologyError
from hexrl_platform.task_pipeline.repository import TaskRepository
from hexrl_platform.task_pipeline.schemas import Prediction, TaskStatus

PAYLOAD = {"q": 0, "r": 0, "goal_q": 2, "goal_r": -1, "steps_remaining": 20}


class Publisher:
    def __init__(self):
        self.calls = []
        self.error = None
        self.delay = 0
        self.on_publish = None
        self.entered = asyncio.Event()

    async def publish(self, message):
        self.calls.append(message)
        self.entered.set()
        await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        if self.on_publish is not None:
            await self.on_publish(message)


@pytest.fixture
async def api(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'api.db').as_posix()}")
    await create_schema(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    publisher = Publisher()
    app = create_app()

    async def session():
        async with sessions() as value:
            yield value

    settings = get_settings().model_copy(update={"task_submission_timeout_s": 0.1})
    pipeline = PipelineSettings(_env_file=None, rabbitmq_user="test", rabbitmq_password="test")
    app.dependency_overrides[get_session] = session
    app.dependency_overrides[get_publisher] = lambda: publisher
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_pipeline_settings] = lambda: pipeline
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client, publisher, sessions
    finally:
        await engine.dispose()


async def test_task_is_accepted_and_can_be_read(api):
    client, publisher, sessions = api
    response = await client.post("/api/v1/tasks", json=PAYLOAD)
    assert response.status_code == 202
    body = response.json()
    task_id = UUID(body["task_id"])
    assert body["status"] == "ACCEPTED"
    assert publisher.calls[0].task_id == task_id
    detail = await client.get(body["status_url"])
    assert detail.status_code == 200
    assert detail.json()["status"] == "ACCEPTED"
    assert detail.json()["result"] is None
    async with sessions() as session:
        assert await TaskRepository(session).get_status(task_id) is TaskStatus.ACCEPTED


@pytest.mark.parametrize("error", [QueueOverloadedError(), TimeoutError(), TopologyError()])
async def test_rejected_task_returns_retry_after_and_leaves_no_pending(api, error):
    client, publisher, sessions = api
    publisher.error = error
    response = await client.post("/api/v1/tasks", json=PAYLOAD)
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"
    expected = "pipeline_unavailable" if isinstance(error, TopologyError) else "queue_overloaded"
    assert response.json()["error"] == expected
    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(InferenceTask)) == 0


async def test_slow_publisher_is_bounded_and_pending_is_removed(api):
    client, publisher, sessions = api
    publisher.delay = 5
    started = time.perf_counter()
    response = await client.post("/api/v1/tasks", json=PAYLOAD)
    assert time.perf_counter() - started < 0.8
    assert response.status_code == 503
    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(InferenceTask)) == 0


async def test_fast_worker_result_is_not_reverted_to_accepted(api):
    client, publisher, sessions = api

    async def complete(message):
        async with sessions() as session:
            await TaskRepository(session).save_result(
                message.task_id, Prediction(3, 1.0, 500.0), "7", model_name="navigation"
            )
            await session.commit()

    publisher.on_publish = complete
    response = await client.post("/api/v1/tasks", json=PAYLOAD)
    detail = (await client.get(response.json()["status_url"])).json()
    assert detail["status"] == "COMPLETED"
    assert detail["result"]["action"] == 3
    assert detail["result"]["model_name"] == "navigation"
    assert detail["result"]["model_version"] == "7"
    assert detail["completed_at"] is not None


async def test_unknown_and_malformed_task_ids(api):
    client, _, _ = api
    assert (await client.get(f"/api/v1/tasks/{uuid4()}")).status_code == 404
    assert (await client.get("/api/v1/tasks/invalid")).status_code == 422


async def test_invalid_request_never_reaches_the_publisher(api):
    client, publisher, _ = api
    assert (
        await client.post("/api/v1/tasks", json={**PAYLOAD, "steps_remaining": 0})
    ).status_code == 422
    assert not publisher.calls


async def test_domain_validation_is_left_to_the_worker(api):
    client, _, _ = api
    response = await client.post("/api/v1/tasks", json={**PAYLOAD, "q": 999})
    assert response.status_code == 202


async def test_cancelled_request_cleans_up_committed_pending(api):
    client, publisher, sessions = api
    publisher.delay = 5
    request = asyncio.create_task(client.post("/api/v1/tasks", json=PAYLOAD))
    await asyncio.wait_for(publisher.entered.wait(), timeout=1)
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(InferenceTask)) == 0


@pytest.mark.parametrize("delay,workers,concurrency,expected", [(0.5, 1, 2, 1), (10, 3, 2, 2)])
def test_retry_after_is_derived_from_estimated_capacity(delay, workers, concurrency, expected):
    settings = get_settings().model_copy(update={"estimated_worker_count": workers})
    pipeline = PipelineSettings(
        _env_file=None,
        rabbitmq_user="test",
        rabbitmq_password="test",
        inference_delay_s=delay,
        worker_concurrency=concurrency,
    )
    assert retry_after(settings, pipeline) == expected
