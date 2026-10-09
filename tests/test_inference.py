import asyncio
import time

import pandas as pd
import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from hexrl_platform.inference.config import InferenceSettings
from hexrl_platform.inference.main import create_app
from hexrl_platform.inference.model_loader import LoadedModel

MAX_STEPS = 50
VERSION = "7"


class FakeModel:
    """Подменяет pyfunc-модель: отдаёт действие, падает или тормозит."""

    def __init__(self, action=0, error=None, delay=0.0):
        self.action = action
        self.error = error
        self.delay = delay

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        return pd.DataFrame({"action": [self.action] * len(frame)})


class CountingLoader:
    """Считает, сколько раз приложение ходило за моделью."""

    def __init__(self, model):
        self.model = model
        self.calls = 0

    def __call__(self, settings: InferenceSettings) -> LoadedModel:
        self.calls += 1
        return LoadedModel(
            model=self.model,
            name="hex-navigation-default",
            alias="champion",
            version=VERSION,
            run_id="run-abc",
            max_steps=MAX_STEPS,
        )


def valid_state(**overrides):
    state = {"q": 0, "r": 0, "goal_q": 2, "goal_r": -1, "steps_remaining": 20}
    return state | overrides


async def make_client(loader):
    app = create_app(loader=loader)
    manager = LifespanManager(app)
    await manager.__aenter__()
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    return client, manager


@pytest.fixture
async def client():
    loader = CountingLoader(FakeModel(action=3))
    client, manager = await make_client(loader)
    async with client:
        yield client
    await manager.__aexit__(None, None, None)


async def test_healthz_does_not_touch_the_model(client):
    response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_process_returns_action_direction_and_model_version(client):
    response = await client.post("/process", json=valid_state())

    assert response.status_code == 200
    body = response.json()
    assert body["action"] == 3
    # DIRECTIONS[3] == HexCoord(-1, 0)
    assert body["direction"] == {"dq": -1, "dr": 0}
    assert body["model"] == {
        "name": "hex-navigation-default",
        "version": VERSION,
        "alias": "champion",
        "run_id": "run-abc",
        "max_steps": MAX_STEPS,
    }


async def test_model_endpoint_reports_the_loaded_version(client):
    response = await client.get("/model")

    assert response.status_code == 200
    assert response.json()["version"] == VERSION


@pytest.mark.parametrize(
    "payload",
    [
        valid_state(steps_remaining=0),
        valid_state(steps_remaining=-1),
        {"q": 0, "r": 0, "goal_q": 2},
        valid_state(q="сюда"),
    ],
    ids=["zero-steps", "negative-steps", "missing-field", "not-an-int"],
)
async def test_schema_rejects_broken_payloads(client, payload):
    response = await client.post("/process", json=payload)

    assert response.status_code == 422


async def test_model_error_about_a_wall_becomes_422():
    error = ValueError("Position and goal must be walkable cells of the model map")
    loader = CountingLoader(FakeModel(error=error))
    client, manager = await make_client(loader)

    async with client:
        response = await client.post("/process", json=valid_state())

    await manager.__aexit__(None, None, None)
    assert response.status_code == 422
    assert "walkable" in response.json()["detail"]


async def test_model_is_loaded_once_for_many_requests():
    loader = CountingLoader(FakeModel())
    client, manager = await make_client(loader)

    async with client:
        for _ in range(3):
            assert (await client.post("/process", json=valid_state())).status_code == 200

    await manager.__aexit__(None, None, None)
    assert loader.calls == 1


async def test_concurrent_requests_do_not_block_each_other():
    """Инференс уходит в поток: пять запросов по 0.2с не должны сложиться в секунду."""
    delay = 0.2
    requests = 5
    loader = CountingLoader(FakeModel(delay=delay))
    client, manager = await make_client(loader)

    async with client:
        started = time.perf_counter()
        responses = await asyncio.gather(
            *(client.post("/process", json=valid_state()) for _ in range(requests))
        )
        elapsed = time.perf_counter() - started

    await manager.__aexit__(None, None, None)
    assert all(response.status_code == 200 for response in responses)
    assert elapsed < delay * requests * 0.6
