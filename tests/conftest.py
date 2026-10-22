import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_DB", "test")
os.environ.setdefault("RABBITMQ_USER", "test")
os.environ.setdefault("RABBITMQ_PASSWORD", "test")

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from hexrl_platform import main as main_module
from hexrl_platform.db.session import get_session
from hexrl_platform.main import create_app


class FakeResult:
    def __init__(self, value: str) -> None:
        self._value = value

    def scalar_one(self) -> str:
        return self._value


class FakeSession:
    """Подменяет AsyncSession: либо отвечает версией, либо падает/зависает."""

    def __init__(self, value: str = "PostgreSQL 16.0", error: Exception | None = None) -> None:
        self.value = value
        self.error = error

    async def execute(self, *_args: object, **_kwargs: object) -> FakeResult:
        if self.error:
            raise self.error
        return FakeResult(self.value)


@pytest.fixture
def app(monkeypatch) -> FastAPI:
    # Lifespan проверяем без внешних сервисов; реальные подключения покрыты integration.
    monkeypatch.setattr(main_module, "create_schema", AsyncMock())
    monkeypatch.setattr(main_module, "cleanup_pending", AsyncMock(return_value=0))
    monkeypatch.setattr(main_module, "declare_topology", AsyncMock())
    monkeypatch.setattr(main_module, "RabbitBroker", lambda _: AsyncMock())
    return create_app()


@pytest.fixture
def session() -> FakeSession:
    return FakeSession()


@pytest.fixture
async def client(app: FastAPI, session: FakeSession) -> AsyncIterator[AsyncClient]:
    async def override() -> AsyncIterator[FakeSession]:
        yield session

    app.dependency_overrides[get_session] = override
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
