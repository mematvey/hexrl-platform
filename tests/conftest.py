import logging
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
from opentelemetry import trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from hexrl_platform import main as main_module
from hexrl_platform.core.metrics import histogram_views
from hexrl_platform.core.telemetry import Telemetry
from hexrl_platform.db.session import get_session
from hexrl_platform.main import create_app


def pytest_sessionfinish(session, exitstatus):
    # Captured stdout закрывается после сессии; atexit-логи PyTorch не должны
    # обращаться к оставшемуся handler с уже закрытым потоком pytest.
    root = logging.getLogger()
    for handler in root.handlers[:]:
        if handler.name == "hexrl-stdout":
            root.removeHandler(handler)
            handler.close()


@pytest.fixture
def traces(monkeypatch):
    exporters = []
    providers = []

    def create(service, **kwargs):
        provider = TracerProvider(resource=Resource({"service.name": service}))
        exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        providers.append(provider)
        exporters.append(exporter)
        return Telemetry(provider)

    monkeypatch.setattr(Telemetry, "from_env", staticmethod(create))
    yield lambda: tuple(span for exporter in exporters for span in exporter.get_finished_spans())
    for provider in providers:
        provider.shutdown()


@pytest.fixture
def metrics(monkeypatch):
    readers = {}
    telemetries = []

    def create(service, **kwargs):
        reader = InMemoryMetricReader()
        readers.setdefault(service, []).append(reader)
        provider = MeterProvider(
            metric_readers=[reader],
            resource=Resource({"service.name": service}),
            views=histogram_views(kwargs.get("inference_delay_s", 0.5)),
            shutdown_on_exit=False,
        )
        telemetry = Telemetry(trace.NoOpTracerProvider(), provider, role=service.rsplit("-", 1)[-1])
        telemetries.append(telemetry)
        return telemetry

    def collect(service):
        result = {}
        for reader in readers[service]:
            data = reader.get_metrics_data()
            if data is not None:
                for resource in data.resource_metrics:
                    for scope in resource.scope_metrics:
                        for metric in scope.metrics:
                            result.setdefault(metric.name, []).extend(metric.data.data_points)
        return result

    monkeypatch.setattr(Telemetry, "from_env", staticmethod(create))
    yield collect
    for telemetry in telemetries:
        telemetry.shutdown()


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
    monkeypatch.setattr(main_module, "RabbitBroker", lambda *a, **k: AsyncMock())
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
