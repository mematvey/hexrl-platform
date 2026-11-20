import asyncio
import random
import threading
from argparse import Namespace
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.responses import Response
from httpx import ASGITransport, AsyncClient
from opentelemetry.metrics import NoOpMeterProvider
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest

from hexrl_platform.core.metrics import API_BUCKETS, PREDICT_BUCKETS, histogram_views
from hexrl_platform.core.middleware import add_request_logging
from hexrl_platform.core.telemetry import Telemetry
from hexrl_platform.task_pipeline import batch_source
from hexrl_platform.task_pipeline.batch_source import Backoff, SourceStats, run_batch
from hexrl_platform.task_pipeline.errors import QueueOverloadedError, TopologyError
from hexrl_platform.task_pipeline.runner import InferenceRunner
from hexrl_platform.task_pipeline.schemas import HexState

STATE = HexState(q=0, r=0, goal_q=2, goal_r=-1, steps_remaining=20)


def test_api_exports_zero_refusals_before_the_first_503(metrics):
    Telemetry.from_env("hexrl-api")
    requests = {
        point.attributes["status_code"]: point.value
        for point in metrics("hexrl-api")["hexrl.api.requests"]
        if point.attributes["route"] == "/api/v1/tasks"
    }
    assert requests == {202: 0, 422: 0, 503: 0}


async def test_api_metrics_use_resolved_routes_and_include_errors(metrics):
    app = FastAPI()
    app.state.telemetry = Telemetry.from_env("hexrl-api")
    add_request_logging(app)

    @app.get("/api/v1/tasks/{task_id}")
    async def task(task_id: str):
        return Response(status_code=404)

    @app.post("/api/v1/tasks")
    async def overloaded():
        return Response(status_code=503)

    @app.get("/broken")
    async def broken():
        raise RuntimeError("test failure")

    @app.get("/healthz")
    async def health():
        return {}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for index in range(20):
            await client.get(f"/api/v1/tasks/id-{index}")
            await client.get(f"/missing/{index}")
        await client.post("/api/v1/tasks")
        await client.get("/healthz")
        await client.request("NONSTANDARD", "/api/v1/tasks/id")
        with pytest.raises(RuntimeError):
            await client.get("/broken")
    data = metrics("hexrl-api")
    requests = {
        (
            point.attributes["method"],
            point.attributes["route"],
            point.attributes["status_code"],
        ): point.value
        for point in data["hexrl.api.requests"]
    }
    assert requests["GET", "/api/v1/tasks/{task_id}", 404] == 20
    assert requests["GET", "unmatched", 404] == 20
    assert requests["POST", "/api/v1/tasks", 503] == 1
    assert requests["GET", "/broken", 500] == 1
    assert any(method == "OTHER" for method, route, status in requests)
    assert all(route != "/healthz" for method, route, status in requests)
    assert all(point.explicit_bounds == API_BUCKETS for point in data["hexrl.api.duration"])


async def test_source_counts_confirmed_publications_and_classifies_failures(metrics):
    telemetry = Telemetry.from_env("hexrl-source")
    failures = [
        QueueOverloadedError(reason="nack"),
        QueueOverloadedError(reason="timeout"),
        QueueOverloadedError(reason="unavailable"),
    ]
    messages = []

    class Publisher:
        async def publish(self, message):
            messages.append(message.task_id)
            if failures:
                raise failures.pop(0)

    async def lines():
        yield 1, "invalid"
        yield 2, STATE.model_dump_json()

    with telemetry.scope():
        await run_batch(
            Publisher(),
            lines(),
            "private-batch",
            SourceStats(),
            Backoff(0.001, 0.001, random.Random(42)),
        )
    data = metrics("hexrl-source")
    assert data["hexrl.source.published"][0].value == 1
    assert data["hexrl.source.invalid"][0].value == 1
    assert {
        point.attributes["reason"]: point.value for point in data["hexrl.source.rejections"]
    } == {
        "nack": 1,
        "timeout": 1,
        "unavailable": 1,
    }
    assert data["hexrl.source.wait"][0].value > 0
    assert len(set(messages)) == 1
    assert all(
        not {"task_id", "batch_id", "trace_id"}.intersection(point.attributes)
        for points in data.values()
        for point in points
    )


async def test_source_records_elapsed_wait_even_when_cancelled(metrics, monkeypatch):
    telemetry = Telemetry.from_env("hexrl-source")
    entered = asyncio.Event()
    clock = iter((10.0, 10.25))
    # Подменяем только ссылку в source, не глобальный модуль time и часы event loop.
    monkeypatch.setattr(batch_source, "time", SimpleNamespace(monotonic=lambda: next(clock)))

    class Publisher:
        async def publish(self, message):
            raise QueueOverloadedError()

    async def lines():
        yield 1, STATE.model_dump_json()

    async def wait(delay):
        entered.set()
        await asyncio.Future()

    with telemetry.scope():
        task = asyncio.create_task(
            run_batch(
                Publisher(),
                lines(),
                "b",
                SourceStats(),
                Backoff(0.001, 0.001, random.Random(42)),
                wait,
            )
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    data = metrics("hexrl-source")
    assert data["hexrl.source.wait"][0].value == 0.25
    assert data["hexrl.source.published"][0].value == 0


async def test_topology_failure_never_counts_as_published(metrics):
    telemetry = Telemetry.from_env("hexrl-source")

    class Publisher:
        async def publish(self, message):
            raise TopologyError()

    async def lines():
        yield 1, STATE.model_dump_json()

    with telemetry.scope(), pytest.raises(TopologyError):
        await run_batch(
            Publisher(), lines(), "b", SourceStats(), Backoff(0.01, 0.01, random.Random())
        )
    assert metrics("hexrl-source")["hexrl.source.published"][0].value == 0


async def test_predict_error_is_measured_and_keeps_millisecond_buckets(metrics):
    telemetry = Telemetry.from_env("hexrl-worker")

    def predict(state):
        raise RuntimeError("test failure")

    runner = InferenceRunner(predict, 1, 0.01)
    try:
        with telemetry.scope(), pytest.raises(RuntimeError):
            await runner.run(STATE)
    finally:
        runner.shutdown()
    data = metrics("hexrl-worker")
    (prediction,) = data["hexrl.worker.predict.duration"]
    assert prediction.explicit_bounds == PREDICT_BUCKETS
    assert prediction.count == 1
    assert prediction.attributes == {"outcome": "error"}
    assert data["hexrl.worker.delay.duration"][0].sum >= 0.01


@pytest.mark.parametrize("delay", [0, 0.5, 3, 10])
def test_processing_buckets_follow_configured_delay(metrics, delay):
    telemetry = Telemetry.from_env("hexrl-worker", inference_delay_s=delay)
    telemetry.metrics.worker_started(redelivered=False)
    telemetry.metrics.worker_finished("ok", delay + 0.001)
    (point,) = metrics("hexrl-worker")["hexrl.worker.processing.duration"]
    if delay:
        assert delay in point.explicit_bounds
        assert delay * 1.1 in point.explicit_bounds
    assert point.explicit_bounds[-1] >= 120


@pytest.mark.parametrize("delay", [-1, float("inf"), float("nan")])
def test_invalid_histogram_configuration_is_rejected(delay):
    with pytest.raises(ValueError):
        histogram_views(delay)


def test_views_remove_unbounded_attributes_from_counters_and_histograms(metrics):
    telemetry = Telemetry.from_env("hexrl-worker")
    for index in range(6000):
        attrs = {
            "task_id": str(index),
            "batch_id": "private",
            "trace_id": str(index),
            "outcome": "ok",
        }
        telemetry.metrics.worker_attempts.add(1, attrs)
        telemetry.metrics.predict_duration.record(0.003, attrs)
    data = metrics("hexrl-worker")
    points = data["hexrl.worker.predict.duration"]
    assert len(points) == 1
    assert points[0].count == 6000
    assert points[0].attributes == {"outcome": "ok"}
    ok = next(
        point for point in data["hexrl.worker.attempts"] if point.attributes["outcome"] == "ok"
    )
    assert ok.value == 6000


def test_invalid_clocks_are_counted_instead_of_clamped_to_zero(metrics):
    telemetry = Telemetry.from_env("hexrl-worker")
    telemetry.metrics.task_completed(datetime.now(UTC) + timedelta(hours=1))
    telemetry.metrics.task_completed(datetime.now())
    telemetry.metrics.task_completed(datetime.now(UTC) - timedelta(seconds=1))
    data = metrics("hexrl-worker")
    assert data["hexrl.worker.completed"][0].value == 3
    assert data["hexrl.worker.invalid.timestamps"][0].value == 2
    assert data["hexrl.worker.task.duration"][0].count == 1


def test_runtime_gauges_and_unbounded_queue_limit(metrics):
    telemetry = Telemetry.from_env("hexrl-source")
    data = metrics("hexrl-source")
    assert data["hexrl.process.memory"][0].value > 0
    assert data["hexrl.process.running"][0].value == 1
    assert "hexrl.queue.limit" not in data
    telemetry.metrics.set_queue_limit(20, "test.tasks")
    limit = metrics("hexrl-source")["hexrl.queue.limit"][0]
    assert limit.value == 20
    assert limit.attributes == {"queue": "test.tasks"}
    telemetry.metrics.set_queue_limit(None)
    assert "hexrl.queue.limit" not in metrics("hexrl-source")
    telemetry.metrics.stop()
    assert metrics("hexrl-source")["hexrl.process.running"][0].value == 0


def test_disabled_sdk_does_not_create_any_exporter(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    for name in ("OTLPMetricExporter", "OTLPSpanExporter"):
        monkeypatch.setattr(
            f"hexrl_platform.core.telemetry.{name}", Mock(side_effect=AssertionError)
        )
    telemetry = Telemetry.from_env("hexrl-source")
    assert isinstance(telemetry.meter_provider, NoOpMeterProvider)
    telemetry.flush_metrics()
    telemetry.shutdown()


async def test_short_source_exports_initial_and_final_cumulative_metrics(monkeypatch):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            message = ExportMetricsServiceRequest()
            message.ParseFromString(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, message))
            self.send_response(200)
            self.end_headers()

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for key, value in {
        "OTEL_SDK_DISABLED": "false",
        "OTEL_TRACES_EXPORTER": "none",
        "OTEL_METRICS_EXPORTER": "otlp",
        "OTEL_METRIC_EXPORT_INTERVAL": "60000",
        "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://127.0.0.1:{server.server_port}",
        "OTEL_RESOURCE_ATTRIBUTES": "service.instance.id=source-1",
    }.items():
        monkeypatch.setenv(key, value)
    broker = AsyncMock()
    monkeypatch.setattr(batch_source, "RabbitBroker", lambda *args, **kwargs: broker)
    monkeypatch.setattr(batch_source, "declare_topology", AsyncMock())
    publisher = Mock(publish=AsyncMock())
    monkeypatch.setattr(batch_source, "TaskPublisher", lambda *args: publisher)

    async def lines(path):
        for index in range(3):
            yield index + 1, STATE.model_dump_json()

    monkeypatch.setattr(batch_source, "read_lines", lines)
    try:
        stats = await batch_source.amain(
            Namespace(
                queue_prefix=None,
                unbounded=False,
                no_backpressure=False,
                metrics_csv=None,
                file=Path("unused"),
                batch_id="private-batch",
            )
        )
        assert stats.published_ok == 3
        broker.stop.assert_awaited_once()
    finally:
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        thread.join(timeout=2)
    assert len(received) == 2
    samples = []
    for path, request in received:
        assert path == "/v1/metrics"
        families = {
            metric.name: metric
            for resource in request.resource_metrics
            for scope in resource.scope_metrics
            for metric in scope.metrics
        }
        counter = families["hexrl.source.published"].sum
        assert counter.aggregation_temporality == 2  # CUMULATIVE
        samples.append(
            (
                counter.data_points[0].as_int,
                families["hexrl.process.running"].gauge.data_points[0].as_int,
            )
        )
        assert b"task_id" not in request.SerializeToString()
    assert samples == [(0, 1), (3, 0)]


def test_flush_and_shutdown_failures_do_not_escape(monkeypatch):
    from opentelemetry import trace
    from opentelemetry.sdk.metrics import MeterProvider

    provider = MeterProvider(shutdown_on_exit=False)
    telemetry = Telemetry(trace.NoOpTracerProvider(), provider)
    monkeypatch.setattr(provider, "force_flush", Mock(side_effect=TimeoutError))
    shutdown = Mock(side_effect=TimeoutError)
    monkeypatch.setattr(provider, "shutdown", shutdown)
    telemetry.flush_metrics()
    telemetry.shutdown()
    telemetry.shutdown()
    assert shutdown.call_count == 1
