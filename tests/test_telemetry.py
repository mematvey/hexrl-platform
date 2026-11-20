import asyncio
import json
import logging
import random
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from opentelemetry import trace
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.sdk.trace import TracerProvider

from hexrl_platform.core.logging import JsonFormatter, task_context, task_context_var
from hexrl_platform.core.telemetry import Telemetry, operation, record_error
from hexrl_platform.task_pipeline.batch_source import Backoff, SourceStats, run_batch
from hexrl_platform.task_pipeline.errors import QueueOverloadedError
from hexrl_platform.task_pipeline.runner import InferenceRunner
from hexrl_platform.task_pipeline.schemas import HexState

STATE = HexState(q=0, r=0, goal_q=2, goal_r=-1, steps_remaining=20)


async def test_concurrent_threads_keep_their_task_and_parent_span(traces):
    telemetry = Telemetry.from_env("worker")
    seen = []

    def predict(state):
        seen.append((task_context_var.get().copy(), trace.get_current_span().get_span_context()))
        return 2

    runner = InferenceRunner(predict, 2, 0, model_alias="champion", model_version="7")

    async def execute(index):
        with telemetry.scope(), operation("process"), task_context(str(index)):
            await runner.run(STATE)
            assert task_context_var.get() == {"task_id": str(index)}

    try:
        await asyncio.gather(*(execute(index) for index in range(12)))
    finally:
        runner.shutdown()
    assert task_context_var.get() == {}
    roots = {span.context.trace_id: span for span in traces() if span.name == "process"}
    inferences = {span.context.span_id: span for span in traces() if span.name == "inference"}
    assert len(roots) == len(inferences) == 12
    for context, span_context in seen:
        inference = inferences[span_context.span_id]
        parent = roots[span_context.trace_id]
        assert inference.parent.span_id == parent.context.span_id
        assert parent.attributes["task_id"] == context["task_id"]
        assert inference.attributes["model.version"] == "7"
        assert inference.attributes["model.alias"] == "champion"


async def test_source_has_one_trace_per_task_and_records_backoff(traces):
    telemetry = Telemetry.from_env("source")
    attempts = {}

    class Publisher:
        async def publish(self, message):
            count = attempts.get(message.task_id, 0)
            attempts[message.task_id] = count + 1
            if count == 0:
                raise QueueOverloadedError()

    async def lines():
        for index in range(1, 4):
            yield index, STATE.model_dump_json()

    stats = SourceStats()
    with telemetry.scope():
        await run_batch(
            Publisher(), lines(), "batch", stats, Backoff(0.001, 0.001, random.Random(42))
        )
    roots = [span for span in traces() if span.name == "source.task"]
    waits = [span for span in traces() if span.name == "source.wait"]
    assert len({span.context.trace_id for span in roots}) == 3
    assert all(span.parent is None for span in roots)
    assert len(waits) == stats.rejected == stats.published_ok == 3
    assert {span.parent.span_id for span in waits} == {span.context.span_id for span in roots}
    assert all(count == 2 for count in attempts.values())


async def test_context_resets_after_failed_inference(traces):
    telemetry = Telemetry.from_env("worker")

    def predict(state):
        raise ValueError("private payload")

    runner = InferenceRunner(predict, 1, 0)
    try:
        with (
            pytest.raises(ValueError),
            telemetry.scope(),
            operation("process"),
            task_context("bad"),
        ):
            await runner.run(STATE)
        with telemetry.scope(), operation("unrelated"):
            assert task_context_var.get() == {}
    finally:
        runner.shutdown()
    failed = [span for span in traces() if span.name in {"process", "inference"}]
    assert all(span.status.status_code is trace.StatusCode.ERROR for span in failed)
    assert all(span.events[0].name == "exception" for span in failed)
    assert "private payload" not in repr(
        [event.attributes for span in failed for event in span.events]
    )


def test_json_logs_automatically_include_context_and_redact_secrets(traces, monkeypatch):
    monkeypatch.setenv("POSTGRES_PASSWORD", "database-secret")
    telemetry = Telemetry.from_env("worker")
    with telemetry.scope(), operation("process") as span, task_context("task-1", "batch-1"):
        record = logging.LogRecord(
            "library",
            logging.ERROR,
            __file__,
            1,
            "connection amqp://user:password@broker failed: database-secret",
            None,
            None,
        )
        row = json.loads(JsonFormatter().format(record))
        assert row["trace_id"] == f"{span.get_span_context().trace_id:032x}"
        assert row["span_id"] == f"{span.get_span_context().span_id:016x}"
        assert row["task_id"] == "task-1"
        assert row["batch_id"] == "batch-1"
        assert "password" not in row["event"]
        assert "database-secret" not in row["event"]
    assert "task_id" not in json.loads(JsonFormatter().format(record))


def test_disabled_sdk_never_constructs_an_exporter(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.setattr(
        "hexrl_platform.core.telemetry.OTLPSpanExporter",
        lambda **kwargs: pytest.fail("disabled SDK created exporter"),
    )
    telemetry = Telemetry.from_env("worker")
    with telemetry.scope(), operation("inference") as span:
        assert not span.is_recording()
    telemetry.shutdown()


def test_otlp_exports_protobuf_with_configured_resource_and_error(monkeypatch):
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "none")
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            message = ExportTraceServiceRequest()
            message.ParseFromString(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, message))
            self.send_response(200)
            self.end_headers()

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "worker-test")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("OTEL_TRACES_SAMPLER", "always_on")
    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES",
        "service.instance.id=replica-2,service.version=42,deployment.environment.name=test",
    )
    telemetry = Telemetry.from_env("worker")
    try:
        with telemetry.scope(), operation("test"):
            record_error(ValueError("private"))
        telemetry.shutdown()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert len(requests) == 1
    path, request = requests[0]
    assert path == "/v1/traces"
    resource = {
        entry.key: entry.value.string_value
        for entry in request.resource_spans[0].resource.attributes
    }
    assert resource["service.name"] == "worker-test"
    assert resource["service.version"] == "42"
    assert resource["service.instance.id"] == "replica-2"
    span = request.resource_spans[0].scope_spans[0].spans[0]
    assert span.status.code == 2
    assert span.events[0].name == "exception"
    assert b"private" not in request.SerializeToString()


def test_instance_id_defaults_to_hostname_and_survives_restarts(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "none")
    monkeypatch.delenv("OTEL_RESOURCE_ATTRIBUTES", raising=False)
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    first, second = Telemetry.from_env("hexrl-worker"), Telemetry.from_env("hexrl-worker")
    try:
        resources = []
        for telemetry in (first, second):
            assert isinstance(telemetry.provider, TracerProvider)
            resources.append(telemetry.provider.resource.attributes)
        assert {item["service.instance.id"] for item in resources} == {socket.gethostname()}
        assert resources[0]["service.name"] == "hexrl-worker"
    finally:
        first.shutdown()
        second.shutdown()


def test_missing_collector_does_not_block_work_or_shutdown(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TIMEOUT", "0.05")
    monkeypatch.setenv("OTEL_TRACES_SAMPLER", "always_on")
    telemetry = Telemetry.from_env("worker")
    started = time.monotonic()
    with telemetry.scope():
        for _ in range(300):
            with operation("work"):
                pass
    elapsed = time.monotonic() - started
    telemetry.shutdown()
    assert elapsed < 1
    assert time.monotonic() - started < 2
