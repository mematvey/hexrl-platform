import logging
import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from faststream.rabbit.opentelemetry import RabbitTelemetryMiddleware
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Tracer
from opentelemetry.util.types import Attributes

from hexrl_platform.core.config import APP_VERSION
from hexrl_platform.core.logging import exception_stack

logger = logging.getLogger(__name__)
tracer_var: ContextVar[Tracer | None] = ContextVar("tracer", default=None)


def current_tracer() -> Tracer:
    return tracer_var.get() or trace.get_tracer("hexrl_platform")


def record_error(error: BaseException) -> None:
    # Сообщения исключений могут содержать входную запись или DSN. Сохраняем
    # тип и кадры стека без значений аргументов и без текста исходного payload.
    span = trace.get_current_span()
    span.set_status(trace.StatusCode.ERROR)
    span.record_exception(
        error,
        attributes={
            "exception.message": type(error).__name__,
            "exception.stacktrace": exception_stack(error.__traceback__),
        },
    )


@contextmanager
def operation(name: str, attributes: Attributes = None) -> Iterator[Span]:
    with current_tracer().start_as_current_span(
        name, attributes=attributes, record_exception=False, set_status_on_exception=False
    ) as span:
        try:
            yield span
        except BaseException as error:
            record_error(error)
            raise


class Telemetry:
    def __init__(self, provider: trace.TracerProvider) -> None:
        self.provider = provider
        self.tracer = provider.get_tracer("hexrl_platform")

    @classmethod
    def from_env(cls, service: str) -> "Telemetry":
        if os.getenv("OTEL_SDK_DISABLED", "true").lower() == "true":
            return cls(trace.NoOpTracerProvider())
        resource = Resource.create(
            {
                "service.name": os.getenv("OTEL_SERVICE_NAME", service),
                "service.version": APP_VERSION,
                "deployment.environment.name": "development",
                "service.instance.id": socket.gethostname(),
            }
        )
        # OTEL_RESOURCE_ATTRIBUTES имеет приоритет над значениями по умолчанию.
        resource = resource.merge(Resource.create())
        if str(resource.attributes.get("service.name", "")).startswith("unknown_service"):
            resource = resource.merge(Resource({"service.name": service}))
        provider = TracerProvider(resource=resource, shutdown_on_exit=False)
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(timeout=float(os.getenv("OTEL_EXPORTER_OTLP_TIMEOUT", "0.5"))),
                max_queue_size=128,
                max_export_batch_size=128,
                schedule_delay_millis=1000,
            )
        )
        return cls(provider)

    def middleware(self) -> RabbitTelemetryMiddleware:
        return RabbitTelemetryMiddleware(tracer_provider=self.provider)

    @contextmanager
    def scope(self) -> Iterator[None]:
        token = tracer_var.set(self.tracer)
        try:
            yield
        finally:
            tracer_var.reset(token)

    def shutdown(self) -> None:
        if isinstance(self.provider, TracerProvider):
            self.provider.shutdown()
