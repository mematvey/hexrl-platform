import logging
import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from faststream.rabbit.opentelemetry import RabbitTelemetryMiddleware
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.metrics import MeterProvider as MeterProviderAPI
from opentelemetry.metrics import NoOpMeterProvider
from opentelemetry.sdk.metrics import Counter, Histogram, MeterProvider
from opentelemetry.sdk.metrics.export import AggregationTemporality, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Tracer
from opentelemetry.util.types import Attributes

from hexrl_platform.core.config import APP_VERSION
from hexrl_platform.core.logging import exception_stack
from hexrl_platform.core.metrics import PipelineMetrics, histogram_views, metrics_var

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
    def __init__(
        self,
        provider: trace.TracerProvider,
        meter_provider: MeterProviderAPI | None = None,
        *,
        role: str = "",
    ) -> None:
        self.provider = provider
        self.tracer = provider.get_tracer("hexrl_platform")
        self.meter_provider = meter_provider or NoOpMeterProvider()
        self.metrics = PipelineMetrics(
            self.meter_provider.get_meter("hexrl_platform"),
            role,
            runtime=isinstance(self.meter_provider, MeterProvider),
        )
        self._closed = False

    @classmethod
    def from_env(cls, service: str, *, inference_delay_s: float = 0.5) -> "Telemetry":
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
        timeout = float(os.getenv("OTEL_EXPORTER_OTLP_TIMEOUT", "0.5"))
        provider: trace.TracerProvider = trace.NoOpTracerProvider()
        meter_provider: MeterProviderAPI = NoOpMeterProvider()
        if os.getenv("OTEL_TRACES_EXPORTER", "otlp") != "none":
            provider = TracerProvider(resource=resource, shutdown_on_exit=False)
            provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(timeout=timeout),
                    max_queue_size=128,
                    max_export_batch_size=128,
                    schedule_delay_millis=1000,
                )
            )
        if os.getenv("OTEL_METRICS_EXPORTER", "otlp") != "none":
            reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(
                    timeout=timeout,
                    preferred_temporality={
                        Counter: AggregationTemporality.CUMULATIVE,
                        Histogram: AggregationTemporality.CUMULATIVE,
                    },
                ),
                export_interval_millis=float(os.getenv("OTEL_METRIC_EXPORT_INTERVAL", "5000")),
                export_timeout_millis=float(os.getenv("OTEL_METRIC_EXPORT_TIMEOUT", "1000")),
            )
            meter_provider = MeterProvider(
                metric_readers=[reader],
                resource=resource,
                views=histogram_views(inference_delay_s),
                shutdown_on_exit=False,
            )
        return cls(provider, meter_provider, role=service.rsplit("-", 1)[-1])

    def middleware(self) -> RabbitTelemetryMiddleware:
        return RabbitTelemetryMiddleware(
            tracer_provider=self.provider, meter_provider=self.meter_provider
        )

    @contextmanager
    def scope(self) -> Iterator[None]:
        token = tracer_var.set(self.tracer)
        metrics_token = metrics_var.set(self.metrics)
        try:
            yield
        finally:
            tracer_var.reset(token)
            metrics_var.reset(metrics_token)

    def flush_metrics(self) -> None:
        if isinstance(self.meter_provider, MeterProvider):
            try:
                self.meter_provider.force_flush(
                    timeout_millis=float(os.getenv("OTEL_METRIC_EXPORT_TIMEOUT", "1000"))
                )
            except Exception:
                logger.exception("metrics_flush_failed")

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.metrics.stop()
        try:
            if isinstance(self.meter_provider, MeterProvider):
                self.meter_provider.shutdown(
                    timeout_millis=float(os.getenv("OTEL_METRIC_EXPORT_TIMEOUT", "1000")) * 2
                )
        except Exception:
            logger.exception("metrics_shutdown_failed")
        finally:
            if isinstance(self.provider, TracerProvider):
                self.provider.shutdown()
