import math
from collections.abc import Iterable
from contextvars import ContextVar
from datetime import UTC, datetime
from threading import Lock
from typing import Literal

import psutil
from opentelemetry.metrics import Counter, Meter, NoOpMeterProvider, ObservableGauge, Observation
from opentelemetry.sdk.metrics.view import (
    DropAggregation,
    ExplicitBucketHistogramAggregation,
    View,
)

Outcome = Literal["ok", "error", "dlq", "duplicate", "cancelled"]
LABELS = {"method", "route", "status_code", "outcome", "reason"}
PREDICT_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)
API_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 10)
TASK_BUCKETS = (0.01, 0.05, 0.1, 0.5, 1, 2.5, 5, 10, 15, 30, 60, 120, 300)


def histogram_views(delay_s: float) -> list[View]:
    if not math.isfinite(delay_s) or delay_s < 0:
        raise ValueError("inference delay must be finite and non-negative")
    processing = tuple(
        sorted(
            {
                *API_BUCKETS,
                30.0,
                60.0,
                120.0,
                *(delay_s * factor for factor in (0.5, 1, 1.1, 2, 4) if delay_s > 0),
            }
        )
    )
    boundaries = {
        "hexrl.api.duration": API_BUCKETS,
        "hexrl.worker.predict.duration": PREDICT_BUCKETS,
        "hexrl.worker.delay.duration": processing,
        "hexrl.worker.processing.duration": processing,
        "hexrl.worker.task.duration": TASK_BUCKETS,
    }
    return [
        View(meter_name="hexrl_platform", instrument_type=Counter, attribute_keys=LABELS),
        View(meter_name="hexrl_platform", instrument_type=ObservableGauge, attribute_keys=set()),
        # У middleware остаются трейсы. Метрики приложения имеют собственный
        # контракт без routing key и без второго набора стандартных гистограмм.
        View(meter_name="faststream.opentelemetry.middleware", aggregation=DropAggregation()),
        *(
            View(
                meter_name="hexrl_platform",
                instrument_name=name,
                attribute_keys=LABELS,
                aggregation=ExplicitBucketHistogramAggregation(bounds),
            )
            for name, bounds in boundaries.items()
        ),
    ]


class PipelineMetrics:
    def __init__(self, meter: Meter, role: str = "", *, runtime: bool = True) -> None:
        self.api_requests = meter.create_counter("hexrl.api.requests", unit="1")
        self.api_duration = meter.create_histogram("hexrl.api.duration", unit="s")
        self.source_published = meter.create_counter("hexrl.source.published", unit="1")
        self.source_rejections = meter.create_counter("hexrl.source.rejections", unit="1")
        self.source_wait = meter.create_counter("hexrl.source.wait", unit="s")
        self.source_invalid = meter.create_counter("hexrl.source.invalid", unit="1")
        self.worker_attempts = meter.create_counter("hexrl.worker.attempts", unit="1")
        self.worker_completed = meter.create_counter("hexrl.worker.completed", unit="1")
        self.worker_redeliveries = meter.create_counter("hexrl.worker.redeliveries", unit="1")
        self.worker_retries = meter.create_counter("hexrl.worker.retries", unit="1")
        self.worker_dlq = meter.create_counter("hexrl.worker.dlq.publications", unit="1")
        self.predict_duration = meter.create_histogram("hexrl.worker.predict.duration", unit="s")
        self.delay_duration = meter.create_histogram("hexrl.worker.delay.duration", unit="s")
        self.processing_duration = meter.create_histogram(
            "hexrl.worker.processing.duration", unit="s"
        )
        self.task_duration = meter.create_histogram("hexrl.worker.task.duration", unit="s")
        self.invalid_timestamps = meter.create_counter("hexrl.worker.invalid.timestamps", unit="1")
        self._lock = Lock()
        self._inflight = 0
        self._running = True
        self._queue_limit: int | None = None
        if runtime:
            process = psutil.Process()

            def memory(_options: object) -> Iterable[Observation]:
                try:
                    yield Observation(process.memory_info().rss)
                except psutil.Error:
                    return

            meter.create_observable_gauge("hexrl.process.memory", [memory], unit="By")
            meter.create_observable_gauge(
                "hexrl.process.running", [self._observe_running], unit="1"
            )
            if role == "worker":
                meter.create_observable_gauge(
                    "hexrl.worker.inflight", [self._observe_inflight], unit="1"
                )
            if role in {"source", "api"}:
                meter.create_observable_gauge("hexrl.queue.limit", [self._observe_limit], unit="1")
        if role == "source":
            self.source_published.add(0)
            self.source_wait.add(0)
            self.source_invalid.add(0)
            for reason in ("nack", "timeout", "unavailable"):
                self.source_rejections.add(0, {"reason": reason})
        if role == "worker":
            for outcome in ("ok", "error", "dlq", "duplicate", "cancelled"):
                self.worker_attempts.add(0, {"outcome": outcome})
            for counter in (
                self.worker_completed,
                self.worker_redeliveries,
                self.worker_retries,
                self.worker_dlq,
                self.invalid_timestamps,
            ):
                counter.add(0)

    def _observe_running(self, _options: object) -> Iterable[Observation]:
        with self._lock:
            yield Observation(int(self._running))

    def _observe_inflight(self, _options: object) -> Iterable[Observation]:
        with self._lock:
            yield Observation(self._inflight)

    def _observe_limit(self, _options: object) -> Iterable[Observation]:
        if self._queue_limit is not None:
            yield Observation(self._queue_limit)

    def set_queue_limit(self, limit: int | None) -> None:
        self._queue_limit = limit

    def stop(self) -> None:
        with self._lock:
            self._running = False

    def worker_started(self, *, redelivered: bool) -> None:
        with self._lock:
            self._inflight += 1
        if redelivered:
            self.worker_redeliveries.add(1)

    def worker_finished(self, outcome: Outcome, duration_s: float) -> None:
        with self._lock:
            self._inflight -= 1
        self.worker_attempts.add(1, {"outcome": outcome})
        self.processing_duration.record(duration_s, {"outcome": outcome})

    def task_completed(self, submitted_at: datetime) -> None:
        self.worker_completed.add(1)
        if submitted_at.tzinfo is None:
            self.invalid_timestamps.add(1)
            return
        duration = (datetime.now(UTC) - submitted_at).total_seconds()
        if duration < 0:
            self.invalid_timestamps.add(1)
        else:
            self.task_duration.record(duration)

    def request_finished(self, method: str, route: str, status: int, duration_s: float) -> None:
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}:
            method = "OTHER"
        attrs = {"method": method, "route": route, "status_code": status}
        self.api_requests.add(1, attrs)
        self.api_duration.record(duration_s, attrs)


_NOOP = PipelineMetrics(NoOpMeterProvider().get_meter("hexrl_platform"), runtime=False)
metrics_var: ContextVar[PipelineMetrics] = ContextVar("metrics", default=_NOOP)


def current_metrics() -> PipelineMetrics:
    return metrics_var.get()
