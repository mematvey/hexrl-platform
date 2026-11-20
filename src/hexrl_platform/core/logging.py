import json
import logging
import os
import re
import socket
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from types import TracebackType

from opentelemetry import trace

from hexrl_platform.core.config import APP_VERSION, DIST_NAME

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
task_context_var: ContextVar[dict[str, str]] = ContextVar("task_context", default={})
service_name = DIST_NAME

LOG_FIELDS = (
    "method",
    "path",
    "status_code",
    "duration_ms",
    "component",
    "component_version",
    "error_type",
    "error",
    "task_id",
    "attempt",
    "reason",
    "batch_id",
    "model_alias",
    "model_version",
    "prefetch_count",
    "worker_concurrency",
    "wait_s",
    "queue",
)


@contextmanager
def task_context(task_id: str, batch_id: str | None = None) -> Iterator[None]:
    values = {"task_id": task_id}
    if batch_id is not None:
        values["batch_id"] = batch_id
    token = task_context_var.set(values)
    span = trace.get_current_span()
    for key, value in values.items():
        span.set_attribute(key, value)
    try:
        yield
    finally:
        task_context_var.reset(token)


def redact(value: str) -> str:
    value = re.sub(r"(\w[\w+.-]*://)[^\s/@]+(?::[^\s/@]*)?@", r"\1[redacted]@", value)
    for key, secret in os.environ.items():
        if (
            secret
            and len(secret) >= 4
            and any(word in key.upper() for word in ("PASSWORD", "SECRET", "TOKEN", "ACCESS_KEY"))
        ):
            value = value.replace(secret, "[redacted]")
    return value


def exception_stack(tb: TracebackType | None) -> str:
    # Кадры сохраняются без текста строк: литералы в исходниках тоже могут быть секретами.
    return redact(
        "".join(
            f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}\n'
            for frame in traceback.extract_tb(tb)
        )
    )


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, str | int | float] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "event": redact(record.getMessage()),
            "service": os.getenv("OTEL_SERVICE_NAME", service_name),
            "service_version": APP_VERSION,
            "env": "development",
            "service_instance_id": socket.gethostname(),
        }
        resource = dict(
            entry.split("=", 1)
            for entry in os.getenv("OTEL_RESOURCE_ATTRIBUTES", "").split(",")
            if "=" in entry
        )
        data["env"] = resource.get("deployment.environment.name", "development")
        data["service"] = os.getenv("OTEL_SERVICE_NAME", resource.get("service.name", service_name))
        data["service_version"] = resource.get("service.version", APP_VERSION)
        data["service_instance_id"] = resource.get("service.instance.id", socket.gethostname())
        request_id = request_id_var.get()
        if request_id is not None:
            data["request_id"] = request_id
        data.update(task_context_var.get())
        context = trace.get_current_span().get_span_context()
        if context.is_valid:
            data["trace_id"] = f"{context.trace_id:032x}"
            data["span_id"] = f"{context.span_id:016x}"
        for field in LOG_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                data[field] = redact(value) if isinstance(value, str) else value
        if record.exc_info is not None:
            data["error_type"] = record.exc_info[0].__name__ if record.exc_info[0] else "Exception"
            data["stacktrace"] = exception_stack(record.exc_info[2])
        return json.dumps(data, ensure_ascii=False)


def configure_logging(level: str, service: str = DIST_NAME) -> None:
    global service_name
    service_name = service
    level_number = logging.getLevelNamesMapping().get(level.upper())
    if level_number is None:
        raise ValueError(f"Unknown log level: {level}")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    # Один stdout handler для приложения и библиотек, без дублирования Uvicorn.
    root = logging.getLogger()
    for old_handler in root.handlers[:]:
        if old_handler.name == "hexrl-stdout":
            root.removeHandler(old_handler)
    handler.set_name("hexrl-stdout")
    root.addHandler(handler)
    root.setLevel(level_number)
    for logger in list(logging.Logger.manager.loggerDict.values()):
        if isinstance(logger, logging.Logger):
            for old_handler in logger.handlers[:]:
                logger.removeHandler(old_handler)
            logger.setLevel(logging.NOTSET)
            logger.propagate = True
    # Сброс выше делает эти логгеры наследниками INFO. SQLAlchemy писал бы каждый запрос
    # с параметрами, то есть с payload задачи; uvicorn.access оживал бы вопреки
    # access_log=False и дублировал http_request_completed без trace_id. Ошибки видны.
    for name in ("sqlalchemy", "uvicorn.access"):
        logging.getLogger(name).setLevel(max(level_number, logging.WARNING))
