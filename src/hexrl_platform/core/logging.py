import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime

from hexrl_platform.core.config import APP_VERSION, DIST_NAME

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

LOG_FIELDS = (
    "method",
    "path",
    "status_code",
    "duration_ms",
    "component",
    "component_version",
    "error_type",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, str | int | float] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "service": DIST_NAME,
            "service_version": APP_VERSION,
        }
        request_id = request_id_var.get()
        if request_id is not None:
            data["request_id"] = request_id
        for field in LOG_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                data[field] = value
        return json.dumps(data, ensure_ascii=False)


def configure_logging(level: str) -> None:
    level_number = logging.getLevelNamesMapping().get(level.upper())
    if level_number is None:
        raise ValueError(f"Unknown log level: {level}")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    logger = logging.getLogger("hexrl_platform")
    for old_handler in logger.handlers[:]:
        logger.removeHandler(old_handler)
    logger.addHandler(handler)
    logger.setLevel(level_number)
    logger.propagate = False
