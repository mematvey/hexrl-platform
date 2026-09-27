import json
import logging
from io import StringIO

import pytest
from asgi_lifespan import LifespanManager

from hexrl_platform.core.logging import JsonFormatter, configure_logging


@pytest.fixture
def log_output(app):
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("hexrl_platform")
    logger.addHandler(handler)
    yield stream
    logger.removeHandler(handler)


def events(stream):
    return [json.loads(line) for line in stream.getvalue().splitlines()]


async def test_request_log_contains_response_details(client, log_output):
    response = await client.get("/healthz", headers={"X-Request-ID": "request-123"})

    assert response.headers["X-Request-ID"] == "request-123"
    (event,) = events(log_output)
    assert event["event"] == "http_request_completed"
    assert event["request_id"] == "request-123"
    assert event["method"] == "GET"
    assert event["path"] == "/healthz"
    assert event["status_code"] == 200
    assert event["duration_ms"] >= 0
    assert event["timestamp"].endswith("Z")
    assert event["service"] == "hexrl-platform"


async def test_component_failure_uses_same_request_id(client, session, log_output):
    session.error = ConnectionRefusedError("db is down")

    response = await client.get("/api/v1/health")

    assert response.status_code == 503
    component_event, request_event = events(log_output)
    request_id = response.headers["X-Request-ID"]
    assert component_event["event"] == "component_check_failed"
    assert component_event["component"] == "postgres"
    assert component_event["error_type"] == "ConnectionRefusedError"
    assert component_event["request_id"] == request_id
    assert request_event["request_id"] == request_id
    assert request_event["status_code"] == 503
    assert "db is down" not in log_output.getvalue()


async def test_unhandled_error_is_logged_without_message(app, client, log_output):
    @app.get("/broken")
    async def broken():
        raise RuntimeError("private-value")

    with pytest.raises(RuntimeError, match="private-value"):
        await client.get("/broken")

    (event,) = events(log_output)
    assert event["event"] == "http_request_failed"
    assert event["status_code"] == 500
    assert event["error_type"] == "RuntimeError"
    assert "private-value" not in log_output.getvalue()


async def test_lifecycle_logs(app, log_output):
    async with LifespanManager(app):
        pass

    assert [event["event"] for event in events(log_output)] == [
        "application_started",
        "application_stopped",
    ]


def test_formatter_ignores_unapproved_fields(log_output):
    logging.getLogger("hexrl_platform").info(
        "configuration_checked", extra={"postgres_password": "secret-value"}
    )

    (event,) = events(log_output)
    assert event["event"] == "configuration_checked"
    assert "secret-value" not in log_output.getvalue()


def test_invalid_log_level():
    with pytest.raises(ValueError, match="Unknown log level"):
        configure_logging("not-a-level")
