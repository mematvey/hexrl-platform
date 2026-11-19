import logging
import time
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from starlette.middleware.base import RequestResponseEndpoint

from hexrl_platform.core.logging import request_id_var
from hexrl_platform.core.telemetry import record_error

logger = logging.getLogger(__name__)


def add_request_logging(app: FastAPI) -> None:
    @app.middleware("http")
    async def log_request(request: Request, call_next: RequestResponseEndpoint) -> Response:
        telemetry = app.state.telemetry
        path = request.url.path
        route = "/api/v1/tasks/{task_id}" if path.startswith("/api/v1/tasks/") else path
        with (
            telemetry.scope(),
            telemetry.tracer.start_as_current_span(
                f"{request.method} {route}",
                context=TraceContextTextMapPropagator().extract(dict(request.headers)),
                kind=trace.SpanKind.SERVER,
                attributes={"http.request.method": request.method, "http.route": route},
                record_exception=False,
                set_status_on_exception=False,
            ) as span,
        ):
            try:
                response = await logged_request(request, call_next)
                span.set_attribute("http.response.status_code", response.status_code)
                if response.status_code >= 500:
                    span.set_status(trace.StatusCode.ERROR)
                return response
            except Exception as error:
                record_error(error)
                raise

    async def logged_request(request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status_code = 500
        error_type = None
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            details = {
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            }
            if hasattr(request.state, "task_id"):
                details["task_id"] = request.state.task_id
            if error_type is None:
                level = (
                    logging.DEBUG
                    if request.url.path == "/healthz" and status_code < 400
                    else logging.INFO
                )
                if status_code >= 500:
                    level = logging.ERROR
                logger.log(level, "http_request_completed", extra=details)
            else:
                logger.error("http_request_failed", extra={**details, "error_type": error_type})
            request_id_var.reset(token)
