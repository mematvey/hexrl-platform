import logging
import time
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import RequestResponseEndpoint

from hexrl_platform.core.logging import request_id_var

logger = logging.getLogger(__name__)


def add_request_logging(app: FastAPI) -> None:
    @app.middleware("http")
    async def log_request(request: Request, call_next: RequestResponseEndpoint) -> Response:
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
            if error_type is None:
                logger.info("http_request_completed", extra=details)
            else:
                logger.error("http_request_failed", extra={**details, "error_type": error_type})
            request_id_var.reset(token)
