import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import RequestResponseEndpoint

from hexrl_platform.api import healthz
from hexrl_platform.api.v1 import health, version
from hexrl_platform.core.config import APP_VERSION, get_settings
from hexrl_platform.core.logging import configure_logging, request_id_var
from hexrl_platform.db.session import create_engine, create_session_factory

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    engine = create_engine(settings.db_dsn)
    app.state.engine = engine
    app.state.session_factory = create_session_factory(engine)
    logger.info("application_started")
    yield
    await engine.dispose()
    logger.info("application_stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title=settings.app_name, version=APP_VERSION, lifespan=lifespan)

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

    app.include_router(healthz.router)
    app.include_router(version.router)
    app.include_router(health.router)
    return app


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "hexrl_platform.main:create_app",
        factory=True,
        host=settings.app_host,
        port=settings.app_port,
        access_log=False,
    )
