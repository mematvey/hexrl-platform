import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from hexrl_platform.api import healthz
from hexrl_platform.api.v1 import health, version
from hexrl_platform.core.config import APP_VERSION, get_settings
from hexrl_platform.db.session import create_engine, create_session_factory

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    engine = create_engine(settings.db_dsn)
    app.state.engine = engine
    app.state.session_factory = create_session_factory(engine)
    logger.info("application started, version=%s", APP_VERSION)
    yield
    await engine.dispose()
    logger.info("application stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    app = FastAPI(title=settings.app_name, version=APP_VERSION, lifespan=lifespan)
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
    )
