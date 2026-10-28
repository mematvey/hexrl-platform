import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

import uvicorn
from fastapi import FastAPI
from faststream.rabbit import RabbitBroker

from hexrl_platform.api import healthz
from hexrl_platform.api.v1 import health, tasks, version
from hexrl_platform.core.config import APP_VERSION, get_settings
from hexrl_platform.core.logging import configure_logging
from hexrl_platform.core.middleware import add_request_logging
from hexrl_platform.db.schema import create_schema
from hexrl_platform.db.session import create_engine, create_session_factory
from hexrl_platform.service.tasks import cleanup_pending, pending_cleanup_loop
from hexrl_platform.task_pipeline.config import get_pipeline_settings
from hexrl_platform.task_pipeline.publisher import TaskPublisher
from hexrl_platform.task_pipeline.topology import build_topology, declare_topology

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    pipeline = get_pipeline_settings()
    async with AsyncExitStack() as stack:
        engine = create_engine(settings.db_dsn)
        stack.push_async_callback(engine.dispose)
        await create_schema(engine)
        sessions = create_session_factory(engine)
        await cleanup_pending(sessions, settings.pending_task_ttl_s)
        broker = await stack.enter_async_context(RabbitBroker(pipeline.rabbitmq_url))
        topology = build_topology(
            pipeline.queue_prefix, pipeline.queue_max_length, pipeline.retry_delay_ms
        )
        await declare_topology(broker, topology)
        app.state.engine = engine
        app.state.session_factory = sessions
        app.state.task_publisher = TaskPublisher(
            broker, topology, pipeline.publish_confirm_timeout_s
        )
        cleaner = asyncio.create_task(pending_cleanup_loop(sessions, settings))
        logger.info("application_started")
        try:
            yield
        finally:
            cleaner.cancel()
            await asyncio.gather(cleaner, return_exceptions=True)
    logger.info("application_stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title=settings.app_name, version=APP_VERSION, lifespan=lifespan)
    add_request_logging(app)

    app.include_router(healthz.router)
    app.include_router(version.router)
    app.include_router(health.router)
    app.include_router(tasks.router)
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
