import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from hexrl_platform.core.config import APP_VERSION
from hexrl_platform.core.logging import configure_logging
from hexrl_platform.core.middleware import add_request_logging
from hexrl_platform.inference import api
from hexrl_platform.inference.config import InferenceSettings, get_inference_settings
from hexrl_platform.inference.model_loader import LoadedModel, load_champion
from hexrl_platform.inference.service import PolicyService

logger = logging.getLogger(__name__)

Loader = Callable[[InferenceSettings], LoadedModel]


def create_app(loader: Loader = load_champion) -> FastAPI:
    settings = get_inference_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # падаем на старте, если модель не загрузилась: оркестратор перезапустит,
        # притворяться живым с неработающим inference хуже
        loaded = loader(settings)
        app.state.policy_service = PolicyService(loaded)
        logger.info(
            "model_loaded",
            extra={"component": loaded.name, "component_version": loaded.version},
        )
        yield
        logger.info("inference_stopped")

    app = FastAPI(title="hexrl-inference", version=APP_VERSION, lifespan=lifespan)
    add_request_logging(app)
    app.include_router(api.router)
    return app


def main() -> None:
    settings = get_inference_settings()
    uvicorn.run(
        "hexrl_platform.inference.main:create_app",
        factory=True,
        host=settings.inference_host,
        port=settings.inference_port,
        access_log=False,
    )
