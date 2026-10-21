import logging
from functools import lru_cache
from typing import Self
from urllib.parse import quote

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class PipelineSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    rabbitmq_user: str
    rabbitmq_password: str
    rabbitmq_host: str = "localhost"
    rabbitmq_port: int = 5672

    queue_prefix: str = "hexrl.inference"
    queue_max_length: int | None = Field(default=20, ge=1)

    worker_concurrency: int = Field(default=2, ge=1)
    prefetch_count: int = Field(default=2, ge=1)
    inference_delay_s: float = Field(default=0.5, ge=0)

    max_deliveries: int = Field(default=3, ge=1)
    retry_delay_ms: int = Field(default=5000, ge=1)

    publish_confirm_timeout_s: float = Field(default=0.5, gt=0)
    backoff_base_s: float = Field(default=0.05, gt=0)
    backoff_cap_s: float = Field(default=1.0, gt=0)

    @property
    def rabbitmq_url(self) -> str:
        user = quote(self.rabbitmq_user, safe="")
        password = quote(self.rabbitmq_password, safe="")
        return f"amqp://{user}:{password}@{self.rabbitmq_host}:{self.rabbitmq_port}/"

    @model_validator(mode="after")
    def _warn_prefetch_above_concurrency(self) -> Self:
        if self.prefetch_count > self.worker_concurrency:
            logger.warning(
                "prefetch_above_concurrency",
                extra={
                    "prefetch_count": self.prefetch_count,
                    "worker_concurrency": self.worker_concurrency,
                },
            )
        return self


@lru_cache
def get_pipeline_settings() -> PipelineSettings:
    return PipelineSettings()
