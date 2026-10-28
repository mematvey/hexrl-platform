from functools import lru_cache
from importlib.metadata import PackageNotFoundError, distribution, version

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DIST_NAME: str = "hexrl-platform"

try:
    APP_VERSION: str = version(DIST_NAME)
    APP_NAME: str = distribution(DIST_NAME).name
except PackageNotFoundError:
    APP_VERSION = "0.0.0"
    APP_NAME = DIST_NAME


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = APP_NAME
    log_level: str = "INFO"

    app_host: str = "0.0.0.0"
    app_port: int = 8000

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str
    postgres_password: str
    postgres_db: str

    health_timeout_seconds: float = 2.0
    task_submission_timeout_s: float = Field(default=0.75, gt=0)
    task_cleanup_timeout_s: float = Field(default=0.2, gt=0)
    pending_task_ttl_s: float = Field(default=60, ge=1)
    pending_cleanup_interval_s: float = Field(default=30, ge=1)
    estimated_worker_count: int = Field(default=1, ge=1)

    @property
    def db_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
