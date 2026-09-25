from functools import lru_cache
from importlib.metadata import PackageNotFoundError, distribution, version

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
    postgres_user: str = "hexrl"
    postgres_password: str = "hexrl"
    postgres_db: str = "hexrl"

    health_timeout_seconds: float = 2.0

    @property
    def db_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
