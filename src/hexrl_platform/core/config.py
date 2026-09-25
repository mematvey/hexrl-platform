from functools import lru_cache
from importlib.metadata import version

from pydantic_settings import BaseSettings, SettingsConfigDict

APP_VERSION: str = version("hexrl-platform")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "hexrl-platform"
    log_level: str = "INFO"

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
