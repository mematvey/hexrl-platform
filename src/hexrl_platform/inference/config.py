from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class InferenceSettings(BaseSettings):
    # protected_namespaces отключён: поля model_* конфликтуют с префиксом pydantic
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    log_level: str = "INFO"
    inference_host: str = "0.0.0.0"
    inference_port: int = 8001

    mlflow_tracking_uri: str = "http://localhost:5000"
    model_name: str = "hex-navigation-default"
    model_alias: str = "champion"


@lru_cache
def get_inference_settings() -> InferenceSettings:
    return InferenceSettings()
