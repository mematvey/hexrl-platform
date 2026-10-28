from pydantic import BaseModel


class ModelInfo(BaseModel):
    """Какая версия модели обслуживается: отдаётся воркером на GET /model."""

    name: str
    version: str
    alias: str
    run_id: str | None
    max_steps: int
