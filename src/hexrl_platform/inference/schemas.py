from pydantic import BaseModel, ConfigDict, Field


class ProcessRequest(BaseModel):
    q: int
    r: int
    goal_q: int
    goal_r: int
    # верхняя граница зависит от max_steps загруженной модели, поэтому её проверяет модель
    steps_remaining: int = Field(ge=1)


class Direction(BaseModel):
    dq: int
    dr: int


class ModelInfo(BaseModel):
    name: str
    version: str
    alias: str
    run_id: str | None
    max_steps: int


class ProcessResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    action: int
    direction: Direction
    model: ModelInfo
