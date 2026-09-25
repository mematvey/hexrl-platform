from typing import Literal

from pydantic import BaseModel


class LivenessResponse(BaseModel):
    status: Literal["ok"]


class VersionResponse(BaseModel):
    name: str
    version: str


class ComponentHealth(BaseModel):
    name: str
    status: Literal["up", "down"]
    version: str | None = None
    latency_ms: float
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    components: list[ComponentHealth]
