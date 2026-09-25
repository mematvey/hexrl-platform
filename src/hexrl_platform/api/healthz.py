from fastapi import APIRouter

from hexrl_platform.schemas.health import LivenessResponse

router = APIRouter(tags=["infra"])


@router.get("/healthz", summary="Liveness probe")
async def healthz() -> LivenessResponse:
    return LivenessResponse(status="ok")
