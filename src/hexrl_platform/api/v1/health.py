from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from hexrl_platform.core.config import Settings, get_settings
from hexrl_platform.db.session import get_session
from hexrl_platform.schemas.health import HealthResponse
from hexrl_platform.service.health_service import collect_health

router = APIRouter(prefix="/api/v1", tags=["infra"])


@router.get("/health", summary="End-to-end health check")
async def health(
    response: Response,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    report = await collect_health(session, settings.health_timeout_seconds)
    if report.status != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return report
