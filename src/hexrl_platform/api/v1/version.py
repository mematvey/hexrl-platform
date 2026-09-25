from fastapi import APIRouter, Depends

from hexrl_platform.core.config import APP_VERSION, Settings, get_settings
from hexrl_platform.schemas.health import VersionResponse

router = APIRouter(prefix="/api/v1", tags=["meta"])


@router.get("/version", summary="Application version")
async def get_version(settings: Settings = Depends(get_settings)) -> VersionResponse:
    return VersionResponse(name=settings.app_name, version=APP_VERSION)
