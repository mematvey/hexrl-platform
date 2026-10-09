from fastapi import APIRouter, Depends, HTTPException, Request

from hexrl_platform.inference.schemas import ModelInfo, ProcessRequest, ProcessResponse
from hexrl_platform.inference.service import InvalidStateError, PolicyService

router = APIRouter(tags=["inference"])


def get_service(request: Request) -> PolicyService:
    return request.app.state.policy_service


@router.get("/healthz", summary="Liveness probe")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/model", summary="Which model version is being served")
async def get_model_info(service: PolicyService = Depends(get_service)) -> ModelInfo:
    return service.info


@router.post("/process", summary="Choose the next action for a hex state")
async def process(
    payload: ProcessRequest,
    service: PolicyService = Depends(get_service),
) -> ProcessResponse:
    try:
        return await service.process(payload)
    except InvalidStateError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
