import torch
from fastapi import APIRouter, Depends

from app.api.dependencies import get_container
from app.api.schemas import (
    DeviceInfo,
    HealthLiveResponse,
    HealthReadyResponse,
)
from app.services import AppContainer

router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=HealthLiveResponse)
async def health_live() -> HealthLiveResponse:
    return HealthLiveResponse(status="alive")


@router.get("/health/ready", response_model=HealthReadyResponse)
async def health_ready(
    container: AppContainer = Depends(get_container),
) -> HealthReadyResponse:
    return HealthReadyResponse(
        status="ready" if container.is_ready else "not_ready",
        is_ready=container.is_ready,
        attack_store=container.attack_store.health(),
        mitre_documents_count=len(container.mitre_documents),
        translator_enabled=container.translator is not None,
        hybrid_enabled=container.hybrid_retriever is not None,
        reranker_enabled=container.reranker is not None,
        warnings=container.warnings,
        device=_device_info(),
    )


def _device_info() -> DeviceInfo:
    if not torch.cuda.is_available():
        return DeviceInfo(cuda_available=False)

    try:
        return DeviceInfo(
            cuda_available=True,
            device_count=torch.cuda.device_count(),
            device_name=torch.cuda.get_device_name(0),
            memory_allocated_mb=round(
                torch.cuda.memory_allocated(0) / 1024 / 1024, 1
            ),
            memory_reserved_mb=round(
                torch.cuda.memory_reserved(0) / 1024 / 1024, 1
            ),
        )
    except Exception:
        return DeviceInfo(cuda_available=True)
