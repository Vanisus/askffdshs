import asyncio
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from app.api.dependencies import get_container
from app.api.schemas import BatchEnrichRequest, BatchEnrichResponse, EnrichResponse, MitreEnrichment, ValidationInfo
from app.models import Incident
from app.services import AppContainer, EnrichmentPipeline

router = APIRouter(tags=["enrichment"])


def _public_errors(exc: ValidationError) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in error.items() if key not in {"input", "url", "ctx"}}
        for error in exc.errors()
    ]


def _invalid(payload: dict[str, Any], exc: ValidationError, container: AppContainer) -> EnrichResponse:
    return EnrichResponse(status="invalid", attack_version=container.attack_store.version, attack_loaded=container.attack_store.loaded, validation=ValidationInfo(is_valid=False, errors=_public_errors(exc)), incident=payload, mitre=MitreEnrichment(abstained=True))


async def _enrich(payload: dict[str, Any], container: AppContainer) -> EnrichResponse:
    try:
        normalized = Incident.model_validate(payload).model_dump(mode="json")
    except ValidationError as exc:
        return _invalid(payload, exc, container)
    pipeline = EnrichmentPipeline(container)
    semaphore = container.inference_semaphore or asyncio.Semaphore(1)
    try:
        async with semaphore:
            result = await asyncio.wait_for(run_in_threadpool(pipeline.run, normalized), timeout=container.settings.request_timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="enrichment timeout") from exc
    return EnrichResponse(status="ok", attack_version=container.attack_store.version, attack_loaded=container.attack_store.loaded, validation=ValidationInfo(is_valid=True, errors=[]), incident=result.incident, mitre=MitreEnrichment(**result.mitre))


@router.post("/api/v1/enrich", response_model=EnrichResponse, summary="Enrich one SIEM incident")
async def enrich_incident(correlation_rule: dict[str, Any] = Body(...), container: AppContainer = Depends(get_container)) -> EnrichResponse:
    return await _enrich(correlation_rule, container)


@router.post("/api/v1/enrich/batch", response_model=BatchEnrichResponse, summary="Enrich a bounded batch of SIEM incidents")
async def enrich_batch(request: BatchEnrichRequest, container: AppContainer = Depends(get_container)) -> BatchEnrichResponse:
    if len(request.incidents) > container.settings.batch_max_size:
        raise HTTPException(status_code=413, detail=f"batch size exceeds {container.settings.batch_max_size}")
    results = await asyncio.gather(*(_enrich(item, container) for item in request.incidents))
    return BatchEnrichResponse(items=results)
