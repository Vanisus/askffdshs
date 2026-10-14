from typing import Any

from fastapi import APIRouter, Body, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from app.api.dependencies import get_container
from app.api.schemas import EnrichResponse, MitreEnrichment, ValidationInfo
from app.models import Incident
from app.services import AppContainer, EnrichmentPipeline

router = APIRouter(tags=["enrichment"])


@router.post(
    "/api/v1/enrich",
    response_model=EnrichResponse,
    summary="Enrich SIEM incident with MITRE ATT&CK techniques",
)
async def enrich_incident(
    correlation_rule: dict[str, Any] = Body(..., description="Raw SIEM incident JSON"),
    container: AppContainer = Depends(get_container),
) -> EnrichResponse:
    pipeline = EnrichmentPipeline(container)

    try:
        incident_model = Incident.model_validate(correlation_rule)
        normalized = incident_model.model_dump(mode="json")
        validation = ValidationInfo(is_valid=True, errors=[])
    except ValidationError as exc:
        # отдаём клиенту исходный payload + ошибки валидации
        return EnrichResponse(
            status="invalid",
            attack_version=container.attack_store.version,
            attack_loaded=container.attack_store.loaded,
            validation=ValidationInfo(is_valid=False, errors=exc.errors()),
            incident=correlation_rule,
            mitre=MitreEnrichment(),
        )

    result = await run_in_threadpool(pipeline.run, normalized)

    return EnrichResponse(
        status="ok",
        attack_version=container.attack_store.version,
        attack_loaded=container.attack_store.loaded,
        validation=validation,
        incident=result.incident,
        mitre=MitreEnrichment(**result.mitre),
    )
