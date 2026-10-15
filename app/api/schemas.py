from typing import Any

from pydantic import BaseModel, Field


class ValidationInfo(BaseModel):
    is_valid: bool
    errors: list[dict[str, Any]] = Field(default_factory=list)


class MitreTechnique(BaseModel):
    id: str
    name: str
    tactics: list[dict[str, str]] = Field(default_factory=list)
    is_subtechnique: bool = False


class MitreDecision(BaseModel):
    source: str = "automatic"
    abstained: bool = True
    reason: str = "no_candidates"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class MitreEnrichment(BaseModel):
    techniques: list[MitreTechnique] = Field(default_factory=list)
    tactics: list[dict[str, str]] = Field(default_factory=list)
    decision: MitreDecision = Field(default_factory=MitreDecision)
    debug: dict[str, Any] | None = None


class EnrichResponse(BaseModel):
    status: str = "ok"
    attack_version: str
    attack_loaded: bool
    validation: ValidationInfo
    incident: dict[str, Any]
    mitre: MitreEnrichment


class BatchEnrichRequest(BaseModel):
    incidents: list[dict[str, Any]] = Field(min_length=1)
    debug: bool = False


class BatchEnrichResponse(BaseModel):
    items: list[EnrichResponse]


class HealthLiveResponse(BaseModel):
    status: str


class DeviceInfo(BaseModel):
    cuda_available: bool = False
    device_count: int = 0
    device_name: str | None = None
    memory_allocated_mb: float | None = None
    memory_reserved_mb: float | None = None


class HealthReadyResponse(BaseModel):
    status: str
    is_ready: bool
    attack_store: dict[str, Any]
    mitre_documents_count: int
    translator_enabled: bool
    hybrid_enabled: bool
    reranker_enabled: bool
    warnings: list[str] = Field(default_factory=list)
    device: DeviceInfo = Field(default_factory=DeviceInfo)


class DebugSearchResponse(BaseModel):
    query: str
    results: list[dict[str, Any]]
    reranked: list[dict[str, Any]] = Field(default_factory=list)


class DebugSearchTextResponse(BaseModel):
    original_query: str
    translated_query: str
    final_query: str
    results: list[dict[str, Any]]
    reranked: list[dict[str, Any]] = Field(default_factory=list)
