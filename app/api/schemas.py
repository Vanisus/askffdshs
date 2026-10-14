from typing import Any

from pydantic import BaseModel, Field


class ValidationInfo(BaseModel):
    is_valid: bool
    errors: list[dict[str, Any]] = Field(default_factory=list)


class MitreEnrichment(BaseModel):
    model_config = {"extra": "allow"}

    input_technique_ids: list[str] = Field(default_factory=list)
    matched_techniques: list[dict[str, Any]] = Field(default_factory=list)
    matched_tactics: list[dict[str, Any]] = Field(default_factory=list)
    unresolved_technique_ids: list[str] = Field(default_factory=list)
    mapping_reasons: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_candidates: list[dict[str, Any]] = Field(default_factory=list)
    reranked_candidates: list[dict[str, Any]] = Field(default_factory=list)
    suggested_technique_ids: list[str] = Field(default_factory=list)
    search_query: str = ""


class EnrichResponse(BaseModel):
    status: str = "ok"
    attack_version: str
    attack_loaded: bool
    validation: ValidationInfo
    incident: dict[str, Any]
    mitre: MitreEnrichment


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
