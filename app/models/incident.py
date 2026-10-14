import datetime
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Incident(BaseModel):
    model_config = ConfigDict(extra="allow")

    alertId: uuid.UUID
    ruleId: str
    ruleName: str
    severity: int
    title: str
    description: str
    triggeredAt: datetime.datetime
    eventIds: list[uuid.UUID] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    mitre_technique_ids: list[str] = Field(default_factory=list)
    mitre_tactic_ids: list[str] = Field(default_factory=list)