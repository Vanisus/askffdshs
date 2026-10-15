import datetime
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Incident(BaseModel):
    """Canonical incident accepted from both enriched alerts and raw correlations."""

    model_config = ConfigDict(extra="allow")

    alertId: uuid.UUID | None = None
    ruleId: str = ""
    ruleName: str = ""
    severity: int | None = None
    title: str = ""
    description: str = ""
    triggeredAt: datetime.datetime | None = None
    eventIds: list[uuid.UUID] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)
    sourceId: str = ""
    correlationKey: str = ""
    aggregationState: dict[str, Any] = Field(default_factory=dict)

    mitre_technique_ids: list[str] = Field(default_factory=list)
    mitre_tactic_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_raw_correlation(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        events = data.get("events")
        if not isinstance(events, list):
            events = []

        if not data.get("eventIds"):
            data["eventIds"] = [
                event.get("eventId")
                for event in events
                if isinstance(event, dict) and event.get("eventId")
            ]

        if not data.get("triggeredAt"):
            data["triggeredAt"] = (
                data.get("updatedAt")
                or data.get("createdAt")
                or cls._latest_event_timestamp(events)
            )

        context = dict(data.get("context") or {})
        for event in events:
            if not isinstance(event, dict):
                continue
            fields = event.get("fields")
            if not isinstance(fields, dict):
                continue
            for key, item in fields.items():
                if item not in (None, ""):
                    context.setdefault(str(key), item)

        aggregation = data.get("aggregationState")
        if isinstance(aggregation, dict):
            total = aggregation.get("totalEvents")
            if total is not None:
                context.setdefault("total_events", total)
            results = aggregation.get("results")
            if isinstance(results, dict):
                for key, item in results.items():
                    if isinstance(item, (str, int, float, bool)):
                        canonical = str(key).rsplit(".", 1)[-1]
                        context.setdefault(canonical, item)

        data["context"] = context
        return data

    @staticmethod
    def _latest_event_timestamp(events: list[Any]) -> Any:
        timestamps = [
            event.get("timestamp")
            for event in events
            if isinstance(event, dict) and event.get("timestamp")
        ]
        return max(timestamps) if timestamps else None
