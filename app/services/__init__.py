from typing import Any

from .container import AppContainer, ResultCache
from .enrichment_pipeline import EnrichmentPipeline, EnrichmentResult


def build_container(*args: Any, **kwargs: Any) -> AppContainer:
    from .startup import build_container as _build_container

    return _build_container(*args, **kwargs)


__all__ = ["AppContainer", "ResultCache", "EnrichmentPipeline", "EnrichmentResult", "build_container"]
