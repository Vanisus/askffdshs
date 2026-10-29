"""Validated HTTP settings; no model weights are loaded at import time."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .services.catalog import MAX_BYTES
from .services.features import FeatureConfig
from .services.foreground import ForegroundConfig
from .services.inspection import InspectionConfig, Inspector


@dataclass(frozen=True)
class HTTPSettings:
    data_dir: Path = Path("./data")
    device: str = "cuda"
    api_key: str | None = field(default=None, repr=False)
    max_upload_bytes: int = MAX_BYTES
    max_request_bytes: int = 128 * 1024 * 1024
    features: FeatureConfig = field(default_factory=FeatureConfig)
    foreground: ForegroundConfig = field(default_factory=ForegroundConfig)
    inspection: InspectionConfig = field(default_factory=InspectionConfig)
    hub_dir: str | None = None

    def __post_init__(self):
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("INSPECTION_DEVICE must be cpu or cuda")
        if not 0 < self.max_upload_bytes <= MAX_BYTES:
            raise ValueError("Upload limit must be in (0, 50 MiB]")
        if self.max_request_bytes < self.max_upload_bytes:
            raise ValueError("Invalid request size")
        if self.api_key is not None and (not self.api_key.strip() or not self.api_key.isascii()):
            raise ValueError("API key must be nonempty ASCII")

    @classmethod
    def from_env(cls):
        return cls(
            data_dir=Path(os.getenv("INSPECTION_DATA_DIR", "./data")),
            device=os.getenv("INSPECTION_DEVICE", "cuda"),
            api_key=os.getenv("INSPECTION_API_KEY") or None,
            max_upload_bytes=int(os.getenv("INSPECTION_MAX_UPLOAD_BYTES", str(MAX_BYTES))),
            max_request_bytes=int(os.getenv("INSPECTION_MAX_REQUEST_BYTES", str(128 * 1024 * 1024))),
            features=FeatureConfig(
                backbone=os.getenv("INSPECTION_BACKBONE", "dinov2_vits14"),
                tile_size=int(os.getenv("INSPECTION_TILE_SIZE", "448")),
                overlap=int(os.getenv("INSPECTION_TILE_OVERLAP", "112")),
                local_radius=int(os.getenv("INSPECTION_LOCAL_RADIUS", "1"))),
            foreground=ForegroundConfig(min_area=int(os.getenv("INSPECTION_MIN_PART_AREA", "5000"))),
            inspection=InspectionConfig(
                threshold=float(os.getenv("INSPECTION_THRESHOLD", ".15")),
                reference_quantile=float(os.getenv("INSPECTION_REFERENCE_QUANTILE", ".25")),
                min_area_px=int(os.getenv("INSPECTION_MIN_DEFECT_AREA", "30")),
                edge_margin_px=int(os.getenv("INSPECTION_EDGE_MARGIN", "4"))),
            hub_dir=os.getenv("DINO_HUB_DIR") or None,
        )

    def make_inspector(self):
        return Inspector(self.data_dir, self.features, self.foreground, self.inspection,
                         device=self.device, hub_dir=self.hub_dir)
