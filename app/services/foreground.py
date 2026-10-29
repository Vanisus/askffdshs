"""Original main preparation by default; explicit masks bypass segmentation."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from .part_preparation import (
    PreparedPart, PreparationConfig, PreparationError, crop_part,
    place_on_background, prepare_part, _validate_image,
)


@dataclass(frozen=True)
class ForegroundConfig:
    max_side: int = 1280
    min_area: int = 5000
    padding: int = 35
    background_distance: float = 16.0
    method: str = "legacy_grabcut"
    box_ratio: float = .48

    def __post_init__(self):
        if self.method != "legacy_grabcut":
            raise ValueError("Only legacy_grabcut is supported; explicit masks bypass segmentation")
        if not .10 <= self.box_ratio <= .95:
            raise ValueError("box_ratio must lie in [0.10, 0.95]")
        if self.max_side < 64 or self.min_area < 1 or self.padding < 0 or self.background_distance <= 0:
            raise ValueError("Invalid foreground configuration")


def prepare_foreground(image, config=None, mask=None):
    config = config or ForegroundConfig()
    _validate_image(image)
    if mask is None:
        # Call the original implementation on the full-resolution image.
        # Preserve its contour-filling semantics. Disable only edge feathering
        # so table pixels are not blended into the inspection features.
        # If the init box lies fully inside a large textured part, GrabCut may
        # fail to separate it from the table: retry with a wider box, keeping
        # the configured ratio as the first attempt.
        last_error = None
        for box_ratio in (config.box_ratio, .75, .9):
            try:
                return prepare_part(image, PreparationConfig(
                    box_ratio=box_ratio, min_area=config.min_area,
                    padding=config.padding, feather_px=0, fill_holes=False))
            except PreparationError as exc:
                last_error = exc
        raise last_error
    if mask.shape != image.shape[:2] or mask.dtype != np.uint8:
        raise ValueError("Explicit mask must be uint8 HxW matching the image")
    mask = np.uint8(mask > 0) * 255
    if cv2.countNonZero(mask) < config.min_area:
        raise PreparationError("Foreground is empty or too small")
    crop, crop_mask, bbox = crop_part(image, mask, config.padding)
    return PreparedPart(mask, crop, crop_mask,
                        place_on_background(crop, crop_mask, feather_px=0), bbox)
