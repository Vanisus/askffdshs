"""Uniform-table segmentation, preserving holes; explicit masks override automation."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from .part_preparation import PreparedPart, PreparationError, crop_part, place_on_background, _validate_image


@dataclass(frozen=True)
class ForegroundConfig:
    max_side: int = 1280
    min_area: int = 5000
    padding: int = 35
    background_distance: float = 16.0

    def __post_init__(self):
        if self.max_side < 64 or self.min_area < 1 or self.padding < 0 or self.background_distance <= 0:
            raise ValueError("Invalid foreground configuration")


def prepare_foreground(image, config=None, mask=None):
    config = config or ForegroundConfig()
    _validate_image(image)
    h, w = image.shape[:2]
    if mask is None:
        scale = min(1., config.max_side / max(h, w))
        small = cv2.resize(image, (max(32, round(w*scale)), max(32, round(h*scale))))
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        background = np.median(border, axis=0)
        distance = np.linalg.norm(lab-background, axis=2)
        noise = np.percentile(np.linalg.norm(border-background, axis=1), 95)
        if noise > 35:
            raise PreparationError("Table border is not uniform; provide a mask or improve framing")
        threshold = max(config.background_distance, float(noise)*2)
        candidate = (distance > threshold).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
        if count < 2:
            raise PreparationError("No part found; table/part contrast too low")
        index = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        seed = np.uint8(labels == index)
        if stats[index, cv2.CC_STAT_AREA] / (small.shape[0]*small.shape[1]) < config.min_area / (h*w):
            raise PreparationError("Part is too small")
        if np.any(seed[0]) or np.any(seed[-1]) or np.any(seed[:, 0]) or np.any(seed[:, -1]):
            raise PreparationError("Part touches image border; capture the complete part")
        kernel = np.ones((3, 3), np.uint8)
        certain = cv2.erode(seed, kernel, iterations=2)
        if not certain.any():
            raise PreparationError("Part too thin for automatic masking; provide a mask")
        gc = np.full(seed.shape, cv2.GC_BGD, np.uint8)
        gc[cv2.dilate(seed, kernel, iterations=3) > 0] = cv2.GC_PR_BGD
        gc[seed > 0] = cv2.GC_PR_FGD
        gc[certain > 0] = cv2.GC_FGD
        cv2.grabCut(small, gc, None, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64), 3, cv2.GC_INIT_WITH_MASK)
        fg = np.uint8((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
        if count < 2:
            raise PreparationError("GrabCut returned empty foreground")
        fg = np.uint8(labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))) * 255
        mask = cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        if mask.shape != (h, w) or mask.dtype != np.uint8:
            raise ValueError("Explicit mask must be uint8 HxW matching the image")
        mask = np.uint8(mask > 0)*255
    if cv2.countNonZero(mask) < config.min_area:
        raise PreparationError("Foreground is empty or too small")
    crop, crop_mask, bbox = crop_part(image, mask, config.padding)
    return PreparedPart(mask, crop, crop_mask, place_on_background(crop, crop_mask, feather_px=0), bbox)
