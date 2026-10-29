import cv2
import numpy as np
import pytest

from app.services.features import FeatureConfig
from app.services.foreground import ForegroundConfig
from app.services.inspection import InspectionConfig, Inspector


class FixtureEncoder:
    """Deterministic test double, deliberately NOT a substitute ML backend."""
    def __init__(self, config):
        self.config = config
        self.calls = 0

    def extract(self, image, mask):
        self.calls += 1
        size = self.config.tile_size
        canvas = np.full((size, size, 3), 30, np.float32)
        canvas[:image.shape[0], :image.shape[1]] = image
        canvas[:image.shape[0], :image.shape[1]][mask == 0] = 30
        f = cv2.resize(canvas/255, (size//14, size//14), interpolation=cv2.INTER_AREA)
        f = np.concatenate([f, np.full((*f.shape[:2], 1), .2, np.float32)], axis=2).transpose(2, 0, 1)
        return (f/np.maximum(np.linalg.norm(f, axis=0, keepdims=True), 1e-8)).astype(np.float16)


@pytest.fixture
def scene():
    # Artificial geometry for functional assertions only, not an accuracy dataset.
    rng = np.random.default_rng(123)
    image = np.full((480, 640, 3), 35, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(mask, (135, 85), (505, 395), 255, -1)
    cv2.circle(mask, (230, 160), 25, 0, -1)
    image[mask > 0] = (180, 180, 180)
    for _ in range(180):
        x, y = int(rng.integers(160, 480)), int(rng.integers(105, 375))
        c = int(rng.integers(70, 155))
        cv2.circle(image, (x, y), int(rng.integers(2, 5)), (c, c, c), -1)
    image[mask == 0] = 35
    return image, mask


@pytest.fixture
def service(tmp_path):
    config = FeatureConfig(tile_size=224, overlap=56)
    return Inspector(tmp_path, config, ForegroundConfig(min_area=1000, padding=14),
                     InspectionConfig(threshold=.01, min_area_px=5), device="cpu", encoder=FixtureEncoder(config))
