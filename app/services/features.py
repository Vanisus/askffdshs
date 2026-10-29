"""Frozen pretrained encoders. No fine tuning and no per-part model copies."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import threading

import cv2
import numpy as np
import torch
import torch.nn.functional as F

DINO_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"


@dataclass(frozen=True)
class FeatureConfig:
    backbone: str = "dinov2_vits14"
    tile_size: int = 448
    overlap: int = 112
    local_radius: int = 1

    def __post_init__(self):
        if self.backbone not in {"dinov2_vits14", "dinov2_vitb14", "resnet18"}:
            raise ValueError("Unsupported pretrained backbone")
        if self.tile_size < 56 or self.tile_size % 14:
            raise ValueError("tile_size must be >=56 and divisible by 14")
        if not 0 <= self.overlap < self.tile_size or not 0 <= self.local_radius <= 3:
            raise ValueError("Invalid overlap or local_radius")

    @property
    def signature(self):
        payload = {**asdict(self), "format": 1, "dino_revision": DINO_REVISION,
                   "resnet_weights": "IMAGENET1K_V1", "torch": torch.__version__}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def tile_grid(shape, size, overlap):
    """Cover every pixel, including bottom/right edges, without resizing a 4K frame."""
    def starts(length):
        return sorted(set(list(range(0, max(1, length - size + 1), size - overlap))
                          + [max(0, length - size)]))
    return [(x, y, min(size, shape[1] - x), min(size, shape[0] - y))
            for y in starts(shape[0]) for x in starts(shape[1])]


class FrozenEncoder:
    def __init__(self, config: FeatureConfig, device="cuda", hub_dir=None):
        self.config = config
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; install CUDA PyTorch or use --device cpu")
        if config.backbone == "resnet18":
            from torchvision.models import ResNet18_Weights, resnet18
            net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.model = torch.nn.Sequential(*list(net.children())[:7])
        else:
            self.model = torch.hub.load(
                str(hub_dir) if hub_dir else f"facebookresearch/dinov2:{DINO_REVISION}",
                config.backbone, source="local" if hub_dir else "github",
                pretrained=True, trust_repo=True, skip_validation=True,
            )
        self.model.eval().requires_grad_(False).to(self.device)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self._lock = threading.RLock()
        self.mean = torch.tensor([.485, .456, .406], device=self.device)[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device=self.device)[None, :, None, None]

    @torch.inference_mode()
    def extract(self, image, mask):
        h, w = image.shape[:2]
        size = self.config.tile_size
        canvas = np.full((size, size, 3), 30, dtype=np.uint8)
        canvas[:h, :w] = image
        canvas[:h, :w][mask == 0] = 30
        x = torch.from_numpy(canvas[:, :, ::-1].copy()).to(self.device)
        x = x.permute(2, 0, 1).unsqueeze(0).float() / 255
        with self._lock:
            x = (x - self.mean) / self.std
            if self.config.backbone == "resnet18":
                features = self.model(x)
            else:
                tokens = self.model.forward_features(x)["x_norm_patchtokens"]
                side = size // 14
                features = tokens.transpose(1, 2).reshape(1, -1, side, side)
            features = F.normalize(features.float(), dim=1)
        return features[0].cpu().numpy().astype(np.float16)


def feature_support(mask, grid_shape, tile_size):
    canvas = np.zeros((tile_size, tile_size), np.float32)
    canvas[:mask.shape[0], :mask.shape[1]] = mask > 0
    return cv2.resize(canvas, (grid_shape[1], grid_shape[0]), interpolation=cv2.INTER_AREA) >= .98


def local_distance(query, reference, reference_support, radius):
    """Cosine NN within an aligned spatial neighbourhood, not anywhere on the part."""
    q = query.astype(np.float32)
    r = reference.astype(np.float32)
    q /= np.maximum(np.linalg.norm(q, axis=0, keepdims=True), 1e-8)
    r /= np.maximum(np.linalg.norm(r, axis=0, keepdims=True), 1e-8)
    _, h, w = q.shape
    scores = np.full((h, w), np.inf, np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y0, y1 = max(0, -dy), min(h, h - dy)
            x0, x1 = max(0, -dx), min(w, w - dx)
            similarity = np.sum(q[:, y0:y1, x0:x1] * r[:, y0+dy:y1+dy, x0+dx:x1+dx], axis=0)
            distance = np.clip(1 - similarity, 0, 2)
            distance[~reference_support[y0+dy:y1+dy, x0+dx:x1+dx]] = np.inf
            scores[y0:y1, x0:x1] = np.minimum(scores[y0:y1, x0:x1], distance)
    return scores
