"""Fixed-scale anomaly visualization and original-frame instance coordinates."""
from __future__ import annotations
import cv2
import numpy as np


def components(score, domain, threshold, min_area, inverse, bbox, source_shape):
    binary = np.uint8((score > threshold) & domain)*255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    detections = []
    h, w = source_shape
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        component = np.uint8(labels == i)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            points = cv2.perspectiveTransform(contour.astype(np.float32), inverse).reshape(-1, 2)
            points += np.array(bbox[:2], np.float32)
            points[:, 0] = np.clip(points[:, 0], 0, w-1)
            points[:, 1] = np.clip(points[:, 1], 0, h-1)
            polygons.append(points.tolist())
        all_points = np.concatenate([np.asarray(p) for p in polygons])
        x0, y0 = np.floor(all_points.min(axis=0)).astype(int)
        x1, y1 = np.ceil(all_points.max(axis=0)).astype(int) + 1
        yy, xx = np.unravel_index(np.argmax(np.where(component, score, -np.inf)), score.shape)
        peak = cv2.perspectiveTransform(np.float32([[[xx, yy]]]), inverse)[0, 0] + np.array(bbox[:2])
        detections.append({
            "bbox_xyxy": [int(x0), int(y0), min(w, int(x1)), min(h, int(y1))],
            "polygons_xy": polygons,
            "peak_xy": [float(np.clip(peak[0], 0, w-1)), float(np.clip(peak[1], 0, h-1))],
            "area_canonical_px": int(stats[i, cv2.CC_STAT_AREA]),
            "score_max": float(score[component > 0].max()),
            "score_mean": float(score[component > 0].mean()),
        })
    return sorted(detections, key=lambda d: d["score_max"], reverse=True)


def to_source(array, inverse, bbox, source_shape, nearest=False):
    x, y, w, h = bbox
    crop = cv2.warpPerspective(array, inverse, (w, h),
                               flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)
    full = np.zeros(source_shape, dtype=array.dtype)
    full[y:y+h, x:x+w] = crop
    return full


def render(image, scores, mask, detections, display_max):
    normalized = np.clip(scores / max(display_max, 1e-6), 0, 1)
    colors = cv2.applyColorMap(np.uint8(normalized*255), cv2.COLORMAP_TURBO)
    out = image.copy()
    selected = (mask > 0) & (scores > 0)
    out[selected] = cv2.addWeighted(image, .6, colors, .4, 0)[selected]
    for det in detections:
        x0, y0, x1, y1 = det["bbox_xyxy"]
        cv2.rectangle(out, (x0, y0), (x1-1, y1-1), (0, 0, 255), 2)
    return out
