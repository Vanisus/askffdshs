"""Registration-aware tiled few-shot anomaly detection; no per-part training."""
from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .catalog import Catalog, decode_image, load_part, save_part, write_image
from .features import FeatureConfig, FrozenEncoder, feature_support, local_distance, tile_grid
from .foreground import ForegroundConfig, prepare_foreground
from .part_alignment import AlignmentConfig, AlignmentError, align_prepared_parts
from .part_preparation import PreparationConfig
from .postprocessing import components, render, to_source


@dataclass(frozen=True)
class InspectionConfig:
    threshold: float = .15
    reference_quantile: float = .25
    min_area_px: int = 30
    edge_margin_px: int = 4
    min_overlap: float = .85
    max_reprojection_px: float = 3.
    min_inlier_spread: float = .02
    min_scale: float = .8
    max_scale: float = 1.25
    min_comparable_fraction: float = .90
    max_shape_disagreement: float = .03
    calibration_quantile: float = .995
    calibration_margin: float = .02

    def __post_init__(self):
        if not 0 < self.threshold <= 2 or not 0 <= self.reference_quantile <= 1:
            raise ValueError("Invalid threshold or reference_quantile")
        if self.min_area_px < 1 or self.edge_margin_px < 0 or self.max_reprojection_px <= 0:
            raise ValueError("Invalid area, margin or registration error")
        for v in (self.min_overlap, self.min_inlier_spread, self.min_comparable_fraction,
                  self.max_shape_disagreement, self.calibration_quantile):
            if not 0 < v <= 1:
                raise ValueError("Fractions must lie in (0,1]")
        if not 0 < self.min_scale <= self.max_scale or self.calibration_margin < 0:
            raise ValueError("Invalid scale or calibration margin")


def safe_quantile(values, quantile):
    """Lower order statistic over available references, ignoring unsupported cells."""
    stack = np.stack(values)
    count = np.isfinite(stack).sum(axis=0)
    stack = np.sort(np.where(np.isfinite(stack), stack, np.inf), axis=0)
    index = np.floor(np.maximum(count-1, 0)*quantile).astype(int)
    fused = np.take_along_axis(stack, index[None], axis=0)[0]
    return np.where(count > 0, fused, 0).astype(np.float32), count


class Inspector:
    def __init__(self, root, feature_config=None, foreground_config=None,
                 inspection_config=None, device="cuda", encoder=None, hub_dir=None):
        self.catalog = Catalog(root)
        self.features = feature_config or FeatureConfig()
        self.foreground = foreground_config or ForegroundConfig()
        self.config = inspection_config or InspectionConfig()
        self._encoder = encoder
        self.device = device
        self.hub_dir = hub_dir
        self._encoder_lock = threading.Lock()

    @property
    def encoder(self):
        with self._encoder_lock:
            if self._encoder is None:
                self._encoder = FrozenEncoder(self.features, self.device, self.hub_dir)
        return self._encoder

    def add_reference(self, part_id, image, mask=None):
        self.catalog.part_path(part_id)
        return self.catalog.add_reference(part_id, prepare_foreground(image, self.foreground, mask))

    def _align(self, reference, current):
        # First attempt is strict; texture-poor surfaces (e.g. polished metal)
        # can leave too few stable ORB pairs. Retry once with a relaxed Lowe
        # ratio and inlier counts. The geometric gates below are unchanged
        # and remain the actual safeguard against a wrong transform.
        attempts = (AlignmentConfig(preparation=PreparationConfig(feather_px=0), max_features=8000,
                                     minimum_inliers=12, minimum_inlier_ratio=.4),
                    AlignmentConfig(preparation=PreparationConfig(feather_px=0), max_features=12000,
                                    ratio=.8, ransac_threshold=4., minimum_inliers=8,
                                    minimum_inlier_ratio=.25))
        result = None
        errors = []
        for config in attempts:
            try:
                result = align_prepared_parts(reference, current, config, include_debug=False)
                break
            except AlignmentError as exc:
                errors.append(exc)
        if result is None:
            raise errors[-1]
        m = result.metrics
        if m["valid_overlap_ratio"] < self.config.min_overlap:
            raise AlignmentError("Insufficient foreground overlap; do not classify this image")
        if m["median_reprojection_error_px"] > self.config.max_reprojection_px:
            raise AlignmentError("Registration error exceeds tolerance")
        if not self.config.min_scale <= m["scale"] <= self.config.max_scale:
            raise AlignmentError("Unexpected part scale")
        if m["inlier_spread"] < self.config.min_inlier_spread:
            raise AlignmentError("Inlier keypoints are too spatially concentrated")
        return result

    def build(self, part_id):
        start = time.perf_counter()
        with self.catalog.lock(part_id):
            state = self.catalog.state(part_id)
            refs = self.catalog.references(part_id)
            if not refs:
                raise ValueError("Upload at least one reference")
            anchor = load_part(refs[0]["path"])
            aligned = []
            registration = []
            for i, row in enumerate(refs):
                ref = load_part(row["path"])
                if i == 0:
                    aligned.append((ref.prepared, ref.crop_mask))
                    registration.append({"id": row["id"], "anchor": True})
                else:
                    result = self._align(anchor, ref)
                    aligned.append((result.aligned_prepared, result.aligned_mask))
                    registration.append({"id": row["id"], **result.metrics})
            profile_id = uuid.uuid4().hex
            profiles = self.catalog.part_path(part_id) / "profiles"
            profiles.mkdir(exist_ok=True)
            temp = profiles / (".building-" + profile_id)
            final = profiles / profile_id
            temp.mkdir()
            try:
                save_part(temp / "anchor", anchor)
                tiles = []
                normal_scores = []
                for tile in tile_grid(anchor.crop.shape[:2], self.features.tile_size, self.features.overlap):
                    x, y, w, h = tile
                    if not anchor.crop_mask[y:y+h, x:x+w].any():
                        continue
                    keys = {}
                    embeddings = []
                    supports = []
                    for i, (image, mask) in enumerate(aligned):
                        patch_mask = mask[y:y+h, x:x+w]
                        f = self.encoder.extract(image[y:y+h, x:x+w], patch_mask)
                        s = feature_support(patch_mask, f.shape[1:], self.features.tile_size)
                        keys[f"f{i}"] = f
                        keys[f"s{i}"] = s
                        embeddings.append(f)
                        supports.append(s)
                    if len(refs) >= 2:
                        for i, f in enumerate(embeddings):
                            distances = [local_distance(f, other, supports[j], self.features.local_radius)
                                         for j, other in enumerate(embeddings) if i != j]
                            fused, count = safe_quantile(distances, self.config.reference_quantile)
                            values = fused[(count > 0) & supports[i]]
                            if values.size:
                                normal_scores.append(values[::max(1, values.size // 2048)])
                    np.savez_compressed(temp / f"tile_{len(tiles)}.npz", **keys)
                    tiles.append(tile)
                if not tiles:
                    raise ValueError("No foreground tiles")
                samples = np.concatenate(normal_scores) if normal_scores else np.empty(0)
                calibration = {"method": "reference_leave_one_out" if samples.size else "uncalibrated",
                               "sample_count": int(samples.size), "requires_held_out_validation": True}
                threshold = self.config.threshold
                if samples.size:
                    threshold = min(2., max(.05, float(np.quantile(samples, self.config.calibration_quantile))
                                             + self.config.calibration_margin))
                metadata = {"version": 1, "id": profile_id, "generation": state["generation"],
                            "features": asdict(self.features), "signature": self.features.signature,
                            "foreground": asdict(self.foreground), "inspection": asdict(self.config),
                            "threshold": threshold, "calibration": calibration,
                            "reference_ids": [r["id"] for r in refs], "registration": registration,
                            "tiles": tiles, "build_seconds": time.perf_counter()-start}
                (temp / "profile.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                temp.rename(final)
                self.catalog.activate(part_id, state["generation"], final)
            except Exception:
                shutil.rmtree(temp, ignore_errors=True)
                raise
        return metadata

    def _profile(self, part_id):
        state = self.catalog.state(part_id)
        if not state["active_profile"]:
            raise ValueError("No active profile; upload a reference to build it")
        path = Path(state["active_profile"])
        meta = json.loads((path / "profile.json").read_text())
        if meta["generation"] != state["generation"]:
            raise ValueError("References changed; upload a reference again to rebuild the profile")
        if meta["signature"] != self.features.signature:
            raise ValueError("Feature configuration changed; upload a reference again to rebuild the profile")
        if meta["foreground"] != asdict(self.foreground) or meta["inspection"] != asdict(self.config):
            raise ValueError("Inspection/preparation configuration changed; upload a reference again to rebuild the profile")
        return path, meta

    def inspect(self, part_id, image, mask=None):
        start = time.perf_counter()
        path, meta = self._profile(part_id)
        anchor = load_part(path / "anchor")
        current = prepare_foreground(image, self.foreground, mask)
        result = self._align(anchor, current)
        ah, aw = anchor.crop_mask.shape
        interior = (anchor.crop_mask > 0) & (result.aligned_mask > 0)
        if self.config.edge_margin_px:
            k = 2*self.config.edge_margin_px+1
            interior = cv2.erode(np.uint8(interior), np.ones((k, k), np.uint8)) > 0
        score_sum = np.zeros((ah, aw), np.float32)
        weight_sum = np.zeros_like(score_sum)
        votes_sum = np.zeros_like(score_sum)
        evidence_sum = np.zeros_like(score_sum)
        ref_stats = [{"reference_id": rid, "sum": 0., "count": 0, "max": 0.} for rid in meta["reference_ids"]]
        threshold = meta["threshold"]
        for index, (x, y, w, h) in enumerate(meta["tiles"]):
            qmask = result.aligned_mask[y:y+h, x:x+w]
            q = self.encoder.extract(result.aligned_prepared[y:y+h, x:x+w], qmask)
            support = feature_support(qmask, q.shape[1:], self.features.tile_size)
            distances = []
            with np.load(path / f"tile_{index}.npz", allow_pickle=False) as data:
                for i in range(len(meta["reference_ids"])):
                    d = local_distance(q, data[f"f{i}"], data[f"s{i}"], self.features.local_radius)
                    d[~support] = np.inf
                    values = d[np.isfinite(d)]
                    if values.size:
                        ref_stats[i]["sum"] += float(values.sum())
                        ref_stats[i]["count"] += int(values.size)
                        ref_stats[i]["max"] = max(ref_stats[i]["max"], float(values.max()))
                    distances.append(d)
            fused, count = safe_quantile(distances, self.config.reference_quantile)
            valid = (count > 0) & support
            votes = np.sum([(d > threshold) & np.isfinite(d) for d in distances], axis=0)
            def expand(v, h=h, w=w):
                return cv2.resize(np.float32(v), (self.features.tile_size, self.features.tile_size),
                                  interpolation=cv2.INTER_LINEAR)[:h, :w]
            denominator = expand(valid)
            supported = denominator >= .99
            denominator = np.maximum(denominator, 1e-8)
            heat = expand(np.where(valid, fused, 0))/denominator
            vmap = expand(np.where(valid, votes, 0))/denominator
            emap = expand(np.where(valid, count, 0))/denominator
            blend = np.outer(np.hanning(self.features.tile_size), np.hanning(self.features.tile_size))[:h, :w]
            blend = np.maximum(blend, .01)*supported*interior[y:y+h, x:x+w]
            score_sum[y:y+h, x:x+w] += heat*blend
            votes_sum[y:y+h, x:x+w] += vmap*blend
            evidence_sum[y:y+h, x:x+w] += emap*blend
            weight_sum[y:y+h, x:x+w] += blend
        domain = (weight_sum > 0) & interior
        if not domain.any():
            raise ValueError("No comparable foreground patches; use smaller tiles or better masks")
        scores = np.divide(score_sum, weight_sum, out=np.zeros_like(score_sum), where=weight_sum > 0)
        votes = np.divide(votes_sum, weight_sum, out=np.zeros_like(score_sum), where=weight_sum > 0)
        evidence = np.divide(evidence_sum, weight_sum, out=np.zeros_like(score_sum), where=weight_sum > 0)
        inverse = np.linalg.inv(result.matrix_current_crop_to_reference_crop)
        detections = components(scores, domain, threshold, self.config.min_area_px, inverse,
                                current.bbox_xywh, image.shape[:2])
        for det in detections:
            p = np.float32([[[*det["peak_xy"]]]])
            p[0, 0] -= np.array(current.bbox_xywh[:2])
            px, py = cv2.perspectiveTransform(p, result.matrix_current_crop_to_reference_crop)[0, 0]
            ix, iy = int(np.clip(round(px), 0, aw-1)), int(np.clip(round(py), 0, ah-1))
            det["reference_votes_at_peak"] = float(votes[iy, ix])
            det["references_compared_at_peak"] = float(evidence[iy, ix])
        comparable_fraction = float(domain.sum()/max(1, interior.sum()))
        union = (anchor.crop_mask > 0) | (result.aligned_mask > 0)
        shape_disagreement = float(np.count_nonzero((anchor.crop_mask > 0) ^ (result.aligned_mask > 0))/max(1, union.sum()))
        warnings = []
        if meta["calibration"]["method"] == "uncalibrated":
            warnings.append("Threshold is uncalibrated; upload another normal reference to calibrate it")
        if comparable_fraction < self.config.min_comparable_fraction:
            warnings.append("Insufficient inspected surface coverage")
        if shape_disagreement > self.config.max_shape_disagreement:
            warnings.append("Silhouette discrepancy: possible geometry defect or segmentation error")
        for stat in ref_stats:
            stat["score_mean"] = stat.pop("sum") / max(1, stat["count"])
            stat["score_max"] = stat.pop("max")
        source_scores = to_source(scores, inverse, current.bbox_xywh, image.shape[:2])
        source_mask = to_source(np.uint8(domain)*255, inverse, current.bbox_xywh, image.shape[:2], True)
        source_mask[current.source_mask == 0] = 0
        source_scores[source_mask == 0] = 0
        payload = {"part_id": part_id, "profile_id": meta["id"], "reference_count": len(meta["reference_ids"]),
                   "decision": "inconclusive" if warnings else ("defect_candidates" if detections else "no_candidates"),
                   "production_acceptance": "not_validated", "warnings": warnings,
                   "coordinate_frame": "original_exif_oriented_image", "bbox_convention": "xyxy_half_open",
                   "image_size_wh": [image.shape[1], image.shape[0]], "threshold": threshold,
                   "score_semantics": "cosine_distance_not_probability", "image_score": float(scores[domain].max()),
                   "calibration": meta["calibration"], "comparable_fraction": comparable_fraction,
                   "shape_disagreement": shape_disagreement, "alignment": result.metrics,
                   "matrix_current_crop_to_anchor_crop": result.matrix_current_crop_to_reference_crop.tolist(),
                   "current_crop_bbox_xywh": list(current.bbox_xywh), "reference_comparisons": ref_stats,
                   "defects": detections, "elapsed_seconds": time.perf_counter()-start}
        return payload, source_scores, source_mask, render(image, source_scores, source_mask, detections, threshold*2)

    def calibrate(self, part_id, normal_paths, masks=None):
        """Calibrate on held-out normal captures. This does not validate defect recall."""
        paths = [Path(p) for p in normal_paths]
        if not paths:
            raise ValueError("Provide held-out normal images")
        masks = list(masks) if masks is not None else [None]*len(paths)
        if len(masks) != len(paths) or any(m is not None and not isinstance(m, np.ndarray) for m in masks):
            raise ValueError("Provide one optional mask per calibration image")
        with self.catalog.lock(part_id):
            path, meta = self._profile(part_id)
            samples = []
            reference_pixels = []
            for row in self.catalog.references(part_id):
                ref = load_part(row["path"])
                reference_pixels.append((ref.crop, ref.crop_mask))
            for p, m in zip(paths, masks):
                image = decode_image(p.read_bytes())
                prepared = prepare_foreground(image, self.foreground, m)
                if any(prepared.crop.shape == crop.shape and np.array_equal(prepared.crop, crop)
                       and np.array_equal(prepared.crop_mask, mask) for crop, mask in reference_pixels):
                    raise ValueError("Calibration images must not duplicate references")
                result, scores, domain, _ = self.inspect(part_id, image, m)
                if (result["comparable_fraction"] < self.config.min_comparable_fraction
                        or result["shape_disagreement"] > self.config.max_shape_disagreement):
                    raise ValueError("Calibration image has inadequate coverage or silhouette mismatch")
                values = scores[domain > 0]
                samples.append(values[::max(1, values.size // 100000)])
            values = np.concatenate(samples)
            threshold = min(2., max(.01, float(np.quantile(values, self.config.calibration_quantile))
                                    + self.config.calibration_margin))
            new_id = uuid.uuid4().hex
            final = path.parent / new_id
            temp = path.parent / (".building-" + new_id)
            try:
                shutil.copytree(path, temp)
                meta = {**meta, "id": new_id, "threshold": threshold, "calibration": {
                    "method": "held_out_normal_images", "image_count": len(paths),
                    "sample_count": int(values.size), "quantile": self.config.calibration_quantile,
                    "requires_held_out_validation": True,
                    "note": "Normal-only calibration estimates false positives, not defect recall"}}
                (temp / "profile.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                temp.rename(final)
                self.catalog.activate(part_id, meta["generation"], final)
            except Exception:
                shutil.rmtree(temp, ignore_errors=True)
                raise
        return meta

    def inspect_and_save(self, part_id, image, output_dir, mask=None):
        payload, scores, domain, overlay = self.inspect(part_id, image, mask)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "heatmap.npy", scores, allow_pickle=False)
        write_image(out / "inspection_mask.png", domain)
        write_image(out / "overlay.png", overlay)
        result_temp = out / ".result.json.tmp"
        result_temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        result_temp.replace(out / "result.json")
        return payload
