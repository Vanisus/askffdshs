from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from services.part_preparation import (
    PreparedPart,
    PreparationConfig,
    PreparationError,
    place_on_background,
    prepare_part,
)


class AlignmentError(RuntimeError):
    """Не удалось надёжно выровнять текущую деталь относительно эталона."""


@dataclass(frozen=True)
class AlignmentConfig:
    preparation: PreparationConfig = PreparationConfig()

    max_features: int = 3_000
    ratio: float = 0.72
    ransac_threshold: float = 3.0

    minimum_matches: int = 6
    minimum_inliers: int = 6
    minimum_inlier_ratio: float = 0.20


@dataclass(frozen=True)
class AlignmentResult:
    reference: PreparedPart
    current: PreparedPart

    aligned_raw: np.ndarray
    aligned_mask: np.ndarray
    aligned_prepared: np.ndarray
    valid_mask: np.ndarray

    matrix_current_crop_to_reference_crop: np.ndarray

    metrics: dict[str, float]
    debug_images: dict[str, np.ndarray]


def _validate_config(config: AlignmentConfig) -> None:
    if config.max_features < 100:
        raise ValueError("max_features должен быть не меньше 100.")

    if not 0.40 <= config.ratio <= 0.95:
        raise ValueError("ratio должен быть в диапазоне 0.40–0.95.")

    if config.ransac_threshold <= 0:
        raise ValueError("ransac_threshold должен быть больше нуля.")

    if config.minimum_matches < 3:
        raise ValueError("minimum_matches должен быть не меньше 3.")

    if config.minimum_inliers < 3:
        raise ValueError("minimum_inliers должен быть не меньше 3.")

    if not 0.0 < config.minimum_inlier_ratio <= 1.0:
        raise ValueError(
            "minimum_inlier_ratio должен быть в диапазоне (0, 1]."
        )


def _to_homogeneous(matrix_2x3: np.ndarray) -> np.ndarray:
    matrix_3x3 = np.eye(3, dtype=np.float64)
    matrix_3x3[:2, :] = matrix_2x3.astype(np.float64)

    return matrix_3x3


def _make_orb_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    return clahe.apply(gray)


def _find_matches(
    reference: PreparedPart,
    current: PreparedPart,
    config: AlignmentConfig,
) -> tuple[
    list[cv2.KeyPoint],
    list[cv2.KeyPoint],
    list[cv2.DMatch],
]:
    orb = cv2.ORB_create(
        nfeatures=config.max_features,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        fastThreshold=7,
    )

    reference_keypoints, reference_descriptors = (
        orb.detectAndCompute(
            _make_orb_gray(reference.crop),
            reference.crop_mask,
        )
    )

    current_keypoints, current_descriptors = (
        orb.detectAndCompute(
            _make_orb_gray(current.crop),
            current.crop_mask,
        )
    )

    if reference_descriptors is None or current_descriptors is None:
        raise AlignmentError(
            "Недостаточно ORB-признаков внутри маски детали. "
            "Проверьте фокус, освещение, разрешение и сегментацию."
        )

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False,
    )

    pairs = matcher.knnMatch(
        current_descriptors,
        reference_descriptors,
        k=2,
    )

    ratio_matches: list[cv2.DMatch] = []

    for pair in pairs:
        if len(pair) != 2:
            continue

        best, second = pair

        if best.distance < config.ratio * second.distance:
            ratio_matches.append(best)

    ratio_matches.sort(key=lambda match: match.distance)

    matches: list[cv2.DMatch] = []
    used_current: set[int] = set()
    used_reference: set[int] = set()

    for match in ratio_matches:
        if match.queryIdx in used_current:
            continue

        if match.trainIdx in used_reference:
            continue

        matches.append(match)
        used_current.add(match.queryIdx)
        used_reference.add(match.trainIdx)

    if len(matches) < config.minimum_matches:
        raise AlignmentError(
            f"После Lowe ratio test осталось {len(matches)} ORB-сопоставлений; "
            f"необходимо минимум {config.minimum_matches}. "
            "Можно попробовать ratio=0.76–0.78 или повысить max_features."
        )

    return reference_keypoints, current_keypoints, matches


def _make_valid_mask(
    reference_mask: np.ndarray,
    aligned_current_mask: np.ndarray,
) -> np.ndarray:
    valid_mask = cv2.bitwise_and(
        reference_mask,
        aligned_current_mask,
    )

    kernel = np.ones((3, 3), dtype=np.uint8)

    return cv2.erode(
        valid_mask,
        kernel,
        iterations=1,
    )


def _mask_preview(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = np.zeros_like(image)
    overlay[:, :, 1] = mask

    return cv2.addWeighted(
        image,
        0.75,
        overlay,
        0.25,
        0,
    )


def _difference_preview(
    reference_prepared: np.ndarray,
    aligned_prepared: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    difference = cv2.absdiff(
        reference_prepared,
        aligned_prepared,
    )

    difference[valid_mask == 0] = 0

    return difference


def align_prepared_parts(
    reference: PreparedPart,
    current: PreparedPart,
    config: AlignmentConfig | None = None,
    include_debug: bool = True,
) -> AlignmentResult:
    active_config = config or AlignmentConfig()
    _validate_config(active_config)

    (
        reference_keypoints,
        current_keypoints,
        matches,
    ) = _find_matches(
        reference=reference,
        current=current,
        config=active_config,
    )

    source_points = np.float32(
        [
            current_keypoints[match.queryIdx].pt
            for match in matches
        ]
    ).reshape(-1, 1, 2)

    target_points = np.float32(
        [
            reference_keypoints[match.trainIdx].pt
            for match in matches
        ]
    ).reshape(-1, 1, 2)

    matrix_2x3, inlier_mask = cv2.estimateAffinePartial2D(
        source_points,
        target_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=active_config.ransac_threshold,
        maxIters=5_000,
        confidence=0.995,
        refineIters=20,
    )

    if matrix_2x3 is None or inlier_mask is None:
        raise AlignmentError(
            "RANSAC не смог оценить similarity transform current → reference."
        )

    inlier_flags = inlier_mask.ravel().astype(bool)
    inliers = int(inlier_flags.sum())
    inlier_ratio = inliers / len(matches)

    if inliers < active_config.minimum_inliers:
        raise AlignmentError(
            f"Недостаточно RANSAC-inliers: {inliers}/{len(matches)}; "
            f"минимум: {active_config.minimum_inliers}."
        )

    if inlier_ratio < active_config.minimum_inlier_ratio:
        raise AlignmentError(
            f"Ненадёжный alignment: {inliers}/{len(matches)} "
            f"RANSAC-inliers ({inlier_ratio:.1%}); минимум: "
            f"{active_config.minimum_inlier_ratio:.1%}."
        )

    projected_points = cv2.transform(
        source_points,
        matrix_2x3,
    ).reshape(-1, 2)

    target_points_flat = target_points.reshape(-1, 2)

    reprojection_errors = np.linalg.norm(
        projected_points - target_points_flat,
        axis=1,
    )

    reference_height, reference_width = reference.crop.shape[:2]

    aligned_raw = cv2.warpAffine(
        current.crop,
        matrix_2x3,
        (reference_width, reference_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=active_config.preparation.background_bgr,
    )

    aligned_mask = cv2.warpAffine(
        current.crop_mask,
        matrix_2x3,
        (reference_width, reference_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    aligned_prepared = place_on_background(
        aligned_raw,
        aligned_mask,
        background_bgr=active_config.preparation.background_bgr,
        feather_px=active_config.preparation.feather_px,
    )

    valid_mask = _make_valid_mask(
        reference.crop_mask,
        aligned_mask,
    )

    reference_area = max(
        1,
        cv2.countNonZero(reference.crop_mask),
    )

    overlap_ratio = (
        cv2.countNonZero(valid_mask) / reference_area
    )

    matrix_3x3 = _to_homogeneous(matrix_2x3)

    metrics = {
        "matches": float(len(matches)),
        "inliers": float(inliers),
        "inlier_ratio": float(inlier_ratio),
        "median_reprojection_error_px": float(
            np.median(reprojection_errors[inlier_flags])
        ),
        "rotation_deg": float(
            np.degrees(
                np.arctan2(
                    matrix_2x3[1, 0],
                    matrix_2x3[0, 0],
                )
            )
        ),
        "scale": float(
            np.hypot(
                matrix_2x3[0, 0],
                matrix_2x3[1, 0],
            )
        ),
        "valid_overlap_ratio": float(overlap_ratio),
    }

    debug_images: dict[str, np.ndarray] = {}

    if include_debug:
        inlier_matches = cv2.drawMatches(
            current.crop,
            current_keypoints,
            reference.crop,
            reference_keypoints,
            matches,
            None,
            matchesMask=inlier_mask.ravel().tolist(),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        debug_images = {
            "01_reference_prepared.png": reference.prepared,
            "02_current_prepared_before_alignment.png": current.prepared,
            "03_orb_inlier_matches.png": inlier_matches,
            "04_current_aligned.png": aligned_prepared,
            "05_alignment_overlay.png": cv2.addWeighted(
                reference.prepared,
                0.5,
                aligned_prepared,
                0.5,
                0,
            ),
            "06_valid_mask.png": valid_mask,
            "07_valid_mask_overlay.png": _mask_preview(
                reference.prepared,
                valid_mask,
            ),
            "08_alignment_difference.png": _difference_preview(
                reference.prepared,
                aligned_prepared,
                valid_mask,
            ),
        }

    return AlignmentResult(
        reference=reference,
        current=current,
        aligned_raw=aligned_raw,
        aligned_mask=aligned_mask,
        aligned_prepared=aligned_prepared,
        valid_mask=valid_mask,
        matrix_current_crop_to_reference_crop=matrix_3x3,
        metrics=metrics,
        debug_images=debug_images,
    )


def align_images(
    reference_image: np.ndarray,
    current_image: np.ndarray,
    preparation_config: PreparationConfig | None = None,
    alignment_config: AlignmentConfig | None = None,
    include_debug: bool = True,
) -> AlignmentResult:
    if alignment_config is not None and preparation_config is not None:
        if alignment_config.preparation != preparation_config:
            raise ValueError(
                "Переданы разные preparation_config и "
                "alignment_config.preparation. Используйте единые настройки."
            )

    active_preparation = (
        preparation_config
        or (
            alignment_config.preparation
            if alignment_config is not None
            else PreparationConfig()
        )
    )

    active_alignment = alignment_config or AlignmentConfig(
        preparation=active_preparation
    )

    try:
        reference = prepare_part(
            reference_image,
            active_preparation,
        )

        current = prepare_part(
            current_image,
            active_preparation,
        )
    except PreparationError as exc:
        raise AlignmentError(
            f"Ошибка подготовки изображения: {exc}"
        ) from exc

    return align_prepared_parts(
        reference=reference,
        current=current,
        config=active_alignment,
        include_debug=include_debug,
    )


def save_alignment_debug(
    result: AlignmentResult,
    output_dir: str | Path,
    prefix: str = "",
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    for filename, image in result.debug_images.items():
        output_path = destination / f"{prefix}{filename}"

        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(
                f"Не удалось сохранить debug-файл: {output_path}"
            )