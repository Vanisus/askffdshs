from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class PreparationError(RuntimeError):
    """Не удалось корректно выделить и подготовить деталь."""


@dataclass(frozen=True)
class PreparationConfig:
    box_ratio: float = 0.48
    min_area: int = 5_000
    padding: int = 35
    background_bgr: tuple[int, int, int] = (30, 30, 30)
    feather_px: int = 2


@dataclass(frozen=True)
class PreparedPart:
    source_mask: np.ndarray
    crop: np.ndarray
    crop_mask: np.ndarray
    prepared: np.ndarray
    bbox_xywh: tuple[int, int, int, int]


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image должен быть numpy.ndarray.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Ожидается BGR-изображение OpenCV с shape H×W×3."
        )

    if image.dtype != np.uint8:
        raise ValueError(
            f"Ожидается uint8-изображение, получен dtype={image.dtype}."
        )

    height, width = image.shape[:2]

    if height < 32 or width < 32:
        raise PreparationError(
            "Изображение слишком маленькое для сегментации."
        )


def _validate_config(config: PreparationConfig) -> None:
    if not 0.10 <= config.box_ratio <= 0.95:
        raise ValueError(
            "box_ratio должен быть в диапазоне от 0.10 до 0.95."
        )

    if config.min_area <= 0:
        raise ValueError("min_area должен быть больше нуля.")

    if config.padding < 0:
        raise ValueError("padding не может быть отрицательным.")

    if config.feather_px < 0:
        raise ValueError("feather_px не может быть отрицательным.")

    if len(config.background_bgr) != 3:
        raise ValueError(
            "background_bgr должен содержать три BGR-значения."
        )

    if any(value < 0 or value > 255 for value in config.background_bgr):
        raise ValueError(
            "Каждый канал background_bgr должен быть в диапазоне 0–255."
        )


def segment_part(
    image: np.ndarray,
    config: PreparationConfig,
) -> np.ndarray:
    _validate_image(image)
    _validate_config(config)

    height, width = image.shape[:2]

    box_size = int(min(height, width) * config.box_ratio)

    x = max(0, width // 2 - box_size // 2)
    y = max(0, height // 2 - box_size // 2)

    box_size = min(
        box_size,
        width - x,
        height - y,
    )

    if box_size < 16:
        raise PreparationError(
            "Стартовый прямоугольник GrabCut слишком маленький."
        )

    grabcut_mask = np.zeros((height, width), dtype=np.uint8)

    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(
            image,
            grabcut_mask,
            (x, y, box_size, box_size),
            background_model,
            foreground_model,
            7,
            cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error as exc:
        raise PreparationError(
            f"GrabCut не смог обработать изображение: {exc}"
        ) from exc

    foreground = np.where(
        (grabcut_mask == cv2.GC_FGD)
        | (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        foreground,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= config.min_area
    ]

    if not candidates:
        raise PreparationError(
            "Не найден достаточно большой контур детали. "
            "Проверьте положение детали, фон, box_ratio и min_area."
        )

    main_contour = max(
        candidates,
        key=cv2.contourArea,
    )

    mask = np.zeros((height, width), dtype=np.uint8)

    cv2.drawContours(
        mask,
        [main_contour],
        -1,
        255,
        thickness=cv2.FILLED,
    )
    return mask


def crop_part(
    image: np.ndarray,
    mask: np.ndarray,
    padding: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    if mask.ndim != 2:
        raise ValueError("mask должна быть одноканальной H×W.")

    if mask.shape != image.shape[:2]:
        raise ValueError(
            "Размер mask должен совпадать с H×W исходного изображения."
        )

    points = cv2.findNonZero(mask)

    if points is None:
        raise PreparationError(
            "Маска детали пуста — невозможно сделать crop."
        )

    x, y, width, height = cv2.boundingRect(points)

    image_height, image_width = image.shape[:2]

    x0 = max(0, x - padding)
    y0 = max(0, y - padding)

    x1 = min(image_width, x + width + padding)
    y1 = min(image_height, y + height + padding)

    if x1 <= x0 or y1 <= y0:
        raise PreparationError(
            "После применения padding получился пустой crop."
        )

    crop = image[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1].copy()

    return crop, crop_mask, (x0, y0, x1 - x0, y1 - y0)


def place_on_background(
    crop: np.ndarray,
    crop_mask: np.ndarray,
    background_bgr: tuple[int, int, int] = (30, 30, 30),
    feather_px: int = 2,
) -> np.ndarray:
    if crop_mask.shape != crop.shape[:2]:
        raise ValueError(
            "crop_mask должна иметь тот же H×W, что и crop."
        )

    if feather_px < 0:
        raise ValueError("feather_px не может быть отрицательным.")

    if feather_px > 0:
        kernel_size = feather_px * 2 + 1

        alpha = cv2.GaussianBlur(
            crop_mask,
            (kernel_size, kernel_size),
            0,
        )
    else:
        alpha = crop_mask

    alpha_float = (
        alpha.astype(np.float32) / 255.0
    )[..., None]

    background = np.full(
        crop.shape,
        background_bgr,
        dtype=np.uint8,
    )

    prepared = (
        crop.astype(np.float32) * alpha_float
        + background.astype(np.float32) * (1.0 - alpha_float)
    )

    return np.clip(prepared, 0, 255).astype(np.uint8)


def prepare_part(
    image: np.ndarray,
    config: PreparationConfig | None = None,
) -> PreparedPart:
    active_config = config or PreparationConfig()

    source_mask = segment_part(
        image,
        active_config,
    )

    crop, crop_mask, bbox_xywh = crop_part(
        image,
        source_mask,
        active_config.padding,
    )

    prepared = place_on_background(
        crop,
        crop_mask,
        active_config.background_bgr,
        active_config.feather_px,
    )

    return PreparedPart(
        source_mask=source_mask,
        crop=crop,
        crop_mask=crop_mask,
        prepared=prepared,
        bbox_xywh=bbox_xywh,
    )


def mask_preview(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    if mask.shape != image.shape[:2]:
        raise ValueError(
            "mask должна совпадать с изображением по H×W."
        )

    overlay = np.zeros_like(image)
    overlay[:, :, 1] = mask

    return cv2.addWeighted(
        image,
        0.75,
        overlay,
        0.25,
        0,
    )


def save_preparation_debug(
    source_image: np.ndarray,
    result: PreparedPart,
    output_dir: str | Path,
    prefix: str = "",
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    artifacts = {
        f"{prefix}01_source_mask.png": result.source_mask,
        f"{prefix}02_mask_overlay.png": mask_preview(
            source_image,
            result.source_mask,
        ),
        f"{prefix}03_crop.png": result.crop,
        f"{prefix}04_crop_mask.png": result.crop_mask,
        f"{prefix}05_prepared.png": result.prepared,
    }

    for filename, artifact in artifacts.items():
        output_path = destination / filename

        if not cv2.imwrite(str(output_path), artifact):
            raise RuntimeError(
                f"Не удалось сохранить debug-файл: {output_path}"
            )