#!/usr/bin/env python3
"""
CLI для подготовки эталона и alignment текущей детали.

1. Подготовить эталон:

python main.py prepare-reference `
  --input ./lib/1.png `
  --reference-dir ./data/parts/part_001/reference `
  --debug-dir ./out/reference_debug

2. Выровнять текущую деталь относительно сохранённого эталона:

python main.py align `
  --reference-dir ./data/parts/part_001/reference `
  --input ./lib/2.png `
  --output ./out/aligned.png `
  --debug-dir ./out/alignment_debug
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from services.part_alignment import (
    AlignmentConfig,
    AlignmentError,
    align_prepared_parts,
    save_alignment_debug,
)
from services.part_preparation import (
    PreparedPart,
    PreparationConfig,
    PreparationError,
    mask_preview,
    prepare_part,
    save_preparation_debug,
)


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(
            f"Невозможно прочитать изображение: {path}"
        )

    return image


def read_mask(path: str | Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(
            f"Невозможно прочитать маску: {path}"
        )

    return mask


def save_image(path: str | Path, image: np.ndarray) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(
            f"Невозможно записать изображение: {output_path}"
        )


def preparation_config_to_dict(
    config: PreparationConfig,
) -> dict[str, object]:
    return {
        "box_ratio": config.box_ratio,
        "min_area": config.min_area,
        "padding": config.padding,
        "background_bgr": list(config.background_bgr),
        "feather_px": config.feather_px,
    }


def preparation_config_from_dict(
    payload: dict[str, object],
) -> PreparationConfig:
    try:
        background = payload["background_bgr"]

        if not isinstance(background, list) or len(background) != 3:
            raise ValueError(
                "metadata.background_bgr должен быть списком из 3 чисел."
            )

        return PreparationConfig(
            box_ratio=float(payload["box_ratio"]),
            min_area=int(payload["min_area"]),
            padding=int(payload["padding"]),
            background_bgr=(
                int(background[0]),
                int(background[1]),
                int(background[2]),
            ),
            feather_px=int(payload["feather_px"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "metadata.json эталона имеет неверный формат."
        ) from exc


def save_reference_package(
    reference: PreparedPart,
    config: PreparationConfig,
    reference_dir: str | Path,
) -> None:
    destination = Path(reference_dir)
    destination.mkdir(parents=True, exist_ok=True)

    save_image(destination / "crop.png", reference.crop)
    save_image(destination / "mask.png", reference.crop_mask)
    save_image(destination / "prepared.png", reference.prepared)

    metadata = {
        "format_version": 1,
        "bbox_xywh": list(reference.bbox_xywh),
        "preparation_config": preparation_config_to_dict(config),
    }

    metadata_path = destination / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_reference_package(
    reference_dir: str | Path,
) -> tuple[PreparedPart, PreparationConfig]:
    source = Path(reference_dir)

    crop = read_image(source / "crop.png")
    crop_mask = read_mask(source / "mask.png")
    prepared = read_image(source / "prepared.png")

    metadata_path = source / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Не найден metadata.json: {metadata_path}"
        )

    try:
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Не удалось прочитать metadata.json: {metadata_path}"
        ) from exc

    if metadata.get("format_version") != 1:
        raise RuntimeError(
            "Неподдерживаемая версия пакета эталона. "
            "Подготовьте эталон повторно."
        )

    bbox = metadata.get("bbox_xywh")

    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or not all(isinstance(value, int) for value in bbox)
    ):
        raise RuntimeError(
            "metadata.bbox_xywh должен быть списком из 4 целых чисел."
        )

    if crop.shape[:2] != crop_mask.shape:
        raise RuntimeError(
            "Размер crop.png не совпадает с размером mask.png."
        )

    if crop.shape != prepared.shape:
        raise RuntimeError(
            "Размер crop.png не совпадает с размером prepared.png."
        )

    config_payload = metadata.get("preparation_config")

    if not isinstance(config_payload, dict):
        raise RuntimeError(
            "В metadata.json отсутствует preparation_config."
        )

    preparation_config = preparation_config_from_dict(
        config_payload,
    )

    source_mask = np.zeros(
        crop.shape[:2],
        dtype=np.uint8,
    )

    return (
        PreparedPart(
            source_mask=source_mask,
            crop=crop,
            crop_mask=crop_mask,
            prepared=prepared,
            bbox_xywh=(
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
            ),
        ),
        preparation_config,
    )


def add_preparation_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Добавляет CLI-параметры этапа preparation."""
    parser.add_argument(
        "--box-ratio",
        type=float,
        default=0.48,
        help="Размер начального GrabCut-box относительно кадра",
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=5_000,
        help="Минимальная площадь контура детали",
    )

    parser.add_argument(
        "--padding",
        type=int,
        default=35,
        help="Padding вокруг bbox детали",
    )

    parser.add_argument(
        "--background",
        type=int,
        nargs=3,
        default=(30, 30, 30),
        metavar=("B", "G", "R"),
        help="Цвет постоянного фона: --background 30 30 30",
    )

    parser.add_argument(
        "--feather-px",
        type=int,
        default=2,
        help="Мягкость края маски на подготовленном фоне",
    )


def build_preparation_config(
    args: argparse.Namespace,
) -> PreparationConfig:
    return PreparationConfig(
        box_ratio=args.box_ratio,
        min_area=args.min_area,
        padding=args.padding,
        background_bgr=tuple(args.background),
        feather_px=args.feather_px,
    )


def run_prepare_reference(args: argparse.Namespace) -> None:
    image = read_image(args.input)
    config = build_preparation_config(args)

    try:
        reference = prepare_part(image, config)
    except PreparationError as exc:
        raise SystemExit(
            f"Не удалось подготовить эталон: {exc}"
        ) from exc

    save_reference_package(
        reference=reference,
        config=config,
        reference_dir=args.reference_dir,
    )

    if args.debug_dir is not None:
        save_preparation_debug(
            source_image=image,
            result=reference,
            output_dir=args.debug_dir,
            prefix="reference_",
        )

    print("Reference preparation completed")
    print(f"reference_dir: {args.reference_dir}")
    print(f"bbox_xywh: {reference.bbox_xywh}")
    print(f"crop_shape: {reference.crop.shape}")

    if args.debug_dir is not None:
        print(f"debug: {args.debug_dir}")


def run_align(args: argparse.Namespace) -> None:
    reference, preparation_config = load_reference_package(
        args.reference_dir,
    )

    current_image = read_image(args.input)

    try:
        current = prepare_part(
            current_image,
            preparation_config,
        )
    except PreparationError as exc:
        raise SystemExit(
            f"Не удалось подготовить текущую деталь: {exc}"
        ) from exc

    alignment_config = AlignmentConfig(
        preparation=preparation_config,
        max_features=args.max_features,
        ratio=args.ratio,
        ransac_threshold=args.ransac_threshold,
    )

    try:
        result = align_prepared_parts(
            reference=reference,
            current=current,
            config=alignment_config,
            include_debug=args.debug_dir is not None,
        )
    except AlignmentError as exc:
        raise SystemExit(
            f"Не удалось выровнять деталь: {exc}"
        ) from exc

    save_image(args.output, result.aligned_prepared)
    save_image(
        Path(args.output).with_name(
            f"{Path(args.output).stem}_mask.png"
        ),
        result.aligned_mask,
    )
    save_image(
        Path(args.output).with_name(
            f"{Path(args.output).stem}_valid_mask.png"
        ),
        result.valid_mask,
    )

    if args.debug_dir is not None:
        save_preparation_debug(
            source_image=current_image,
            result=current,
            output_dir=args.debug_dir,
            prefix="current_",
        )

        save_alignment_debug(
            result=result,
            output_dir=args.debug_dir,
            prefix="alignment_",
        )

    print("Alignment completed")
    print(f"reference_bbox_xywh: {reference.bbox_xywh}")
    print(f"current_bbox_xywh: {current.bbox_xywh}")

    for name, value in result.metrics.items():
        print(f"{name}: {value:.4f}")

    print("matrix_current_crop_to_reference_crop:")

    for row in result.matrix_current_crop_to_reference_crop:
        print(
            "  "
            + " ".join(f"{value: .6f}" for value in row)
        )

    print(f"aligned: {args.output}")

    if args.debug_dir is not None:
        print(f"debug: {args.debug_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Подготовка эталона и alignment текущей детали "
            "к сохранённому эталону."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    prepare_parser = subparsers.add_parser(
        "prepare-reference",
        help="Сегментировать и сохранить эталон",
    )

    prepare_parser.add_argument(
        "--input",
        required=True,
        help="Путь к исходному эталонному изображению",
    )

    prepare_parser.add_argument(
        "--reference-dir",
        required=True,
        help="Папка для пакета подготовленного эталона",
    )

    prepare_parser.add_argument(
        "--debug-dir",
        default=None,
        help="Необязательная папка с debug-артефактами",
    )

    add_preparation_arguments(prepare_parser)
    prepare_parser.set_defaults(handler=run_prepare_reference)

    align_parser = subparsers.add_parser(
        "align",
        help="Выровнять текущую деталь к сохранённому эталону",
    )

    align_parser.add_argument(
        "--reference-dir",
        required=True,
        help="Папка подготовленного эталона",
    )

    align_parser.add_argument(
        "--input",
        required=True,
        help="Путь к текущей фотографии детали",
    )

    align_parser.add_argument(
        "--output",
        required=True,
        help="Куда сохранить выровненную деталь",
    )

    align_parser.add_argument(
        "--debug-dir",
        default=None,
        help="Необязательная папка с debug-артефактами",
    )

    align_parser.add_argument(
        "--max-features",
        type=int,
        default=3_000,
        help="Максимум ORB keypoints",
    )

    align_parser.add_argument(
        "--ratio",
        type=float,
        default=0.72,
        help="Lowe ratio test threshold",
    )

    align_parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=3.0,
        help="Допустимая RANSAC reprojection error в пикселях",
    )

    align_parser.set_defaults(handler=run_align)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
