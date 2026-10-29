#!/usr/bin/env python3
"""CLI: prepare-reference, align, add-reference, build-profile, calibrate, inspect."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.part_alignment import AlignmentConfig, AlignmentError, align_prepared_parts, save_alignment_debug
from app.services.part_preparation import PreparedPart, PreparationConfig, PreparationError, prepare_part, save_preparation_debug


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Невозможно прочитать изображение: {path}")
    return image


def read_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Невозможно прочитать маску: {path}")
    return mask


def save_image(path, image):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Невозможно записать изображение: {output_path}")


def preparation_config_to_dict(config):
    return {"box_ratio": config.box_ratio, "min_area": config.min_area,
            "padding": config.padding, "background_bgr": list(config.background_bgr),
            "feather_px": config.feather_px}


def preparation_config_from_dict(payload):
    try:
        background = payload["background_bgr"]
        if not isinstance(background, list) or len(background) != 3:
            raise ValueError("metadata.background_bgr должен быть списком из 3 чисел.")
        return PreparationConfig(box_ratio=float(payload["box_ratio"]), min_area=int(payload["min_area"]),
                                 padding=int(payload["padding"]), background_bgr=tuple(int(v) for v in background),
                                 feather_px=int(payload["feather_px"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("metadata.json эталона имеет неверный формат.") from exc


def save_reference_package(reference, config, reference_dir):
    destination = Path(reference_dir)
    destination.mkdir(parents=True, exist_ok=True)
    save_image(destination / "crop.png", reference.crop)
    save_image(destination / "mask.png", reference.crop_mask)
    save_image(destination / "prepared.png", reference.prepared)
    metadata = {"format_version": 1, "bbox_xywh": list(reference.bbox_xywh),
                "preparation_config": preparation_config_to_dict(config)}
    (destination / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_reference_package(reference_dir):
    source = Path(reference_dir)
    crop = read_image(source / "crop.png")
    crop_mask = read_mask(source / "mask.png")
    prepared = read_image(source / "prepared.png")
    metadata_path = source / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Не найден metadata.json: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Не удалось прочитать metadata.json: {metadata_path}") from exc
    if metadata.get("format_version") != 1:
        raise RuntimeError("Неподдерживаемая версия пакета эталона. Подготовьте эталон повторно.")
    bbox = metadata.get("bbox_xywh")
    if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(v, int) for v in bbox):
        raise RuntimeError("metadata.bbox_xywh должен быть списком из 4 целых чисел.")
    if crop.shape[:2] != crop_mask.shape or crop.shape != prepared.shape:
        raise RuntimeError("Размеры crop.png, mask.png и prepared.png не совпадают.")
    config_payload = metadata.get("preparation_config")
    if not isinstance(config_payload, dict):
        raise RuntimeError("В metadata.json отсутствует preparation_config.")
    config = preparation_config_from_dict(config_payload)
    return PreparedPart(np.zeros(crop.shape[:2], np.uint8), crop, crop_mask, prepared, tuple(bbox)), config


def add_preparation_arguments(parser):
    parser.add_argument("--box-ratio", type=float, default=.48)
    parser.add_argument("--min-area", type=int, default=5000)
    parser.add_argument("--padding", type=int, default=35)
    parser.add_argument("--background", type=int, nargs=3, default=(30, 30, 30), metavar=("B", "G", "R"))
    parser.add_argument("--feather-px", type=int, default=2)


def build_preparation_config(args):
    return PreparationConfig(box_ratio=args.box_ratio, min_area=args.min_area, padding=args.padding,
                             background_bgr=tuple(args.background), feather_px=args.feather_px)


def run_prepare_reference(args):
    image = read_image(args.input)
    config = build_preparation_config(args)
    try:
        reference = prepare_part(image, config)
    except PreparationError as exc:
        raise SystemExit(f"Не удалось подготовить эталон: {exc}") from exc
    save_reference_package(reference, config, args.reference_dir)
    if args.debug_dir is not None:
        save_preparation_debug(image, reference, args.debug_dir, prefix="reference_")
    print("Reference preparation completed")
    print(f"reference_dir: {args.reference_dir}")
    print(f"bbox_xywh: {reference.bbox_xywh}")
    print(f"crop_shape: {reference.crop.shape}")


def run_align(args):
    reference, preparation_config = load_reference_package(args.reference_dir)
    current_image = read_image(args.input)
    try:
        current = prepare_part(current_image, preparation_config)
    except PreparationError as exc:
        raise SystemExit(f"Не удалось подготовить текущую деталь: {exc}") from exc
    config = AlignmentConfig(preparation=preparation_config, max_features=args.max_features,
                             ratio=args.ratio, ransac_threshold=args.ransac_threshold)
    try:
        result = align_prepared_parts(reference, current, config, include_debug=args.debug_dir is not None)
    except AlignmentError as exc:
        raise SystemExit(f"Не удалось выровнять деталь: {exc}") from exc
    save_image(args.output, result.aligned_prepared)
    for suffix, image in [("mask", result.aligned_mask), ("valid_mask", result.valid_mask)]:
        save_image(Path(args.output).with_name(f"{Path(args.output).stem}_{suffix}.png"), image)
    if args.debug_dir is not None:
        save_preparation_debug(current_image, current, args.debug_dir, prefix="current_")
        save_alignment_debug(result, args.debug_dir, prefix="alignment_")
    print("Alignment completed")
    print(f"reference_bbox_xywh: {reference.bbox_xywh}")
    print(f"current_bbox_xywh: {current.bbox_xywh}")
    for name, value in result.metrics.items():
        print(f"{name}: {value:.4f}")
    print("matrix_current_crop_to_reference_crop:")
    for row in result.matrix_current_crop_to_reference_crop:
        print("  " + " ".join(f"{value: .6f}" for value in row))
    print(f"aligned: {args.output}")


def build_parser():
    parser = argparse.ArgumentParser(description="Подготовка, выравнивание и few-shot инспекция деталей")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-reference", help="Сегментировать и сохранить эталон")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--reference-dir", required=True)
    prepare.add_argument("--debug-dir", default=None)
    add_preparation_arguments(prepare)
    prepare.set_defaults(handler=run_prepare_reference)
    align = subparsers.add_parser("align", help="Выровнять текущую деталь к сохранённому эталону")
    align.add_argument("--reference-dir", required=True)
    align.add_argument("--input", required=True)
    align.add_argument("--output", required=True)
    align.add_argument("--debug-dir", default=None)
    align.add_argument("--max-features", type=int, default=3000)
    align.add_argument("--ratio", type=float, default=.72)
    align.add_argument("--ransac-threshold", type=float, default=3.)
    align.set_defaults(handler=run_align)
    from app.inspection_cli import add_inspection_commands
    add_inspection_commands(subparsers)
    return parser


def main():
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
