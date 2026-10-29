"""Public inspection CLI; legacy prepare-reference/align commands remain available."""
from __future__ import annotations
import json
from pathlib import Path
import cv2

from .services.catalog import decode_image
from .services.features import FeatureConfig
from .services.foreground import ForegroundConfig
from .services.inspection import InspectionConfig, Inspector


def make_inspector(args):
    return Inspector(args.data_dir,
                     FeatureConfig(args.backbone, args.tile_size, args.tile_overlap, args.local_radius),
                     ForegroundConfig(min_area=args.min_part_area),
                     InspectionConfig(threshold=args.threshold, reference_quantile=args.reference_quantile,
                                      min_area_px=args.min_defect_area, edge_margin_px=args.edge_margin),
                     args.device, hub_dir=args.dino_hub_dir)


def run(args):
    service = make_inspector(args)
    mask = None
    if getattr(args, "mask", None):
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError("Cannot read mask")
    if args.command == "add-reference":
        result = {"reference_id": service.add_reference(args.part_id, decode_image(Path(args.input).read_bytes()), mask),
                  "next_step": "build-profile"}
    elif args.command == "calibrate":
        paths = sorted(p for p in Path(args.normal_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"})
        result = service.calibrate(args.part_id, paths)
    elif args.command == "build-profile":
        result = service.build(args.part_id)
    else:
        result = service.inspect_and_save(args.part_id, decode_image(Path(args.input).read_bytes()), args.out, mask)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


def add_inspection_commands(subparsers):
    for name, help_text in [("add-reference", "Add a normal reference to the SQLite part catalog"),
                            ("build-profile", "Cache aligned reference embeddings, without model training"),
                            ("calibrate", "Calibrate a threshold on held-out normal photos"),
                            ("inspect", "Inspect a new photo; output bbox in original image coordinates")]:
        p = subparsers.add_parser(name, help=help_text)
        p.add_argument("--data-dir", default="./data")
        p.add_argument("--part-id", required=True)
        p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
        p.add_argument("--backbone", choices=["dinov2_vits14", "dinov2_vitb14", "resnet18"], default="dinov2_vits14")
        p.add_argument("--tile-size", type=int, default=448)
        p.add_argument("--tile-overlap", type=int, default=112)
        p.add_argument("--local-radius", type=int, default=1)
        p.add_argument("--threshold", type=float, default=.15)
        p.add_argument("--reference-quantile", type=float, default=.25)
        p.add_argument("--min-part-area", type=int, default=5000)
        p.add_argument("--min-defect-area", type=int, default=30)
        p.add_argument("--edge-margin", type=int, default=4)
        p.add_argument("--dino-hub-dir", help="Audited local DINOv2 checkout for offline deployment")
        if name == "calibrate":
            p.add_argument("--normal-dir", required=True)
        if name in {"add-reference", "inspect"}:
            p.add_argument("--input", required=True)
            p.add_argument("--mask", help="Optional original-frame mask; otherwise segment uniform table")
        if name == "inspect":
            p.add_argument("--out", required=True)
        p.set_defaults(handler=run)
