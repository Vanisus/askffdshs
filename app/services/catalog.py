"""SQLite metadata and immutable, versioned on-disk artifacts (no pickle)."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3
import threading
import uuid

import cv2
import numpy as np
from PIL import Image, ImageOps
from filelock import FileLock

from .part_preparation import PreparedPart

MAX_BYTES = 50 * 1024 * 1024
MAX_PIXELS = 25_000_000


def decode_image(data: bytes):
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Image is empty or exceeds 50 MiB")
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.width*im.height > MAX_PIXELS:
                raise ValueError("Image exceeds 25 megapixels")
            if im.format not in {"JPEG", "PNG", "TIFF", "WEBP", "BMP"}:
                raise ValueError("Unsupported image format")
            rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
        return rgb[:, :, ::-1].copy()
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid image") from exc


def write_image(path, image):
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    Path(path).write_bytes(data.tobytes())


def save_part(path, part):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    write_image(path / "crop.png", part.crop)
    write_image(path / "mask.png", part.crop_mask)
    (path / "bbox.json").write_text(json.dumps(list(part.bbox_xywh)), encoding="utf-8")


def load_part(path):
    path = Path(path)
    crop = cv2.imread(str(path / "crop.png"))
    mask = cv2.imread(str(path / "mask.png"), cv2.IMREAD_GRAYSCALE)
    if crop is None or mask is None or crop.shape[:2] != mask.shape:
        raise RuntimeError("Corrupt reference artifacts")
    bbox = tuple(json.loads((path / "bbox.json").read_text()))
    prepared = crop.copy()
    prepared[mask == 0] = 30
    return PreparedPart(np.zeros((0, 0), np.uint8), crop, mask, prepared, bbox)


class Catalog:
    def __init__(self, root):
        self._locks = {}
        self._locks_guard = threading.Lock()
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
            CREATE TABLE IF NOT EXISTS parts(id TEXT PRIMARY KEY, generation INTEGER NOT NULL DEFAULT 0,
                                             active_profile TEXT);
            CREATE TABLE IF NOT EXISTS refs(id TEXT PRIMARY KEY, part_id TEXT NOT NULL,
                                            digest TEXT NOT NULL, path TEXT NOT NULL,
                                            UNIQUE(part_id,digest));
            """)

    def connect(self):
        db = sqlite3.connect(self.root / "catalog.sqlite3", timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def part_path(self, part_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", part_id):
            raise ValueError("part_id must be 1..80 ASCII letters, digits, '-' or '_'")
        return self.root / "parts" / part_id

    def lock(self, part_id):
        path = self.part_path(part_id)
        path.mkdir(parents=True, exist_ok=True)
        # Reuse the same reentrant FileLock for composite add + build operations.
        with self._locks_guard:
            if part_id not in self._locks:
                self._locks[part_id] = FileLock(str(path / ".write.lock"), timeout=300)
            return self._locks[part_id]

    def add_reference(self, part_id, part):
        base = self.part_path(part_id)
        digest = hashlib.sha256(part.crop.tobytes() + part.crop_mask.tobytes()
                                + str(part.crop.shape).encode()).hexdigest()
        with self.lock(part_id), self.connect() as db:
            db.execute("INSERT OR IGNORE INTO parts(id) VALUES (?)", (part_id,))
            row = db.execute("SELECT id FROM refs WHERE part_id=? AND digest=?", (part_id, digest)).fetchone()
            if row:
                return row["id"]
            reference_id = uuid.uuid4().hex
            dest = base / "references" / reference_id
            save_part(dest, part)
            db.execute("INSERT INTO refs VALUES (?,?,?,?)", (reference_id, part_id, digest, str(dest)))
            db.execute("UPDATE parts SET generation=generation+1 WHERE id=?", (part_id,))
        return reference_id

    def references(self, part_id):
        self.part_path(part_id)
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM refs WHERE part_id=? ORDER BY rowid", (part_id,))]

    def state(self, part_id):
        self.part_path(part_id)
        with self.connect() as db:
            row = db.execute("SELECT * FROM parts WHERE id=?", (part_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown part_id")
        return dict(row)

    def activate(self, part_id, generation, profile):
        with self.connect() as db:
            cursor = db.execute("UPDATE parts SET active_profile=? WHERE id=? AND generation=?",
                                (str(profile), part_id, generation))
            if cursor.rowcount != 1:
                raise RuntimeError("References changed during profile construction")
