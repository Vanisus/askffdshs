"""Synchronous reference upload + profile rebuild + leave-one-reference-out calibration."""
from __future__ import annotations

import shutil

from .foreground import prepare_foreground


def register_reference(service, part_id, image, mask=None):
    """Upload + rebuild + reference LOO calibration; roll back failed uploads.

    A single reference has no independent calibration evidence. Its profile is
    usable, but remains explicitly uncalibrated. Duplicates also trigger rebuild.
    """
    part = prepare_foreground(image, service.foreground, mask)
    base = service.catalog.part_path(part_id)
    with service.catalog.lock(part_id):
        with service.catalog.connect() as db:
            row = db.execute("SELECT * FROM parts WHERE id=?", (part_id,)).fetchone()
        previous = dict(row) if row is not None else None
        previous_ids = {r["id"] for r in service.catalog.references(part_id)}
        reference_dir = base / "references"
        previous_reference_dirs = set(reference_dir.iterdir()) if reference_dir.exists() else set()
        profiles = base / "profiles"
        previous_profiles = set(profiles.iterdir()) if profiles.exists() else set()
        reference_id = None
        try:
            reference_id = service.catalog.add_reference(part_id, part)
            profile = service.build(part_id)
        except Exception:
            # Heavy HTTP operations are serialized; FileLock also excludes other
            # cooperating writers while restoring the previously valid generation.
            with service.catalog.connect() as db:
                added_ids = [r[0] for r in db.execute("SELECT id FROM refs WHERE part_id=?", (part_id,))
                             if r[0] not in previous_ids]
                db.executemany("DELETE FROM refs WHERE part_id=? AND id=?",
                               [(part_id, ref_id) for ref_id in added_ids])
                if previous is None:
                    db.execute("DELETE FROM parts WHERE id=?", (part_id,))
                else:
                    db.execute("UPDATE parts SET generation=?, active_profile=? WHERE id=?",
                               (previous["generation"], previous["active_profile"], part_id))
            if reference_dir.exists():
                for path in set(reference_dir.iterdir()) - previous_reference_dirs:
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
            if profiles.exists():
                for path in set(profiles.iterdir()) - previous_profiles:
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
            raise
    calibrated = profile["calibration"]["method"] != "uncalibrated"
    return {"part_id": part_id, "reference_id": reference_id,
            "reference_created": reference_id not in previous_ids,
            "reference_count": len(profile["reference_ids"]),
            "profile_rebuild_required": False, "profile_ready": True,
            "profile_id": profile["id"], "threshold": profile["threshold"],
            "calibration": profile["calibration"],
            "warnings": [] if calibrated else [
                "Only one reference or no comparable calibration patches: threshold is uncalibrated; "
                "upload another normal reference to calibrate it"],
            "profile": profile}

