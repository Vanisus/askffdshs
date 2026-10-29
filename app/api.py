"""Local HTTP integration. Add authentication before exposing this service."""
from __future__ import annotations
import os
import re
import threading
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from .services.catalog import MAX_BYTES, decode_image
from .services.features import FeatureConfig
from .services.inspection import Inspector
from .services.part_alignment import AlignmentError
from .services.part_preparation import PreparationError


def create_app(service=None):
    app = FastAPI(title="Few-shot Part Inspection", version="1.0.0")
    app.state.inspector = service
    init_lock = threading.Lock()
    work_lock = threading.Lock()

    def inspector():
        with init_lock:
            if app.state.inspector is None:
                app.state.inspector = Inspector(os.getenv("INSPECTION_DATA_DIR", "./data"),
                    FeatureConfig(backbone=os.getenv("INSPECTION_BACKBONE", "dinov2_vits14")),
                    device=os.getenv("INSPECTION_DEVICE", "cuda"), hub_dir=os.getenv("DINO_HUB_DIR"))
        return app.state.inspector

    async def read_upload(file):
        data = await file.read(MAX_BYTES+1)
        await file.close()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "Image exceeds 50 MiB")
        try:
            return await run_in_threadpool(decode_image, data)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    async def execute(func, *args):
        try:
            def serial_work():
                with work_lock:
                    return func(*args)
            return await run_in_threadpool(serial_work)
        except (ValueError, AlignmentError, PreparationError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": app.state.inspector is not None and app.state.inspector._encoder is not None}

    @app.post("/parts/{part_id}/references", status_code=201)
    async def reference(part_id: str, image: UploadFile = File(...)):
        decoded = await read_upload(image)
        reference_id = await execute(inspector().add_reference, part_id, decoded)
        return {"part_id": part_id, "reference_id": reference_id, "profile_rebuild_required": True}

    @app.post("/parts/{part_id}/build")
    async def build(part_id: str):
        return await execute(inspector().build, part_id)

    @app.post("/parts/{part_id}/inspect")
    async def inspect(part_id: str, image: UploadFile = File(...)):
        decoded = await read_upload(image)
        current_service = inspector()
        inspection_id = uuid.uuid4().hex
        output = current_service.catalog.root / "inspections" / inspection_id
        result = await execute(current_service.inspect_and_save, part_id, decoded, output)
        return {**result, "inspection_id": inspection_id,
                "artifacts": {name: f"/inspections/{inspection_id}/{name}"
                              for name in ("result.json", "overlay.png", "heatmap.npy", "inspection_mask.png")}}

    @app.get("/inspections/{inspection_id}/{artifact}")
    def artifact(inspection_id: str, artifact: str):
        allowed = {"result.json", "overlay.png", "heatmap.npy", "inspection_mask.png"}
        if not re.fullmatch(r"[a-f0-9]{32}", inspection_id) or artifact not in allowed:
            raise HTTPException(404)
        path = inspector().catalog.root / "inspections" / inspection_id / artifact
        if not path.is_file() or not (path.parent / "result.json").is_file():
            raise HTTPException(404)
        return FileResponse(path)

    return app


app = create_app()
