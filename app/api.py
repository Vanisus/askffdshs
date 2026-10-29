"""HTTP adapter for the existing Inspector; run one worker per GPU."""
from __future__ import annotations

import io
import json
import logging
import secrets
import shutil
import threading
import uuid
from typing import Annotated

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Path as APIPath, Query, Security, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from filelock import Timeout as FileLockTimeout
from PIL import Image, ImageOps
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from .http_config import HTTPSettings
from .services.catalog import MAX_PIXELS, decode_image
from .services.part_alignment import AlignmentError
from .services.part_preparation import PreparationError
from .services.reference_registration import register_reference

logger = logging.getLogger(__name__)
PartId = Annotated[str, APIPath(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")]
InspectionId = Annotated[str, APIPath(pattern=r"^[a-f0-9]{32}$")]
ARTIFACTS = {"result.json": "application/json", "overlay.png": "image/png",
             "heatmap.npy": "application/octet-stream", "inspection_mask.png": "image/png"}


class RequestSizeLimit:
    """Limit complete multipart bodies, including chunked requests and masks."""
    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        content_length = dict(scope.get("headers", [])).get(b"content-length")
        if content_length:
            try:
                length = int(content_length)
                if length < 0:
                    raise ValueError
            except ValueError:
                return await JSONResponse({"detail": "Invalid Content-Length"}, 400)(scope, receive, send)
            if length > self.limit:
                return await JSONResponse({"detail": "Request body exceeds configured limit"}, 413)(scope, receive, send)
        count = 0

        async def limited_receive():
            nonlocal count
            message = await receive()
            if message["type"] == "http.request":
                count += len(message.get("body", b""))
                if count > self.limit:
                    raise HTTPException(413, "Request body exceeds configured limit")
            return message
        await self.app(scope, limited_receive, send)


def decode_mask(data, shape):
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.width * im.height > MAX_PIXELS:
                raise ValueError("Mask exceeds 25 megapixels")
            if im.format != "PNG":
                raise ValueError("Mask must be a lossless PNG")
            mask = np.asarray(ImageOps.exif_transpose(im).convert("L"))
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid mask") from exc
    if mask.shape != shape:
        raise ValueError("Mask must match the EXIF-oriented image dimensions")
    return np.uint8(mask > 0) * 255


def create_app(service=None, settings=None):
    settings = settings or HTTPSettings.from_env()
    key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def authorize(key: str | None = Security(key_header)):
        if settings.api_key and (key is None or not secrets.compare_digest(key.encode(), settings.api_key.encode())):
            raise HTTPException(401, "Invalid or missing X-API-Key", headers={"WWW-Authenticate": "APIKey"})

    app = FastAPI(title="Few-shot Part Inspection", version="1.3.0",
                  description="Reference upload builds and calibrates the profile automatically; "
                              "then inspect localizes defects. Scores are distances, not probabilities. "
                              "One inference worker per GPU.")
    app.add_middleware(RequestSizeLimit, limit=settings.max_request_bytes)
    app.state.inspector = service
    app.state.settings = settings
    init_lock = threading.Lock()
    work_lock = threading.Lock()

    def inspector():
        with init_lock:
            if app.state.inspector is None:
                app.state.inspector = settings.make_inspector()
        return app.state.inspector

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request, exc):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Do not echo raw request bodies or uploads in validation messages.
        return JSONResponse({"detail": [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
                                        for e in exc.errors()]}, status_code=422)

    def existing(svc, part_id):
        try:
            return svc.catalog.state(part_id)
        except ValueError as exc:
            raise HTTPException(404, "Unknown part_id") from exc

    async def execute(func, *args):
        try:
            return await run_in_threadpool(func, *args)
        except HTTPException:
            raise
        except FileLockTimeout as exc:
            raise HTTPException(503, "Part catalog is busy", headers={"Retry-After": "1"}) from exc
        except (ValueError, AlignmentError, PreparationError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            logger.exception("Inspection operation failed")
            raise HTTPException(500, "Operation failed; consult server logs") from exc

    async def exclusive():
        # Fail fast instead of growing an unbounded GPU job queue.
        if not work_lock.acquire(blocking=False):
            raise HTTPException(503, "Inspection worker is busy; retry later", headers={"Retry-After": "1"})
        try:
            yield
        finally:
            work_lock.release()

    async def read_bytes(upload):
        try:
            content = await upload.read(settings.max_upload_bytes + 1)
        finally:
            await upload.close()
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(413, "Uploaded file exceeds configured limit")
        if not content:
            raise HTTPException(422, "Uploaded file is empty")
        return content

    async def read_image_and_mask(image, mask):
        decoded = await execute(decode_image, await read_bytes(image))
        decoded_mask = None
        if mask is not None:
            decoded_mask = await execute(decode_mask, await read_bytes(mask), decoded.shape[:2])
        return decoded, decoded_mask

    def profile_summary(svc, part_id):
        state = existing(svc, part_id)
        if not state["active_profile"]:
            raise HTTPException(409, "No active profile; upload a reference to build it")
        try:
            _, meta = svc._profile(part_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"part_id": part_id, **meta}

    protected = [Depends(authorize)]
    heavy = [Depends(authorize), Depends(exclusive)]

    @app.get("/health", tags=["Health"])
    def health():
        svc = app.state.inspector
        return {"status": "ok", "model_loaded": svc is not None and svc._encoder is not None,
                "busy": work_lock.locked(), "authentication_enabled": settings.api_key is not None}

    @app.get("/parts", dependencies=protected, tags=["Parts"])
    async def parts(limit: Annotated[int, Query(ge=1, le=100)] = 20,
                    offset: Annotated[int, Query(ge=0)] = 0):
        def read():
            catalog = inspector().catalog
            with catalog.connect() as db:
                total = db.execute("SELECT COUNT(*) FROM parts").fetchone()[0]
                rows = db.execute("SELECT p.id, p.generation, p.active_profile IS NOT NULL AS has_profile, "
                                  "(SELECT COUNT(*) FROM refs r WHERE r.part_id=p.id) AS reference_count "
                                  "FROM parts p ORDER BY p.id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
            return {"items": [{**dict(r), "has_profile": bool(r["has_profile"])} for r in rows],
                    "total": total, "limit": limit, "offset": offset}
        return await execute(read)

    @app.get("/parts/{part_id}/references", dependencies=protected, tags=["References"])
    async def references(part_id: PartId, limit: Annotated[int, Query(ge=1, le=100)] = 20,
                         offset: Annotated[int, Query(ge=0)] = 0):
        def read():
            svc = inspector()
            existing(svc, part_id)
            refs = svc.catalog.references(part_id)
            return {"part_id": part_id, "items": [{"reference_id": r["id"]} for r in refs[offset:offset+limit]],
                    "total": len(refs), "limit": limit, "offset": offset}
        return await execute(read)

    @app.post("/parts/{part_id}/references", status_code=201, dependencies=heavy, tags=["References"])
    async def reference(part_id: PartId, image: Annotated[UploadFile, File()],
                        mask: Annotated[UploadFile | None, File()] = None):
        decoded, decoded_mask = await read_image_and_mask(image, mask)
        # Do not return success until the new generation is built and activated.
        return await execute(lambda: register_reference(inspector(), part_id, decoded, decoded_mask))

    @app.get("/parts/{part_id}/profile", dependencies=protected, tags=["Profiles"])
    async def profile(part_id: PartId):
        return await execute(lambda: profile_summary(inspector(), part_id))

    @app.post("/parts/{part_id}/inspect", dependencies=heavy, tags=["Inspections"])
    async def inspect(part_id: PartId, image: Annotated[UploadFile, File()],
                      mask: Annotated[UploadFile | None, File()] = None):
        await execute(lambda: profile_summary(inspector(), part_id))
        decoded, decoded_mask = await read_image_and_mask(image, mask)
        inspection_id = uuid.uuid4().hex
        def run():
            svc = inspector()
            output = svc.catalog.root / "inspections" / inspection_id
            try:
                result = svc.inspect_and_save(part_id, decoded, output, decoded_mask)
            except Exception:
                shutil.rmtree(output, ignore_errors=True)
                raise
            return {**result, "inspection_id": inspection_id,
                    "artifacts": {name: f"/inspections/{inspection_id}/{name}" for name in ARTIFACTS}}
        return await execute(run)

    def result_path(inspection_id):
        base = inspector().catalog.root / "inspections" / inspection_id
        if not (base / "result.json").is_file():
            raise HTTPException(404, "Inspection not found")
        return base

    @app.get("/inspections/{inspection_id}", dependencies=protected, tags=["Inspections"])
    async def inspection_result(inspection_id: InspectionId):
        def read():
            payload = json.loads((result_path(inspection_id) / "result.json").read_text(encoding="utf-8"))
            return {**payload, "inspection_id": inspection_id,
                    "artifacts": {name: f"/inspections/{inspection_id}/{name}" for name in ARTIFACTS}}
        return await execute(read)

    @app.get("/inspections/{inspection_id}/{artifact}", dependencies=protected, tags=["Inspections"])
    async def artifact(inspection_id: str, artifact: str):
        # Keep invalid artifact URLs as 404 for backwards compatibility.
        if len(inspection_id) != 32 or any(c not in "0123456789abcdef" for c in inspection_id) or artifact not in ARTIFACTS:
            raise HTTPException(404, "Artifact not found")
        def read():
            path = result_path(inspection_id) / artifact
            if not path.is_file():
                raise HTTPException(404, "Artifact not found")
            return FileResponse(path, media_type=ARTIFACTS[artifact])
        return await execute(read)

    return app


app = create_app()
