import io
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.http_config import HTTPSettings


def upload(image, name="photo.png"):
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return name, encoded.tobytes(), "image/png"


def register(client, image, mask=None, part_id="bracket"):
    files = {"image": upload(image)}
    if mask is not None:
        files["mask"] = upload(mask, "mask.png")
    response = client.post(f"/parts/{part_id}/references", files=files)
    assert response.status_code == 201, response.text
    return response.json()


def test_http_full_workflow(service, scene):
    image, mask = scene
    client = TestClient(create_app(service))
    assert client.get("/parts").json()["total"] == 0
    ref = register(client, image, mask)
    assert register(client, image, mask)["reference_id"] == ref["reference_id"]
    assert client.get("/parts/bracket/profile").status_code == 409
    assert client.post("/parts/bracket/build").status_code == 200
    assert client.get("/parts/bracket/profile").json()["reference_ids"] == [ref["reference_id"]]
    assert client.get("/parts/bracket/references").json()["items"] == [{"reference_id": ref["reference_id"]}]
    assert client.get("/parts?limit=1&offset=1").json()["items"] == []
    assert client.get("/parts").json()["items"][0]["reference_count"] == 1
    response = client.post("/parts/bracket/inspect", files={"image": upload(image), "mask": upload(mask)})
    assert response.status_code == 200, response.text
    payload = response.json()
    saved = client.get(f"/inspections/{payload['inspection_id']}")
    assert saved.json() == payload
    heat = np.load(io.BytesIO(client.get(payload["artifacts"]["heatmap.npy"]).content), allow_pickle=False)
    assert heat.shape == image.shape[:2]
    assert not heat[mask == 0].any()
    assert client.get(payload["artifacts"]["overlay.png"]).headers["content-type"] == "image/png"
    assert client.get(f"/inspections/{payload['inspection_id']}/secret.txt").status_code == 404
    assert client.get("/inspections/" + "0"*32).status_code == 404
    # Calibration images must be held-out normals, not the reference itself.
    normal = image.copy()
    normal[100:380, 150:490] = np.clip(normal[100:380, 150:490].astype(int)+2, 0, 255)
    response = client.post("/parts/bracket/calibrate", files=[("images", upload(normal, "../../outside.png"))])
    assert response.status_code == 200, response.text
    assert response.json()["calibration"]["method"] == "held_out_normal_images"
    assert client.post("/parts/bracket/calibrate", files=[("images", upload(image))]).status_code == 422


def test_http_status_codes_and_stale_profiles(service, scene):
    image, mask = scene
    client = TestClient(create_app(service))
    assert client.post("/parts/missing/build").status_code == 404
    assert client.get("/parts/missing/references").status_code == 404
    assert client.post("/parts/missing/inspect", files={"image": upload(image)}).status_code == 404
    assert client.get("/parts?limit=0").status_code == 422
    assert client.post("/parts/invalid.id/references", files={"image": upload(image)}).status_code == 422
    register(client, image, mask)
    assert client.post("/parts/bracket/inspect", files={"image": upload(image)}).status_code == 409
    client.post("/parts/bracket/build")
    altered = image.copy()
    altered[300, 300] = (2, 5, 8)
    register(client, altered, mask)
    assert client.get("/parts/bracket/profile").status_code == 409
    assert client.post("/parts/bracket/inspect", files={"image": upload(image)}).status_code == 409


def test_http_auth_and_openapi(service, scene):
    image, mask = scene
    client = TestClient(create_app(service, HTTPSettings(api_key="test-only-secret")))
    assert client.get("/health").status_code == 200
    assert client.get("/health").json()["authentication_enabled"]
    assert client.get("/parts").status_code == 401
    assert client.get("/parts", headers={"X-API-Key": "wrong"}).status_code == 401
    client.headers["X-API-Key"] = "test-only-secret"
    register(client, image, mask)
    assert client.post("/parts/bracket/build").status_code == 200
    result = client.post("/parts/bracket/inspect", files={"image": upload(image), "mask": upload(mask)}).json()
    client.headers.pop("X-API-Key")
    for url in result["artifacts"].values():
        assert client.get(url).status_code == 401
    assert client.get(f"/inspections/{result['inspection_id']}").status_code == 401
    schema = client.get("/openapi.json").json()
    assert "APIKeyHeader" in schema["components"]["securitySchemes"]
    assert "security" in schema["paths"]["/parts/{part_id}/inspect"]["post"]
    assert client.get("/docs").status_code == 200


def test_http_upload_validation(service, scene):
    image, _ = scene
    client = TestClient(create_app(service))
    assert client.post("/parts/test/references", files={"image": ("a.png", b"", "image/png")}).status_code == 422
    assert client.post("/parts/test/references", files={"image": ("a.png", b"garbage", "image/png")}).status_code == 422
    assert client.post("/parts/test/references", files={"image": upload(image), "mask": upload(np.zeros((32,32), np.uint8))}).status_code == 422
    ok, jpeg = cv2.imencode(".jpg", np.zeros(image.shape[:2], np.uint8))
    assert ok
    assert client.post("/parts/test/references", files={"image": upload(image), "mask": ("mask.jpg", jpeg.tobytes(), "image/jpeg")}).status_code == 422
    settings = HTTPSettings(max_upload_bytes=100, max_request_bytes=2000)
    tiny = TestClient(create_app(service, settings))
    assert tiny.post("/parts/test/references", files={"image": ("a.png", b"a"*101, "image/png")}).status_code == 413
    assert tiny.post("/parts/test/references", content=b"x"*2001).status_code == 413
    assert tiny.post("/parts/test/references", content=b"", headers={"Content-Length": "invalid"}).status_code == 400


def test_http_streamed_body_limit(service):
    settings = HTTPSettings(max_upload_bytes=100, max_request_bytes=1000)
    client = TestClient(create_app(service, settings))
    def chunks():
        yield b'--test\r\nContent-Disposition: form-data; name="image"; filename="x.png"\r\nContent-Type: image/png\r\n\r\n'
        yield b"x"*1100
        yield b"\r\n--test--\r\n"
    response = client.post("/parts/test/references", content=chunks(), headers={"Content-Type": "multipart/form-data; boundary=test"})
    assert response.status_code == 413, response.text


def test_http_calibration_limit(service, scene):
    image, mask = scene
    client = TestClient(create_app(service, HTTPSettings(max_calibration_images=1)))
    register(client, image, mask)
    client.post("/parts/bracket/build")
    response = client.post("/parts/bracket/calibrate", files=[("images", upload(image)), ("images", upload(image))])
    assert response.status_code == 422


def test_http_busy_and_health_responsive(service, scene, monkeypatch):
    image, mask = scene
    client = TestClient(create_app(service))
    register(client, image, mask)
    original = service.build
    entered, release = threading.Event(), threading.Event()
    def slow_build(part_id):
        entered.set()
        assert release.wait(10)
        return original(part_id)
    monkeypatch.setattr(service, "build", slow_build)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(client.post, "/parts/bracket/build")
        try:
            assert entered.wait(5)
            assert client.get("/health").json()["busy"]
            busy = client.post("/parts/bracket/build")
            assert busy.status_code == 503
            assert busy.headers["Retry-After"] == "1"
            assert client.get("/parts").status_code == 200
        finally:
            release.set()
        assert future.result(timeout=10).status_code == 200
    assert not client.get("/health").json()["busy"]


def test_http_failure_unlocks_and_cleans_artifacts(service, scene, monkeypatch):
    image, mask = scene
    client = TestClient(create_app(service))
    register(client, image, mask)
    client.post("/parts/bracket/build")
    def fail(part_id, decoded, output, decoded_mask):
        output.mkdir(parents=True)
        (output/"overlay.png").write_bytes(b"partial")
        raise RuntimeError("sensitive internal path")
    monkeypatch.setattr(service, "inspect_and_save", fail)
    result = client.post("/parts/bracket/inspect", files={"image": upload(image)})
    assert result.status_code == 500
    assert "sensitive" not in result.text
    assert not list((service.catalog.root/"inspections").iterdir())
    assert not client.get("/health").json()["busy"]


def test_http_settings_env(tmp_path, monkeypatch):
    monkeypatch.setenv("INSPECTION_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INSPECTION_DEVICE", "cpu")
    monkeypatch.setenv("INSPECTION_TILE_SIZE", "224")
    monkeypatch.setenv("INSPECTION_TILE_OVERLAP", "56")
    monkeypatch.setenv("INSPECTION_API_KEY", "local-test-key")
    settings = HTTPSettings.from_env()
    assert settings.features.tile_size == 224
    assert settings.data_dir == tmp_path
    assert "local-test-key" not in repr(settings)
    svc = settings.make_inspector()
    assert svc._encoder is None
    assert svc.device == "cpu"
    with pytest.raises(ValueError):
        replace(settings, device="invalid")
    with pytest.raises(ValueError):
        replace(settings, max_upload_bytes=0)


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_PRETRAINED_TESTS") != "1", reason="Requires pretrained weights")
def test_http_real_pretrained(tmp_path, scene):
    from app.services.features import FeatureConfig
    from app.services.foreground import ForegroundConfig
    from app.services.inspection import Inspector
    svc = Inspector(tmp_path, FeatureConfig(backbone=os.getenv("TEST_BACKBONE", "resnet18"),
                    tile_size=224, overlap=56), ForegroundConfig(min_area=1000, padding=14), device="cpu")
    image, mask = scene
    client = TestClient(create_app(svc))
    register(client, image, mask)
    assert client.post("/parts/bracket/build").status_code == 200
    assert client.get("/health").json()["model_loaded"]
    response = client.post("/parts/bracket/inspect", files={"image": upload(image), "mask": upload(mask)})
    assert response.status_code == 200, response.text
    assert not response.json()["defects"]
