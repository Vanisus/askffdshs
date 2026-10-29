import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.services.part_alignment import AlignmentError
from test_http import register, upload


def test_upload_inspect_without_build_and_duplicate(service, scene):
    image, mask = scene
    client = TestClient(create_app(service))
    first = register(client, image, mask)
    assert first["reference_created"]
    assert first["reference_count"] == 1
    assert first["profile_ready"] and not first["profile_rebuild_required"]
    assert first["calibration"]["method"] == "uncalibrated"
    assert first["calibration"]["sample_count"] == 0
    assert first["warnings"]
    result = client.post("/parts/bracket/inspect", files={"image": upload(image), "mask": upload(mask)})
    assert result.status_code == 200
    assert result.json()["profile_id"] == first["profile_id"]
    assert result.json()["decision"] == "inconclusive"
    generation = service.catalog.state("bracket")["generation"]
    duplicate = register(client, image, mask)
    assert not duplicate["reference_created"]
    assert duplicate["reference_id"] == first["reference_id"]
    assert duplicate["reference_count"] == 1
    assert duplicate["profile_id"] != first["profile_id"]
    assert service.catalog.state("bracket")["generation"] == generation
    assert duplicate["calibration"]["method"] == "uncalibrated"


def test_every_new_reference_rebuilds_and_calibrates(service, scene, monkeypatch):
    image, mask = scene
    client = TestClient(create_app(service))
    first = register(client, image, mask)
    # Capture every local comparison during LOO: a reference must not match itself.
    import app.services.inspection as module
    original = module.local_distance
    comparisons = []
    def distance(query, ref, support, radius):
        comparisons.append(query is ref)
        return original(query, ref, support, radius)
    monkeypatch.setattr(module, "local_distance", distance)
    alternate = image.copy()
    cv2.circle(alternate, (320, 310), 4, (160, 160, 160), -1)
    second = register(client, alternate, mask)
    assert comparisons and not any(comparisons)
    assert second["reference_count"] == 2
    assert second["profile_id"] != first["profile_id"]
    assert second["calibration"]["method"] == "reference_leave_one_out"
    assert second["calibration"]["sample_count"] > 0
    assert second["calibration"]["requires_held_out_validation"]
    assert not second["warnings"]
    profile = client.get("/parts/bracket/profile").json()
    assert profile["id"] == second["profile_id"]
    assert profile["threshold"] == second["threshold"]
    assert set(profile["reference_ids"]) == {first["reference_id"], second["reference_id"]}


@pytest.mark.parametrize("activate_before_failure", [False, True])
def test_failed_rebuild_restores_previous_profile(service, scene, monkeypatch, activate_before_failure):
    image, mask = scene
    client = TestClient(create_app(service))
    first = register(client, image, mask)
    state = service.catalog.state("bracket")
    refs = service.catalog.references("bracket")
    base = service.catalog.part_path("bracket")
    artifacts = set(base.rglob("*"))
    original = service.build
    def broken(part_id):
        if activate_before_failure:
            original(part_id)
        raise AlignmentError("Reference cannot be aligned")
    monkeypatch.setattr(service, "build", broken)
    changed = image.copy()
    changed[300, 300] = (20, 40, 60)
    response = client.post("/parts/bracket/references", files={"image": upload(changed), "mask": upload(mask)})
    assert response.status_code == 422
    assert service.catalog.state("bracket") == state
    assert service.catalog.references("bracket") == refs
    assert set(base.rglob("*")) == artifacts
    assert client.get("/parts/bracket/profile").json()["id"] == first["profile_id"]
    assert client.post("/parts/bracket/inspect", files={"image": upload(image), "mask": upload(mask)}).status_code == 200
    assert not client.get("/health").json()["busy"]


def test_failed_first_upload_does_not_register_part(service, scene, monkeypatch):
    image, mask = scene
    def broken(part_id):
        raise RuntimeError("model download failed")
    monkeypatch.setattr(service, "build", broken)
    client = TestClient(create_app(service))
    response = client.post("/parts/new/references", files={"image": upload(image), "mask": upload(mask)})
    assert response.status_code == 500
    assert client.get("/parts").json()["total"] == 0
    assert not list((service.catalog.part_path("new") / "references").iterdir())
    assert client.get("/parts/new/profile").status_code == 404
    assert not client.get("/health").json()["busy"]


def test_upload_holds_busy_until_profile_is_ready(service, scene, monkeypatch):
    image, mask = scene
    client = TestClient(create_app(service))
    entered, release = threading.Event(), threading.Event()
    original = service.build
    def slow(part_id):
        entered.set()
        assert release.wait(10)
        return original(part_id)
    monkeypatch.setattr(service, "build", slow)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(client.post, "/parts/bracket/references",
                             files={"image": upload(image), "mask": upload(mask)})
        try:
            assert entered.wait(5)
            assert not future.done()
            assert client.get("/health").json()["busy"]
            assert client.post("/parts/bracket/references",
                               files={"image": upload(image), "mask": upload(mask)}).status_code == 503
        finally:
            release.set()
        assert future.result(timeout=10).json()["profile_ready"]
    assert client.get("/parts/bracket/profile").status_code == 200


def test_actual_incompatible_reference_is_rolled_back(service, scene):
    image, mask = scene
    client = TestClient(create_app(service))
    first = register(client, image, mask)
    incompatible = image.copy()
    incompatible[mask > 0] = 180
    response = client.post("/parts/bracket/references", files={"image": upload(incompatible), "mask": upload(mask)})
    assert response.status_code == 422
    assert len(service.catalog.references("bracket")) == 1
    assert client.get("/parts/bracket/profile").json()["id"] == first["profile_id"]


def test_failed_duplicate_keeps_reference_and_profile(service, scene, monkeypatch):
    image, mask = scene
    client = TestClient(create_app(service))
    first = register(client, image, mask)
    refs = service.catalog.references("bracket")
    def fail(part_id):
        raise RuntimeError("model failure")
    monkeypatch.setattr(service, "build", fail)
    response = client.post("/parts/bracket/references", files={"image": upload(image), "mask": upload(mask)})
    assert response.status_code == 500
    assert service.catalog.references("bracket") == refs
    assert client.get("/parts/bracket/profile").json()["id"] == first["profile_id"]
