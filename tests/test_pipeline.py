import io
import json
import os

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api import create_app
from app.services.catalog import Catalog, decode_image
from app.services.features import FeatureConfig, local_distance, tile_grid
from app.services.foreground import ForegroundConfig, prepare_foreground
from app.services.inspection import Inspector, safe_quantile
from app.services.part_alignment import AlignmentError
from app.services.part_preparation import PreparationError
from app.services.postprocessing import components, render


def test_tile_grid_covers_4k():
    coverage = np.zeros((2160, 3840), bool)
    for x, y, w, h in tile_grid(coverage.shape, 448, 112):
        coverage[y:y+h, x:x+w] = True
    assert coverage.all()


@pytest.mark.parametrize("args", [{"tile_size": 0}, {"tile_size": 223}, {"overlap": 448}, {"local_radius": -1}, {"backbone": "untrained"}])
def test_config_rejects(args):
    with pytest.raises(ValueError):
        FeatureConfig(**args)


def test_local_matching_respects_location_and_mask():
    ref = np.zeros((2, 5, 5), np.float32)
    ref[0] = 1
    query = ref.copy()
    query[:, 2, 2] = [0, 1]
    support = np.ones((5, 5), bool)
    distance = local_distance(query, ref, support, 1)
    assert distance[2, 2] == pytest.approx(1)
    assert distance[0, 0] == pytest.approx(0)
    assert np.isinf(local_distance(query, ref, np.zeros_like(support), 1)).all()


def test_missing_support_fusion_is_finite():
    scores, count = safe_quantile([np.array([[np.inf, .2, .7]], np.float32),
                                  np.array([[np.inf, np.inf, .3]], np.float32)], .25)
    assert np.isfinite(scores).all()
    assert count.tolist() == [[0, 1, 2]]
    assert scores.tolist()[0] == pytest.approx([0, .2, .3])


def test_holes_and_table_excluded(scene):
    image, _ = scene
    part = prepare_foreground(image, ForegroundConfig(min_area=1000, padding=10))
    assert part.source_mask[160, 230] == 0
    assert part.source_mask[300, 300] == 255
    assert part.source_mask[20, 20] == 0


def test_explicit_mask_validation(scene):
    image, mask = scene
    with pytest.raises(ValueError):
        prepare_foreground(image, mask=mask.astype(float))
    with pytest.raises(PreparationError):
        prepare_foreground(image, mask=np.zeros_like(mask))


def test_bad_images_and_path(tmp_path):
    with pytest.raises(ValueError):
        decode_image(b"not an image")
    with pytest.raises(ValueError):
        Catalog(tmp_path).part_path("../../outside")


def test_empty_scene_rejected():
    with pytest.raises(PreparationError):
        prepare_foreground(np.full((200, 200, 3), 30, np.uint8))


def test_bbox_back_projection():
    score = np.zeros((100, 100), np.float32)
    score[20:30, 40:55] = .8
    inverse = np.eye(3)
    inverse[:2, 2] = [5, -3]
    result = components(score, np.ones_like(score, bool), .4, 10, inverse, (100, 200, 100, 100), (500, 500))
    assert result[0]["bbox_xyxy"] == [145, 217, 160, 227]


def test_constant_heatmap_not_stretched():
    image = np.full((20, 20, 3), 100, np.uint8)
    scores = np.zeros((20, 20), np.float32)
    assert np.array_equal(render(image, scores, np.ones_like(scores), [], .3), image)


def test_profile_and_inspect(service, scene, tmp_path):
    image, mask = scene
    first = service.add_reference("part_A", image, mask)
    assert service.add_reference("part_A", image, mask) == first
    meta = service.build("part_A")
    before = service.encoder.calls
    result, heat, domain, _ = service.inspect("part_A", image, mask)
    assert service.encoder.calls-before == len(meta["tiles"])
    assert heat.max() < 1e-5
    assert not result["defects"]
    assert np.count_nonzero(heat[mask == 0]) == 0
    assert np.count_nonzero(domain[mask == 0]) == 0
    assert result["decision"] == "inconclusive"
    json.dumps(result, allow_nan=False)
    changed = image.copy()
    cv2.rectangle(changed, (320, 220), (380, 275), (0, 0, 255), -1)
    result = service.inspect_and_save("part_A", changed, tmp_path/"out", mask)
    assert result["defects"]
    assert (tmp_path/"out"/"heatmap.npy").is_file()
    assert np.load(tmp_path/"out"/"heatmap.npy").shape == mask.shape
    assert result["coordinate_frame"] == "original_exif_oriented_image"
    x0, y0, x1, y1 = result["defects"][0]["bbox_xyxy"]
    assert x0 < 380 and x1 > 320 and y0 < 275 and y1 > 220


def test_table_change_never_becomes_defect(service, scene):
    image, mask = scene
    service.add_reference("part_B", image, mask)
    service.build("part_B")
    changed = image.copy()
    changed[mask == 0] = (200, 20, 0)
    result, heat, _, _ = service.inspect("part_B", changed, mask)
    assert not result["defects"]
    assert heat.max() < 1e-5


def test_multi_reference_and_invalidation(service, scene):
    image, mask = scene
    service.add_reference("part_C", image, mask)
    alternate = image.copy()
    alternate[mask > 0] = np.clip(alternate[mask > 0].astype(int)+5, 0, 255)
    service.add_reference("part_C", alternate, mask)
    meta = service.build("part_C")
    assert len(meta["reference_ids"]) == 2
    assert meta["calibration"]["method"] == "reference_leave_one_out"
    result, _, _, _ = service.inspect("part_C", image, mask)
    assert not result["defects"]
    assert len(result["reference_comparisons"]) == 2
    third = alternate.copy()
    third[300, 300] = (20, 80, 30)
    service.add_reference("part_C", third, mask)
    with pytest.raises(ValueError, match="rebuild"):
        service.inspect("part_C", image, mask)


def test_alignment_retries_with_relaxed_matching(service, scene, monkeypatch):
    # Texture-poor surfaces may fail the strict ORB matching attempt; the
    # relaxed retry must still produce a verified alignment result.
    image, mask = scene
    service.add_reference("retry", image, mask)
    service.build("retry")
    import app.services.inspection as module
    original = module.align_prepared_parts
    attempts = []
    def flaky(reference, current, config, include_debug=True):
        attempts.append(config.minimum_inliers)
        if config.minimum_inliers >= 12:
            raise module.AlignmentError("Недостаточно RANSAC-inliers")
        return original(reference, current, config, include_debug=False)
    monkeypatch.setattr(module, "align_prepared_parts", flaky)
    result, _, domain, _ = service.inspect("retry", image, mask)
    assert attempts == [12, 8]
    assert result["alignment"]["inliers"] >= 8
    assert not result["defects"]


def test_wrong_alignment_fails_closed(service, scene):
    image, mask = scene
    service.add_reference("part_D", image, mask)
    service.build("part_D")
    blank = image.copy()
    blank[mask > 0] = 180
    with pytest.raises(AlignmentError):
        service.inspect("part_D", blank, mask)


def test_two_parts_isolated(service, scene):
    image, mask = scene
    service.add_reference("one", image, mask)
    service.add_reference("two", image, mask)
    assert len(service.catalog.references("one")) == 1
    assert service.catalog.references("one")[0]["id"] != service.catalog.references("two")[0]["id"]


def test_api(service, scene):
    image, _ = scene
    client = TestClient(create_app(service))
    assert client.get("/health").status_code == 200
    assert client.post("/parts/test/references", files={"image": ("bad.png", b"no", "image/png")}).status_code == 422
    _, data = cv2.imencode(".png", image)
    response = client.post("/parts/test/references", files={"image": ("ref.png", data.tobytes(), "image/png")})
    assert response.status_code == 201, response.text
    assert response.json()["profile_ready"]
    response = client.post("/parts/test/inspect", files={"image": ("cur.png", data.tobytes(), "image/png")})
    assert response.status_code == 200, response.text
    for name, url in response.json()["artifacts"].items():
        assert client.get(url).status_code == 200, name
    assert client.get("/inspections/invalid/result.json").status_code == 404


def test_exif_orientation_coordinates():
    source = Image.new("RGB", (100, 60), (128, 128, 128))
    exif = source.getexif()
    exif[274] = 6
    buffer = io.BytesIO()
    source.save(buffer, format="JPEG", exif=exif)
    assert decode_image(buffer.getvalue()).shape == (100, 60, 3)


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_PRETRAINED_TESTS") != "1", reason="Pretrained weights require network/cache")
def test_actual_pretrained_encoder(tmp_path, scene):
    image, mask = scene
    config = FeatureConfig(backbone=os.getenv("TEST_BACKBONE", "resnet18"), tile_size=224, overlap=56)
    service = Inspector(tmp_path, config, ForegroundConfig(min_area=1000, padding=14), device="cpu")
    service.add_reference("pretrained", image, mask)
    service.build("pretrained")
    result, heat, domain, _ = service.inspect("pretrained", image, mask)
    assert np.isfinite(heat).all() and domain.any()
    assert not result["defects"]
    changed = image.copy()
    cv2.rectangle(changed, (320, 220), (380, 275), (0, 0, 255), -1)
    _, changed_heat, _, _ = service.inspect("pretrained", changed, mask)
    assert changed_heat[225:270, 325:375].max() > heat.max()+.01


def test_registration_roundtrip_rotated_photo(service, scene):
    image, mask = scene
    service.add_reference("rotated", image, mask)
    service.build("rotated")
    matrix = cv2.getRotationMatrix2D((320, 240), 17, 1.0)
    matrix[:, 2] += [8, -5]
    changed = image.copy()
    cv2.rectangle(changed, (330, 240), (385, 290), (0, 0, 255), -1)
    rotated = cv2.warpAffine(changed, matrix, (640, 480), borderValue=(35, 35, 35))
    rotated_mask = cv2.warpAffine(mask, matrix, (640, 480), flags=cv2.INTER_NEAREST)
    result, _, domain, _ = service.inspect("rotated", rotated, rotated_mask)
    assert result["defects"]
    expected = cv2.transform(np.float32([[[355, 265]]]), matrix)[0, 0]
    assert any(d["bbox_xyxy"][0] <= expected[0] < d["bbox_xyxy"][2]
               and d["bbox_xyxy"][1] <= expected[1] < d["bbox_xyxy"][3] for d in result["defects"])
    assert not domain[rotated_mask == 0].any()


def test_unexpected_scale_rejected(service, scene):
    image, mask = scene
    service.add_reference("scale", image, mask)
    service.build("scale")
    matrix = cv2.getRotationMatrix2D((320, 240), 0, .6)
    small = cv2.warpAffine(image, matrix, (640, 480), borderValue=(35, 35, 35))
    small_mask = cv2.warpAffine(mask, matrix, (640, 480), flags=cv2.INTER_NEAREST)
    with pytest.raises(AlignmentError):
        service.inspect("scale", small, small_mask)


def test_signature_invalidation(service, scene):
    image, mask = scene
    service.add_reference("signature", image, mask)
    service.build("signature")
    service.features = FeatureConfig(tile_size=448)
    with pytest.raises(ValueError, match="Feature configuration changed"):
        service.inspect("signature", image, mask)


def test_calibration_and_profile_version(service, scene, tmp_path):
    image, mask = scene
    service.add_reference("calibrated", image, mask)
    first = service.build("calibrated")
    normal = image.copy()
    normal[100:380, 150:490] = np.clip(normal[100:380, 150:490].astype(int)+2, 0, 255)
    path = tmp_path/"held_out.png"
    cv2.imwrite(str(path), normal)
    calibrated = service.calibrate("calibrated", [path], [mask])
    assert calibrated["id"] != first["id"]
    assert calibrated["calibration"]["method"] == "held_out_normal_images"
    result, _, _, _ = service.inspect("calibrated", normal, mask)
    assert "uncalibrated" not in " ".join(result["warnings"])
    cv2.imwrite(str(path), image)
    with pytest.raises(ValueError, match="must not duplicate"):
        service.calibrate("calibrated", [path], [mask])
