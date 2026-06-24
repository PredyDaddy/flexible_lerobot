from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from cqy.allign import head_alignment_service


def _write_image(path: Path, image: np.ndarray) -> Path:
    ok = cv2.imwrite(str(path), image)
    assert ok is True
    return path


def _solid_bgr(color: tuple[int, int, int], size: tuple[int, int] = (6, 8)) -> np.ndarray:
    height, width = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = color
    return image


def test_prepare_display_frame_supports_rgb_and_bgr_modes() -> None:
    frame = np.array([[[10, 20, 30]]], dtype=np.uint8)

    rgb_view = head_alignment_service.prepare_display_frame(
        frame,
        channel_mode="rgb",
        contrast=1.0,
    )
    bgr_view = head_alignment_service.prepare_display_frame(
        frame,
        channel_mode="bgr",
        contrast=1.0,
    )

    assert rgb_view.tolist() == [[[10, 20, 30]]]
    assert bgr_view.tolist() == [[[30, 20, 10]]]


def test_render_alignment_view_supports_blend_and_difference(tmp_path: Path) -> None:
    reference_path = _write_image(tmp_path / "reference.png", _solid_bgr((0, 0, 255)))
    service = head_alignment_service.HeadAlignmentService(
        reference_image_path=reference_path,
        autostart=False,
    )
    service.set_latest_frame(_solid_bgr((255, 0, 0)), source_path="memory://live")

    blend = service.render_alignment_view(
        mode="blend",
        channel_mode="bgr",
        alpha=0.5,
        contrast=1.0,
        diff_gain=3.0,
    )
    difference = service.render_alignment_view(
        mode="difference",
        channel_mode="rgb",
        alpha=0.5,
        contrast=1.0,
        diff_gain=3.0,
    )

    assert blend.shape == (6, 8, 3)
    assert difference.shape == (6, 8, 3)
    assert int(difference.mean()) > 0


def test_create_app_serves_html_status_and_images(tmp_path: Path) -> None:
    reference_path = _write_image(tmp_path / "reference.png", _solid_bgr((0, 255, 0)))
    service = head_alignment_service.HeadAlignmentService(
        reference_image_path=reference_path,
        autostart=False,
    )
    service.set_latest_frame(_solid_bgr((255, 0, 0)), source_path="memory://live")
    client = TestClient(head_alignment_service.create_app(service=service, autostart=False))

    root = client.get("/")
    assert root.status_code == 200
    assert "Head Camera Alignment" in root.text
    assert "/api/stream/live.mjpg" in root.text

    status = client.get("/api/status")
    assert status.status_code == 200
    payload = status.json()
    assert payload["capture"]["has_frame"] is True
    assert payload["capture"]["capture_count"] == 1
    assert payload["reference"]["path"] == str(reference_path)

    reference = client.get("/api/frame/reference.png")
    assert reference.status_code == 200
    assert reference.headers["content-type"] == "image/png"

    live = client.get("/api/frame/live.jpg?channel_mode=bgr&contrast=1.5")
    assert live.status_code == 200
    assert live.headers["content-type"] == "image/jpeg"

    align = client.get("/api/frame/align.jpg?mode=difference&channel_mode=bgr&diff_gain=4")
    assert align.status_code == 200
    assert align.headers["content-type"] == "image/jpeg"


def test_config_update_and_mjpeg_stream(tmp_path: Path) -> None:
    reference_path = _write_image(tmp_path / "reference.png", _solid_bgr((0, 255, 255)))
    service = head_alignment_service.HeadAlignmentService(
        reference_image_path=reference_path,
        autostart=False,
    )
    service.set_latest_frame(_solid_bgr((255, 255, 0)), source_path="memory://live")
    client = TestClient(head_alignment_service.create_app(service=service, autostart=False))

    update = client.post("/api/config", json={"capture_interval_s": 0.4})
    assert update.status_code == 200
    assert update.json()["capture_interval_s"] == 0.4

    stream = client.get("/api/stream/live.mjpg?max_frames=1")
    assert stream.status_code == 200
    assert stream.headers["content-type"].startswith("multipart/x-mixed-replace")
    assert b"--frame" in stream.content
    assert b"Content-Type: image/jpeg" in stream.content
