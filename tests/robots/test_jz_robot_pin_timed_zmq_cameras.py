from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from lerobot.robots.jz_robot_pin_timed.orin_realsense_zmq import (
    CAMERA_PRESETS,
    DirectCameraPreset,
    DirectRealSenseZmqServer,
    build_message,
    encode_rgb_jpeg,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_three_camera_presets_keep_training_names_and_use_independent_ports() -> None:
    assert set(CAMERA_PRESETS) == {"camera_head", "camera_left", "camera_right"}
    assert CAMERA_PRESETS["camera_head"].serial_number == "318122304464"
    assert CAMERA_PRESETS["camera_left"].serial_number == "230422272306"
    assert CAMERA_PRESETS["camera_right"].serial_number == "230322272819"
    assert {preset.fps for preset in CAMERA_PRESETS.values()} == {30}
    assert {preset.port for preset in CAMERA_PRESETS.values()} == {5555, 5556, 5557}


def test_rgb_jpeg_encoding_does_not_swap_red_and_blue() -> None:
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[:, :, 0] = 240
    jpeg = encode_rgb_jpeg(rgb, quality=100)
    decoded_bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)

    assert decoded_rgb[:, :, 0].mean() > 230
    assert decoded_rgb[:, :, 2].mean() < 10


def test_message_keeps_legacy_zmq_fields_and_adds_timing() -> None:
    preset = DirectCameraPreset("camera_test", "123", 640, 480, 30, 6000)
    message = build_message(
        preset,
        sequence=7,
        capture_wall_ns=1_700_000_000_000_000_000,
        capture_monotonic_ns=123_000_000,
        encode_completed_monotonic_ns=125_000_000,
        jpeg=b"jpeg",
        jpeg_quality=95,
    )
    decoded = json.loads(message)

    assert decoded["protocol"] == "jz_realsense_zmq"
    assert decoded["protocol_version"] == 1
    assert base64.b64decode(decoded["images"]["camera_test"]) == b"jpeg"
    assert decoded["timestamps"]["camera_test"] == 1_700_000_000.0
    assert decoded["camera_timing"]["camera_test"] == {
        "sequence": 7,
        "timestamp_stage": "after_realsense_read_before_jpeg",
        "capture_wall_ns": 1_700_000_000_000_000_000,
        "capture_monotonic_ns": 123_000_000,
        "encode_completed_monotonic_ns": 125_000_000,
        "width": 640,
        "height": 480,
        "channels": 3,
        "pixel_format": "RGB8",
        "encoding": "jpeg",
        "jpeg_quality": 95,
        "payload_bytes": 4,
    }


@pytest.mark.parametrize("quality", [0, 101])
def test_server_rejects_invalid_jpeg_quality(quality: int) -> None:
    with pytest.raises(ValueError, match="jpeg_quality"):
        DirectRealSenseZmqServer(jpeg_quality=quality, presets=())


@pytest.mark.parametrize(
    "relative_path",
    [
        "my_devs/jz_robot_pin_timed/edge/start_direct_realsense_zmq.sh",
        "my_devs/jz_robot_pin_timed/edge/stop_direct_realsense_zmq.sh",
        "my_devs/jz_robot_pin_timed/edge/status_direct_realsense_zmq.sh",
    ],
)
def test_direct_camera_shell_scripts_pass_bash_syntax(relative_path: str) -> None:
    subprocess.run(["bash", "-n", str(REPO_ROOT / relative_path)], check=True)


def test_start_script_refuses_legacy_camera_owners_and_uses_lerobot_conda() -> None:
    script = (
        REPO_ROOT / "my_devs/jz_robot_pin_timed/edge/start_direct_realsense_zmq.sh"
    ).read_text()

    assert "robot_camera_node still owns" in script
    assert "camera_bridge_node is still running" in script
    assert "robot_bringup.service is still active" in script
    assert "conda run --no-capture-output -n lerobot python" in script
    assert "all cameras ready" in script


def test_stop_and_status_scripts_match_only_the_direct_camera_worker() -> None:
    stop_script = (
        REPO_ROOT / "my_devs/jz_robot_pin_timed/edge/stop_direct_realsense_zmq.sh"
    ).read_text()
    status_script = (
        REPO_ROOT / "my_devs/jz_robot_pin_timed/edge/status_direct_realsense_zmq.sh"
    ).read_text()

    expected_pattern = "^python([0-9.]*)? -m lerobot\\.robots\\.jz_robot_pin_timed\\.orin_realsense_zmq"
    assert expected_pattern in stop_script
    assert expected_pattern in status_script
    assert 'pgrep -af "$WORKER_PATTERN"' in status_script
