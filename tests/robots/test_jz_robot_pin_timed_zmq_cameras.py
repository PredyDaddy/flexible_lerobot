from __future__ import annotations

import base64
import json
import socket
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import zmq

from lerobot.robots.jz_robot_pin_timed.orin_realsense_zmq import (
    CAMERA_PRESETS,
    CameraPublisherWorker,
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


def test_message_accepts_optional_capture_stage_diagnostics() -> None:
    preset = DirectCameraPreset("camera_test", "123", 640, 480, 30, 6000)
    message = build_message(
        preset,
        sequence=8,
        capture_wall_ns=1_700_000_000_000_000_000,
        capture_monotonic_ns=200_000_000,
        encode_completed_monotonic_ns=203_000_000,
        jpeg=b"jpeg",
        jpeg_quality=75,
        image_b64=base64.b64encode(b"jpeg").decode("ascii"),
        extra_timing={
            "read_enter_monotonic_ns": 190_000_000,
            "read_return_monotonic_ns": 200_000_000,
            "realsense_frame_number": 42,
            "realsense_device_timestamp_ms": 1234.5,
        },
    )

    timing = json.loads(message)["camera_timing"]["camera_test"]
    assert timing["read_enter_monotonic_ns"] == 190_000_000
    assert timing["read_return_monotonic_ns"] == 200_000_000
    assert timing["realsense_frame_number"] == 42
    assert timing["realsense_device_timestamp_ms"] == 1234.5


def test_capture_continues_while_publisher_encoding_is_slow(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCamera:
        def __init__(self, _config: object) -> None:
            self.is_connected = False

        def connect(self) -> None:
            self.is_connected = True

        def read(self, timeout_ms: int) -> np.ndarray:
            assert timeout_ms == 500
            time.sleep(0.001)
            return np.zeros((8, 8, 3), dtype=np.uint8)

        def disconnect(self) -> None:
            self.is_connected = False

    def slow_encode(_image: np.ndarray, _quality: int) -> bytes:
        time.sleep(0.02)
        return b"jpeg"

    monkeypatch.setattr(
        "lerobot.robots.jz_robot_pin_timed.orin_realsense_zmq.encode_rgb_jpeg",
        slow_encode,
    )
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]

    context = zmq.Context()
    stop_event = threading.Event()
    worker = CameraPublisherWorker(
        DirectCameraPreset("camera_test", "123", 8, 8, 30, port),
        context,
        bind_host="127.0.0.1",
        jpeg_quality=75,
        count=5,
        stop_event=stop_event,
        camera_factory=FakeCamera,
    )
    try:
        worker.start()
        worker.join(timeout=5)
        assert not worker.thread.is_alive()
        assert worker.exception is None
        assert worker.frames_captured == 5
        assert worker.frames_sent == 5
        assert len(worker.capture_intervals_ms) == 5 - 1
        assert max(worker.capture_intervals_ms) < 10
    finally:
        stop_event.set()
        worker.join(timeout=1)
        context.term()


@pytest.mark.parametrize("quality", [0, 101])
def test_server_rejects_invalid_jpeg_quality(quality: int) -> None:
    with pytest.raises(ValueError, match="jpeg_quality"):
        DirectRealSenseZmqServer(jpeg_quality=quality, presets=())


def test_server_uses_one_non_daemon_process_per_camera() -> None:
    presets = (
        DirectCameraPreset("camera_a", "1", 8, 8, 30, 6001),
        DirectCameraPreset("camera_b", "2", 8, 8, 30, 6002),
    )
    server = DirectRealSenseZmqServer(presets=presets)
    try:
        assert server.mp_context.get_start_method() == "spawn"
        assert [process.name for process in server.processes] == [
            "jz-realsense-process-camera_a",
            "jz-realsense-process-camera_b",
        ]
        assert all(not process.daemon for process in server.processes)
    finally:
        server.ready_queue.close()
        server.error_queue.close()


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


def test_start_script_refuses_legacy_camera_owners_and_allows_camera_free_bringup() -> None:
    script = (
        REPO_ROOT / "my_devs/jz_robot_pin_timed/edge/start_direct_realsense_zmq.sh"
    ).read_text()

    assert "robot_camera_node still owns" in script
    assert "camera_bridge_node is still running" in script
    assert "robot_bringup.service active without legacy camera owners; continuing" in script
    assert "refusing to start: robot_bringup.service" not in script
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
