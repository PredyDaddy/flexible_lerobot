from __future__ import annotations

import base64
import json
import threading
import time

import cv2
import numpy as np
import pytest
import zmq

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig
from lerobot.robots.jz_robot_pin_timed.config_jz_robot_pin_timed import JZRobotPinTimedConfig
from lerobot.robots.jz_robot_pin_timed.jz_robot_pin_timed import JZRobotPinTimed
from lerobot.robots.jz_robot_pin_timed.timestamped_zmq_camera import (
    TimestampedZMQCamera,
    ZMQCameraProtocolError,
)


def make_config(port: int = 5555, **overrides) -> ZMQCameraConfig:
    values = {
        "server_address": "127.0.0.1",
        "port": port,
        "camera_name": "camera_head",
        "fps": 30,
        "width": 32,
        "height": 24,
        "color_mode": ColorMode.RGB,
        "timeout_ms": 500,
    }
    values.update(overrides)
    return ZMQCameraConfig(**values)


def encode_rgb(rgb: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    assert ok
    return encoded.tobytes()


def make_message(sequence: int, *, camera_name: str = "camera_head") -> bytes:
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    rgb[:, :, 0] = 240
    jpeg = encode_rgb(rgb)
    return json.dumps(
        {
            "protocol": "jz_realsense_zmq",
            "protocol_version": 1,
            "timestamps": {camera_name: 1_700_000_000.0},
            "images": {camera_name: base64.b64encode(jpeg).decode("ascii")},
            "camera_timing": {
                camera_name: {
                    "sequence": sequence,
                    "timestamp_stage": "after_realsense_read_before_jpeg",
                    "capture_wall_ns": 1_700_000_000_000_000_000 + sequence,
                    "capture_monotonic_ns": 1_000_000_000 + sequence * 33_333_333,
                    "encode_completed_monotonic_ns": 1_001_000_000 + sequence * 33_333_333,
                    "width": 32,
                    "height": 24,
                    "channels": 3,
                    "pixel_format": "RGB8",
                    "encoding": "jpeg",
                    "jpeg_quality": 100,
                    "payload_bytes": len(jpeg),
                }
            },
        },
        separators=(",", ":"),
    ).encode()


def test_decode_strict_protocol_converts_opencv_bgr_to_rgb() -> None:
    camera = TimestampedZMQCamera(make_config())

    frame = camera._decode_message(make_message(7), 2_000_000_000, 3_000_000_000)

    assert frame.sequence == 7
    assert frame.image[:, :, 0].mean() > 230
    assert frame.image[:, :, 2].mean() < 10
    assert frame.camera_timing["pixel_format"] == "RGB8"
    assert frame.decode_timing["total_decode_ms"] >= 0
    assert frame.decode_timing["jpeg_decode_ms"] >= 0


def test_timed_robot_builds_zmq_camera_without_changing_observation_key() -> None:
    config = make_config()
    robot = JZRobotPinTimed(JZRobotPinTimedConfig(zmq_cameras={"camera_head": config}))

    assert isinstance(robot.cameras["camera_head"], TimestampedZMQCamera)
    assert robot.observation_features["camera_head"] == (24, 32, 3)


def test_timed_config_rejects_mixed_camera_transports() -> None:
    from lerobot.robots.jz_robot_udp.config_jz_robot_udp import RTSPCameraConfig

    with pytest.raises(ValueError, match="mutually exclusive"):
        JZRobotPinTimedConfig(
            zmq_cameras={"camera_head": make_config()},
            rtsp_cameras={"camera_head": RTSPCameraConfig("rtsp://127.0.0.1/test")},
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda data: data.update(protocol_version=2), "expected jz_realsense_zmq"),
        (
            lambda data: data["images"].update(camera_left=data["images"].pop("camera_head")),
            "only camera 'camera_head'",
        ),
        (lambda data: data["camera_timing"]["camera_head"].update(payload_bytes=1), "payload size"),
    ],
)
def test_decode_rejects_wrong_protocol_camera_or_payload(mutation, match: str) -> None:
    camera = TimestampedZMQCamera(make_config())
    data = json.loads(make_message(1))
    mutation(data)

    with pytest.raises(ZMQCameraProtocolError, match=match):
        camera._decode_message(json.dumps(data).encode(), 2_000_000_000, 3_000_000_000)


def test_live_subscriber_uses_latest_frames_and_stops_without_thread_residue() -> None:
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER, 0)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")
    stop = threading.Event()

    def publish() -> None:
        sequence = 0
        while not stop.is_set():
            sequence += 1
            publisher.send(make_message(sequence))
            time.sleep(0.005)

    thread = threading.Thread(target=publish)
    thread.start()
    camera = TimestampedZMQCamera(make_config(port), buffer_size=2)
    try:
        camera.connect()
        first = camera.read_timed()
        time.sleep(0.03)
        latest = camera.read_timed()
        assert latest.sequence > first.sequence
        assert latest.timing(time.monotonic_ns())["protocol"] == "jz_realsense_zmq"
        assert camera.diagnostics["invalid_messages"] == 0
    finally:
        if camera.is_connected:
            camera.disconnect()
        stop.set()
        thread.join(timeout=1)
        publisher.close()
        context.term()

    assert not any(item.name == "timed_zmq_reader_camera_head" for item in threading.enumerate())
    assert not any(item.name == "timed_zmq_decoder_camera_head" for item in threading.enumerate())


def test_socket_receiver_keeps_draining_while_decoder_is_blocked() -> None:
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER, 0)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")
    stop = threading.Event()

    def publish() -> None:
        sequence = 0
        while not stop.is_set():
            sequence += 1
            publisher.send(make_message(sequence))
            time.sleep(0.003)

    publisher_thread = threading.Thread(target=publish)
    publisher_thread.start()
    camera = TimestampedZMQCamera(make_config(port))
    release_decode = threading.Event()
    try:
        camera.connect()
        original_decode = camera._decode_message
        decode_started = threading.Event()

        def blocked_decode(*args, **kwargs):
            if not decode_started.is_set():
                decode_started.set()
                assert release_decode.wait(timeout=1)
            return original_decode(*args, **kwargs)

        camera._decode_message = blocked_decode
        assert decode_started.wait(timeout=1)
        received_before = camera.diagnostics["received_messages"]
        time.sleep(0.05)
        blocked_diagnostics = camera.diagnostics

        assert blocked_diagnostics["received_messages"] > received_before + 5
        assert blocked_diagnostics["raw_queue_drops"] > 0

        release_decode.set()
        deadline = time.monotonic() + 1
        while camera.diagnostics["sequence_gaps"] == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert camera.diagnostics["sequence_gaps"] > 0
        assert camera.diagnostics["max_queue_delay_ms"] >= 40
        assert camera.diagnostics["max_capture_interarrival_ms"] > 0
    finally:
        release_decode.set()
        stop.set()
        publisher_thread.join(timeout=1)
        if camera._reader_thread is not None or camera._decoder_thread is not None:
            camera.disconnect()
        publisher.close()
        context.term()


def test_aligned_read_waits_for_a_genuinely_new_frame() -> None:
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER, 0)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")
    camera = TimestampedZMQCamera(make_config(port))
    stop = threading.Event()

    def publish_initial_frame() -> None:
        while not stop.is_set():
            publisher.send(make_message(1))
            time.sleep(0.005)

    initial_thread = threading.Thread(target=publish_initial_frame)
    initial_thread.start()
    try:
        camera.connect()
        stop.set()
        initial_thread.join(timeout=1)
        time.sleep(0.06)
        target_monotonic_ns = time.monotonic_ns()
        timer = threading.Timer(0.01, lambda: publisher.send(make_message(2)))
        timer.start()

        frame = camera.read_timed_nearest(
            target_monotonic_ns,
            max_receive_skew_ms=50,
            wait_timeout_ms=100,
        )
        timer.join(timeout=1)

        assert frame.sequence == 2
        assert camera.diagnostics["accepted_frames"] >= 2
    finally:
        stop.set()
        initial_thread.join(timeout=1)
        if camera.is_connected:
            camera.disconnect()
        publisher.close()
        context.term()


def test_aligned_read_fails_instead_of_reusing_an_old_frame() -> None:
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER, 0)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")
    camera = TimestampedZMQCamera(make_config(port))
    stop = threading.Event()

    def publish_initial_frame() -> None:
        while not stop.is_set():
            publisher.send(make_message(1))
            time.sleep(0.005)

    thread = threading.Thread(target=publish_initial_frame)
    thread.start()
    try:
        camera.connect()
        stop.set()
        thread.join(timeout=1)
        time.sleep(0.02)

        with pytest.raises(TimeoutError, match="did not produce an aligned frame"):
            camera.read_timed_nearest(
                time.monotonic_ns(),
                max_receive_skew_ms=5,
                wait_timeout_ms=10,
            )
    finally:
        stop.set()
        thread.join(timeout=1)
        if camera.is_connected:
            camera.disconnect()
        publisher.close()
        context.term()
