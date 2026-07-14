from __future__ import annotations

import argparse
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

from lerobot.robots.jz_robot_pin_timed import orin_realsense_zmq as camera_module
from lerobot.robots.jz_robot_pin_timed.orin_realsense_zmq import (
    CAMERA_PRESETS,
    COLOR_CONFIG_SCHEMA_VERSION,
    DEFAULT_MAX_RECONNECT_ATTEMPTS,
    DEFAULT_RECONNECT_BACKOFF_S,
    MAX_RECONNECT_BACKOFF_S,
    CameraPublisherWorker,
    ColorConfiguration,
    DirectCameraPreset,
    DirectRealSenseZmqServer,
    build_message,
    camera_preset_ready_details,
    configure_realsense_color_sensor,
    empty_camera_metrics,
    encode_rgb_jpeg,
    parse_camera_spec,
    save_color_diagnostic_sample,
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
    rgb = np.zeros((32, 64, 3), dtype=np.uint8)
    rgb[:, :32, 0] = 240
    rgb[:, 32:, 2] = 240
    jpeg = encode_rgb_jpeg(rgb, quality=100)
    decoded_bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)

    assert decoded_rgb[:, :28, 0].mean() > 230
    assert decoded_rgb[:, :28, 2].mean() < 10
    assert decoded_rgb[:, 36:, 2].mean() > 230
    assert decoded_rgb[:, 36:, 0].mean() < 10


class FakeOptionRange:
    def __init__(self, minimum: float, maximum: float, step: float, default: float) -> None:
        self.min = minimum
        self.max = maximum
        self.step = step
        self.default = default


class FakeColorSensor:
    def __init__(self, *, awb_supported: bool = True, sticky_awb: bool = False) -> None:
        self.awb_supported = awb_supported
        self.sticky_awb = sticky_awb
        self.values = {
            camera_module.rs.option.enable_auto_white_balance: 1.0,
            camera_module.rs.option.white_balance: 4600.0,
        }
        self.ranges = {
            camera_module.rs.option.enable_auto_white_balance: FakeOptionRange(0, 1, 1, 1),
            camera_module.rs.option.white_balance: FakeOptionRange(2800, 6500, 10, 4600),
        }
        self.events: list[tuple[str, object, float | None]] = []

    def supports(self, option: object) -> bool:
        self.events.append(("supports", option, None))
        return option != camera_module.rs.option.enable_auto_white_balance or self.awb_supported

    def get_option_range(self, option: object) -> FakeOptionRange:
        self.events.append(("range", option, None))
        return self.ranges[option]

    def set_option(self, option: object, value: float) -> None:
        self.events.append(("set", option, float(value)))
        if option == camera_module.rs.option.enable_auto_white_balance and self.sticky_awb:
            return
        self.values[option] = float(value)

    def get_option(self, option: object) -> float:
        self.events.append(("get", option, None))
        return self.values[option]


class FakeDevice:
    def __init__(self, sensor: FakeColorSensor, *, first_color_available: bool = True) -> None:
        self.sensor = sensor
        self.first_color_available = first_color_available

    def first_color_sensor(self) -> FakeColorSensor:
        if not self.first_color_available:
            raise RuntimeError("no first color sensor")
        return self.sensor

    def query_sensors(self) -> list[FakeColorSensor]:
        return [self.sensor]


class FakeProfile:
    def __init__(self, sensor: FakeColorSensor, *, first_color_available: bool = True) -> None:
        self.device = FakeDevice(sensor, first_color_available=first_color_available)

    def get_device(self) -> FakeDevice:
        return self.device


def fake_camera_with_sensor(sensor: FakeColorSensor, *, first_color_available: bool = True) -> object:
    return type(
        "FakeRealSense",
        (),
        {"rs_profile": FakeProfile(sensor, first_color_available=first_color_available)},
    )()


def test_color_sensor_disables_awb_and_applies_each_camera_fixed_value() -> None:
    head_sensor = FakeColorSensor()
    left_sensor = FakeColorSensor()
    head = DirectCameraPreset("camera_head", "head", 8, 8, 30, 6001, white_balance=4500)
    left = DirectCameraPreset("camera_left", "left", 8, 8, 30, 6002, white_balance=4900)

    head_config = configure_realsense_color_sensor(fake_camera_with_sensor(head_sensor), head)
    left_config = configure_realsense_color_sensor(fake_camera_with_sensor(left_sensor), left)

    assert head_config.auto_white_balance == 0
    assert head_config.white_balance == 4500
    assert left_config.auto_white_balance == 0
    assert left_config.white_balance == 4900
    assert head_config.white_balance_source == left_config.white_balance_source == "configured"
    assert head_sensor.events.index(
        ("set", camera_module.rs.option.enable_auto_white_balance, 0.0)
    ) < head_sensor.events.index(("set", camera_module.rs.option.white_balance, 4500.0))


def test_color_sensor_without_config_uses_current_value_after_disabling_awb() -> None:
    sensor = FakeColorSensor()
    preset = DirectCameraPreset("camera_head", "head", 8, 8, 30, 6001)

    configuration = configure_realsense_color_sensor(fake_camera_with_sensor(sensor), preset)

    assert configuration.auto_white_balance == 0
    assert configuration.white_balance == 4600
    assert configuration.white_balance_source == "device_current"
    assert ("set", camera_module.rs.option.white_balance, 4600.0) not in sensor.events


def test_d405_style_device_uses_unique_sensor_supporting_both_color_options() -> None:
    sensor = FakeColorSensor()
    preset = DirectCameraPreset("camera_left", "left", 8, 8, 30, 6001)

    configuration = configure_realsense_color_sensor(
        fake_camera_with_sensor(sensor, first_color_available=False), preset
    )

    assert configuration.auto_white_balance == 0
    assert configuration.white_balance == 4600


def test_color_sensor_rejects_out_of_range_fixed_value() -> None:
    preset = DirectCameraPreset("camera_head", "head", 8, 8, 30, 6001, white_balance=7000)

    with pytest.raises(RuntimeError, match=r"camera_id=camera_head.*requested=7000.*2800,6500"):
        configure_realsense_color_sensor(fake_camera_with_sensor(FakeColorSensor()), preset)


def test_color_sensor_rejects_missing_awb_control() -> None:
    preset = DirectCameraPreset("camera_head", "head", 8, 8, 30, 6001)

    with pytest.raises(RuntimeError, match=r"camera_id=camera_head.*enable_auto_white_balance.*unsupported"):
        configure_realsense_color_sensor(
            fake_camera_with_sensor(FakeColorSensor(awb_supported=False)), preset
        )


def test_color_sensor_rejects_awb_readback_that_remains_enabled() -> None:
    preset = DirectCameraPreset("camera_head", "head", 8, 8, 30, 6001)

    with pytest.raises(RuntimeError, match=r"camera_id=camera_head.*requested=0 actual=1"):
        configure_realsense_color_sensor(fake_camera_with_sensor(FakeColorSensor(sticky_awb=True)), preset)


def test_camera_timing_records_actual_fixed_color_configuration() -> None:
    configuration = ColorConfiguration(0, 4700, "configured", 2800, 6500, 10, 4600)
    message = build_message(
        DirectCameraPreset("camera_test", "123", 8, 8, 30, 6000),
        sequence=1,
        capture_wall_ns=10,
        capture_monotonic_ns=20,
        encode_completed_monotonic_ns=30,
        jpeg=b"jpeg",
        jpeg_quality=75,
        extra_timing=configuration.timing_fields(),
    )

    decoded = json.loads(message)
    timing = decoded["camera_timing"]["camera_test"]
    assert timing["auto_white_balance"] == 0
    assert timing["white_balance"] == 4700
    assert timing["white_balance_source"] == "configured"
    assert timing["color_config_schema_version"] == COLOR_CONFIG_SCHEMA_VERSION
    assert decoded["protocol"] == "jz_realsense_zmq"
    assert decoded["protocol_version"] == 1
    assert base64.b64decode(decoded["images"]["camera_test"]) == b"jpeg"


def test_diagnostic_sample_preserves_encoder_input_rgb_and_exact_sent_jpeg(
    tmp_path: Path,
) -> None:
    rgb = np.zeros((16, 32, 3), dtype=np.uint8)
    rgb[:, :16, 0] = 255
    rgb[:, 16:, 2] = 255
    jpeg = encode_rgb_jpeg(rgb, quality=100)
    preset = DirectCameraPreset("camera_test", "123", 32, 16, 30, 6000)

    rgb_path, jpeg_path = save_color_diagnostic_sample(
        tmp_path, preset, sequence=7, rgb=rgb, jpeg=jpeg
    )

    decoded_png = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
    assert np.array_equal(decoded_png, rgb)
    assert jpeg_path.read_bytes() == jpeg


def test_worker_disables_awb_before_capture_ready_and_first_read() -> None:
    sensor = FakeColorSensor()

    class FakeRealSense:
        def __init__(self, _config: object) -> None:
            self.rs_profile = FakeProfile(sensor)
            self.is_connected = False

        def connect(self) -> None:
            self.is_connected = True

        def read(self, timeout_ms: int) -> np.ndarray:
            assert timeout_ms == 500
            assert sensor.values[camera_module.rs.option.enable_auto_white_balance] == 0
            assert sensor.values[camera_module.rs.option.white_balance] == 4700
            return np.zeros((8, 8, 3), dtype=np.uint8)

        def disconnect(self) -> None:
            self.is_connected = False

    context = zmq.Context()
    stop_event = threading.Event()
    worker = CameraPublisherWorker(
        DirectCameraPreset("camera_test", "123", 8, 8, 30, 6009, white_balance=4700),
        context,
        bind_host="127.0.0.1",
        jpeg_quality=75,
        preview_enabled=False,
        count=1,
        stop_event=stop_event,
        camera_factory=FakeRealSense,
        enforce_color_configuration=True,
    )
    try:
        worker.start()
        worker.join(timeout=5)

        assert not worker.thread.is_alive()
        assert worker.exception is None
        assert worker.ready.is_set()
        assert worker.frames_captured == 1
        assert worker.color_configuration is not None
        assert worker.color_configuration.auto_white_balance == 0
        assert worker.color_configuration.white_balance == 4700
    finally:
        stop_event.set()
        worker.join(timeout=1)
        context.term()


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


def test_capture_read_failure_reconnects_with_generation_and_formal_drop_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecoveringCamera:
        instances = []

        def __init__(self, _config: object) -> None:
            self.instance_index = len(self.instances)
            self.is_connected = False
            self.read_count = 0
            self.disconnect_count = 0
            self.instances.append(self)

        def connect(self) -> None:
            self.is_connected = True

        def read(self, timeout_ms: int) -> np.ndarray:
            assert timeout_ms == 500
            self.read_count += 1
            if self.instance_index == 0:
                raise RuntimeError("synthetic read disconnect")
            return np.zeros((8, 8, 3), dtype=np.uint8)

        def disconnect(self) -> None:
            self.disconnect_count += 1
            self.is_connected = False

    class FakeFormalSink:
        instances = []

        def __init__(self, **_kwargs) -> None:
            self.records = []
            self.instances.append(self)

        def capture_marker(self) -> dict[str, str]:
            return {"episode_id": "episode-test"}

        def bind_capture_context(self, **_kwargs) -> dict[str, str]:
            return {"episode_id": "episode-test"}

        def transition_to_context(self, _context: object) -> bool:
            return True

        def append_bound_frame(
            self, _context: object, _jpeg: bytes, **metadata: object
        ) -> None:
            self.records.append(metadata)

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "lerobot.robots.jz_robot_pin_timed.orin_realsense_zmq.MarkerDrivenCameraSink",
        FakeFormalSink,
    )
    context = zmq.Context()
    stop_event = threading.Event()
    worker = CameraPublisherWorker(
        DirectCameraPreset("camera_test", "123", 8, 8, 30, 6008),
        context,
        bind_host="127.0.0.1",
        jpeg_quality=95,
        preview_enabled=False,
        max_reconnect_attempts=2,
        reconnect_backoff_s=0,
        count=2,
        stop_event=stop_event,
        camera_factory=RecoveringCamera,
        session_marker="/tmp/active_episode.json",
        raw_root="/tmp/raw",
    )
    try:
        worker.start()
        worker.join(timeout=5)

        assert not worker.thread.is_alive()
        assert worker.exception is None
        assert len(RecoveringCamera.instances) == 2
        assert [camera.disconnect_count for camera in RecoveringCamera.instances] == [1, 1]
        records = FakeFormalSink.instances[0].records
        assert [record["reconnect_generation"] for record in records] == [2, 2]
        assert [record["drop_count_since_previous"] for record in records] == [1, 0]
        metrics = worker.metrics_snapshot()
        assert {key: metrics[key] for key in (
            "frames_captured",
            "frames_processed",
            "frames_previewed",
            "reconnect_generation",
            "reconnect_attempts",
            "reconnect_successes",
            "reconnect_failures",
            "reconnect_drop_count",
            "device_frame_drop_count",
            "formal_drop_evidence_count",
            "preview_drop_count",
        )} == {
            "frames_captured": 2,
            "frames_processed": 2,
            "frames_previewed": 0,
            "reconnect_generation": 2,
            "reconnect_attempts": 1,
            "reconnect_successes": 1,
            "reconnect_failures": 1,
            "reconnect_drop_count": 1,
            "device_frame_drop_count": 0,
            "formal_drop_evidence_count": 1,
            "preview_drop_count": 0,
        }
        assert metrics["first_capture_monotonic_ns"] > 0
        assert metrics["last_capture_monotonic_ns"] >= metrics["first_capture_monotonic_ns"]
        assert metrics["capture_queue_depth"] == 0
        assert metrics["preview_queue_depth"] == 0
        assert metrics["capture_interval_p95_us"] >= 0
        assert metrics["jpeg_encode_p95_us"] >= 0
    finally:
        stop_event.set()
        worker.join(timeout=1)
        context.term()


def test_capture_connect_failures_fault_only_after_reconnect_budget_is_exhausted() -> None:
    class FailingConnectCamera:
        instances = []

        def __init__(self, _config: object) -> None:
            self.is_connected = False
            self.disconnect_count = 0
            self.instances.append(self)

        def connect(self) -> None:
            self.is_connected = True
            raise RuntimeError("synthetic connect failure")

        def read(self, _timeout_ms: int) -> np.ndarray:
            raise AssertionError("read must not run after connect failure")

        def disconnect(self) -> None:
            self.disconnect_count += 1
            self.is_connected = False

    context = zmq.Context()
    stop_event = threading.Event()
    worker = CameraPublisherWorker(
        DirectCameraPreset("camera_test", "123", 8, 8, 30, 6008),
        context,
        bind_host="127.0.0.1",
        jpeg_quality=95,
        preview_enabled=False,
        max_reconnect_attempts=2,
        reconnect_backoff_s=0,
        count=1,
        stop_event=stop_event,
        camera_factory=FailingConnectCamera,
    )
    try:
        worker.start()
        worker.join(timeout=5)

        assert not worker.thread.is_alive()
        assert isinstance(worker.exception, RuntimeError)
        assert "reconnect exhausted" in str(worker.exception)
        assert len(FailingConnectCamera.instances) == 3
        assert all(camera.disconnect_count == 1 for camera in FailingConnectCamera.instances)
        metrics = worker.metrics_snapshot()
        assert metrics["reconnect_attempts"] == 2
        assert metrics["reconnect_failures"] == 3
        assert metrics["reconnect_generation"] == 0
        assert metrics["reconnect_successes"] == 0
        assert stop_event.is_set()
    finally:
        stop_event.set()
        worker.join(timeout=1)
        context.term()


def test_capture_connect_failure_can_recover_with_a_new_camera_instance() -> None:
    class RecoveringConnectCamera:
        instances = []

        def __init__(self, _config: object) -> None:
            self.instance_index = len(self.instances)
            self.is_connected = False
            self.disconnect_count = 0
            self.instances.append(self)

        def connect(self) -> None:
            self.is_connected = True
            if self.instance_index == 0:
                raise RuntimeError("synthetic first connect failure")

        def read(self, timeout_ms: int) -> np.ndarray:
            assert timeout_ms == 500
            return np.zeros((8, 8, 3), dtype=np.uint8)

        def disconnect(self) -> None:
            self.disconnect_count += 1
            self.is_connected = False

    context = zmq.Context()
    stop_event = threading.Event()
    worker = CameraPublisherWorker(
        DirectCameraPreset("camera_test", "123", 8, 8, 30, 6008),
        context,
        bind_host="127.0.0.1",
        jpeg_quality=95,
        preview_enabled=False,
        max_reconnect_attempts=1,
        reconnect_backoff_s=0,
        count=1,
        stop_event=stop_event,
        camera_factory=RecoveringConnectCamera,
    )
    try:
        worker.start()
        worker.join(timeout=5)

        assert worker.exception is None
        assert worker.frames_captured == 1
        assert len(RecoveringConnectCamera.instances) == 2
        assert [camera.disconnect_count for camera in RecoveringConnectCamera.instances] == [1, 1]
        metrics = worker.metrics_snapshot()
        assert metrics["reconnect_attempts"] == 1
        assert metrics["reconnect_successes"] == 1
        assert metrics["reconnect_failures"] == 1
        assert metrics["reconnect_generation"] == 1
        assert metrics["reconnect_drop_count"] == 1
    finally:
        stop_event.set()
        worker.join(timeout=1)
        context.term()


def test_slow_preview_never_backpressures_formal_spool_or_reencodes_formal_jpeg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    class FakeFormalSink:
        instances = []

        def __init__(self, **_kwargs) -> None:
            self.sequences = []
            self.generations = []
            self.drop_counts = []
            FakeFormalSink.instances.append(self)

        def capture_marker(self) -> dict[str, str]:
            return {"episode_id": "episode-test"}

        def bind_capture_context(self, **_kwargs) -> dict[str, str]:
            return {"episode_id": "episode-test"}

        def transition_to_context(self, _context: object) -> bool:
            return True

        def append_bound_frame(
            self,
            _context: object,
            _jpeg: bytes,
            *,
            camera_sequence: int,
            reconnect_generation: int,
            drop_count_since_previous: int,
            **_kwargs,
        ) -> None:
            self.sequences.append(camera_sequence)
            self.generations.append(reconnect_generation)
            self.drop_counts.append(drop_count_since_previous)

        def close(self) -> None:
            pass

    encode_qualities = []

    def formal_encode(_image: np.ndarray, quality: int) -> bytes:
        encode_qualities.append(quality)
        return b"formal-jpeg"

    def slow_preview_message(*_args, **_kwargs) -> str:
        time.sleep(0.03)
        return "{}"

    monkeypatch.setattr(
        "lerobot.robots.jz_robot_pin_timed.orin_realsense_zmq.MarkerDrivenCameraSink",
        FakeFormalSink,
    )
    monkeypatch.setattr(
        "lerobot.robots.jz_robot_pin_timed.orin_realsense_zmq.encode_rgb_jpeg",
        formal_encode,
    )
    monkeypatch.setattr(
        "lerobot.robots.jz_robot_pin_timed.orin_realsense_zmq.build_message",
        slow_preview_message,
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
        jpeg_quality=95,
        preview_jpeg_quality=75,
        count=20,
        stop_event=stop_event,
        camera_factory=FakeCamera,
        session_marker="/tmp/active_episode.json",
        raw_root="/tmp/raw",
    )
    try:
        worker.start()
        worker.join(timeout=5)

        assert not worker.thread.is_alive()
        assert worker.exception is None
        assert worker.frames_captured == 20
        assert worker.frames_sent == 20
        assert FakeFormalSink.instances[0].sequences == list(range(1, 21))
        assert FakeFormalSink.instances[0].generations == [1] * 20
        assert FakeFormalSink.instances[0].drop_counts == [0] * 20
        assert encode_qualities == [95] * 20
        assert worker.preview_drops > 0
        assert worker.metrics_snapshot()["reconnect_failures"] == 0
        assert worker.metrics_snapshot()["formal_drop_evidence_count"] == 0
        assert max(worker.capture_intervals_ms) < 10
    finally:
        stop_event.set()
        worker.join(timeout=1)
        context.term()


def test_preview_disabled_skips_network_jpeg_when_no_episode_is_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCamera:
        def __init__(self, _config: object) -> None:
            self.is_connected = False

        def connect(self) -> None:
            self.is_connected = True

        def read(self, timeout_ms: int) -> np.ndarray:
            assert timeout_ms == 500
            return np.zeros((8, 8, 3), dtype=np.uint8)

        def disconnect(self) -> None:
            self.is_connected = False

    def unexpected_encode(_image: np.ndarray, _quality: int) -> bytes:
        raise AssertionError("preview-disabled idle capture must not encode JPEG")

    monkeypatch.setattr(
        "lerobot.robots.jz_robot_pin_timed.orin_realsense_zmq.encode_rgb_jpeg",
        unexpected_encode,
    )
    context = zmq.Context()
    stop_event = threading.Event()
    worker = CameraPublisherWorker(
        DirectCameraPreset("camera_test", "123", 8, 8, 30, 6009),
        context,
        bind_host="127.0.0.1",
        jpeg_quality=95,
        preview_enabled=False,
        preview_jpeg_quality=75,
        count=2,
        stop_event=stop_event,
        camera_factory=FakeCamera,
    )
    try:
        worker.start()
        worker.join(timeout=5)
        assert not worker.thread.is_alive()
        assert worker.exception is None
        assert worker.frames_captured == 2
        assert worker.frames_sent == 2
    finally:
        stop_event.set()
        worker.join(timeout=1)
        context.term()


@pytest.mark.parametrize("quality", [0, 101])
def test_server_rejects_invalid_jpeg_quality(quality: int) -> None:
    with pytest.raises(ValueError, match="jpeg_quality"):
        DirectRealSenseZmqServer(jpeg_quality=quality, presets=())


@pytest.mark.parametrize("quality", [0, 101])
def test_server_rejects_invalid_preview_jpeg_quality(quality: int) -> None:
    with pytest.raises(ValueError, match="preview_jpeg_quality"):
        DirectRealSenseZmqServer(preview_jpeg_quality=quality, presets=())


@pytest.mark.parametrize("attempts", [True, -1, 1.5])
def test_server_rejects_invalid_reconnect_attempt_budget(attempts: object) -> None:
    with pytest.raises(ValueError, match="max_reconnect_attempts"):
        DirectRealSenseZmqServer(max_reconnect_attempts=attempts, presets=())


@pytest.mark.parametrize("backoff_s", [True, -0.1, float("nan"), float("inf")])
def test_server_rejects_invalid_reconnect_backoff(backoff_s: object) -> None:
    with pytest.raises(ValueError, match="reconnect_backoff_s"):
        DirectRealSenseZmqServer(reconnect_backoff_s=backoff_s, presets=())


def test_server_keeps_formal_and_preview_quality_as_separate_contracts() -> None:
    server = DirectRealSenseZmqServer(
        jpeg_quality=95,
        preview_enabled=False,
        preview_jpeg_quality=75,
        presets=(),
    )
    try:
        assert server.jpeg_quality == 95
        assert server.preview_jpeg_quality == 75
        assert server.preview_enabled is False
        assert server.ready_details()["camera_presets"] == []
    finally:
        server.ready_queue.close()
        server.metrics_queue.close()
        server.error_queue.close()


def test_structured_camera_spec_drives_complete_ready_contract() -> None:
    preset = parse_camera_spec(
        json.dumps(
            {
                "camera_id": "camera_test",
                "serial_number": "serial-test",
                "width": 320,
                "height": 240,
                "fps": 15,
                "preview_port": 6000,
                "jpeg_quality": 88,
            }
        )
    )

    assert camera_preset_ready_details(preset, default_jpeg_quality=95) == {
        "camera_id": "camera_test",
        "serial_number": "serial-test",
        "width": 320,
        "height": 240,
        "fps": 15,
        "preview_port": 6000,
        "pixel_format": "rgb8",
        "jpeg_quality": 88,
        "white_balance_requested": None,
    }
    server = DirectRealSenseZmqServer(
        bind_host="127.0.0.1",
        jpeg_quality=95,
        preview_jpeg_quality=75,
        presets=(preset,),
    )
    try:
        assert server.ready_details() == {
            "camera_presets": [
                camera_preset_ready_details(preset, default_jpeg_quality=95)
            ],
            "bind_host": "127.0.0.1",
            "preview_enabled": True,
            "preview_jpeg_quality": 75,
            "formal_local_spool": False,
            "diagnostic_dir": None,
            "capture_reconnect": {
                "max_attempts": DEFAULT_MAX_RECONNECT_ATTEMPTS,
                "backoff_s": DEFAULT_RECONNECT_BACKOFF_S,
                "max_backoff_s": MAX_RECONNECT_BACKOFF_S,
            },
            "camera_metrics": {"camera_test": empty_camera_metrics()},
        }
    finally:
        server.ready_queue.close()
        server.metrics_queue.close()
        server.error_queue.close()


def test_structured_camera_spec_rejects_implicit_type_coercion() -> None:
    value = {
        "camera_id": "camera_test",
        "serial_number": "serial-test",
        "width": True,
        "height": 240,
        "fps": 15,
        "preview_port": 6000,
        "jpeg_quality": 88,
    }

    with pytest.raises(argparse.ArgumentTypeError, match="width must be an integer"):
        parse_camera_spec(json.dumps(value))


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
        server.metrics_queue.close()
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
    assert "RealSense debug tool owns camera devices" in script
    assert "realsense-viewer|rs-capture|rs-color|rs-depth" in script
    assert "it will not be terminated automatically" in script
    assert 'server exited before ready' in script
    assert script.count('stop_direct_realsense_zmq.sh') >= 3
    assert "robot_bringup.service active without legacy camera owners; continuing" in script
    assert "refusing to start: robot_bringup.service" not in script
    assert "conda run --no-capture-output -n lerobot python" in script
    assert "all cameras ready" in script
    assert "JZ_DIRECT_CAMERA_HEAD_WHITE_BALANCE" in script
    assert "JZ_DIRECT_CAMERA_LEFT_WHITE_BALANCE" in script
    assert "JZ_DIRECT_CAMERA_RIGHT_WHITE_BALANCE" in script
    assert "AWB will still be forced off" in script


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
