#!/usr/bin/env python

from __future__ import annotations

import json
import threading
import time
from fractions import Fraction

import numpy as np
import pytest

from lerobot.robots.config import RobotConfig
from lerobot.robots.jz_robot_pin import JZRobotPin
from lerobot.robots.jz_robot_pin_timed import (
    JZRobotPinTimed,
    JZRobotPinTimedConfig,
    TimestampedRTSPCamera,
)
from lerobot.robots.jz_robot_udp.config_jz_robot_udp import RTSPCameraConfig
from lerobot.robots.jz_robot_udp.protocol import (
    PROTOCOL_VERSION,
    STATE_MESSAGE_TYPE,
    make_jz_robot_udp_target_action_packet,
)
from lerobot.robots.jz_robot_udp.state_cache import CachedState, StateCache
from lerobot.robots.utils import make_robot_from_config
from lerobot.teleoperators.jz_robot_pin_target_action import (
    JZRobotPinTargetActionTeleop,
    JZRobotPinTargetActionTeleopConfig,
)


class DummyThread:
    def __init__(self) -> None:
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        del timeout
        self.alive = False


class FakeVideoFrame:
    def __init__(self, image: np.ndarray, *, pts: int | None = 3) -> None:
        self.image = image
        self.pts = pts
        self.time_base = Fraction(1, 30)

    def to_ndarray(self, *, format: str) -> np.ndarray:
        assert format == "rgb24"
        return self.image.copy()


class FakeContainer:
    def __init__(self, frames: list[FakeVideoFrame], *, block_after_frames: bool = False) -> None:
        self.frames = frames
        self.block_after_frames = block_after_frames
        self.closed = threading.Event()

    def decode(self, *, video: int):
        assert video == 0
        yield from self.frames
        while self.block_after_frames and not self.closed.wait(timeout=0.001):
            pass

    def close(self) -> None:
        self.closed.set()


def sample_state_packet(seq: int = 7) -> dict:
    return {
        "version": PROTOCOL_VERSION,
        "type": STATE_MESSAGE_TYPE,
        "robot": "robot1",
        "seq": seq,
        "stamp_ns": 123456789,
        "joints": {
            "left": {f"left_joint{i}": float(i) for i in range(1, 8)},
            "right": {f"right_joint{i}": float(i + 10) for i in range(1, 8)},
        },
        "grippers": {
            "left": {"width": 0.01, "force": 1.0},
            "right": {"width": 0.02, "force": 2.0},
        },
    }


def make_camera_config(**overrides) -> RTSPCameraConfig:
    values = {
        "url": "rtsp://127.0.0.1:8554/test",
        "width": 4,
        "height": 3,
        "warmup_frames": 0,
        "stale_frame_timeout_ms": 1000,
    }
    values.update(overrides)
    return RTSPCameraConfig(**values)


def make_robot_config(**overrides) -> JZRobotPinTimedConfig:
    values = {
        "id": "test_jz_robot_pin_timed",
        "rtsp_cameras": {},
        "use_gripper": True,
        "connect_timeout_s": 0.01,
        "state_timeout_s": 1.0,
    }
    values.update(overrides)
    return JZRobotPinTimedConfig(**values)


def test_jz_robot_pin_timed_is_registered_and_reuses_pin_behavior() -> None:
    config = make_robot_config()
    robot = make_robot_from_config(config)

    assert RobotConfig.get_choice_name(JZRobotPinTimedConfig) == "jz_robot_pin_timed"
    assert isinstance(robot, JZRobotPinTimed)
    assert isinstance(robot, JZRobotPin)
    assert robot.name == "jz_robot_pin_timed"
    assert len(robot.observation_features) == 18
    assert len(robot.action_features) == 18


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("camera_buffer_size", 0),
        ("camera_buffer_size", True),
        ("camera_reconnect_delay_ms", -1),
        ("timing_log_every_n", -1),
        ("max_camera_state_receive_skew_ms", 0.0),
        ("max_camera_state_receive_skew_ms", float("nan")),
        ("enforce_camera_state_receive_skew", "yes"),
        ("reject_reused_camera_frames", 1),
        ("timing_sidecar", None),
    ],
)
def test_jz_robot_pin_timed_rejects_invalid_timing_config(field: str, value) -> None:
    with pytest.raises((TypeError, ValueError)):
        make_robot_config(**{field: value})


def test_jz_robot_pin_timed_constructs_timestamped_camera() -> None:
    config = make_robot_config(rtsp_cameras={"camera_test": make_camera_config()})
    robot = JZRobotPinTimed(config)

    assert isinstance(robot.cameras["camera_test"], TimestampedRTSPCamera)
    assert robot.cameras["camera_test"].buffer_size == config.camera_buffer_size


def test_timestamped_rtsp_camera_keeps_pts_receive_time_and_detects_stale() -> None:
    monotonic_now = [2_000_000_000]
    camera = TimestampedRTSPCamera(
        make_camera_config(),
        wall_time_ns=lambda: 1_000_000_000,
        monotonic_ns=lambda: monotonic_now[0],
    )
    camera._thread = DummyThread()
    camera._stop_event.clear()
    image = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    camera._store_decoded_frame(FakeVideoFrame(image), generation=1)

    frame = camera.read_timed()

    assert np.array_equal(frame.image, image)
    assert frame.decoder_pts_ns == 100_000_000
    assert frame.receive_wall_ns == 1_000_000_000
    assert frame.receive_monotonic_ns == 2_000_000_000
    assert frame.decoder_sequence == 1
    assert camera.last_read_timing["timestamp_stage"] == "decoder_output_before_pixel_conversion"
    assert camera.last_read_timing["age_ms"] == 0.0

    monotonic_now[0] += 1_100_000_000
    with pytest.raises(TimeoutError, match="stale"):
        camera.read_timed()
    camera.disconnect()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("buffer_size", True),
        ("buffer_size", 1.5),
        ("reconnect_delay_ms", False),
        ("reconnect_delay_ms", 1.5),
    ],
)
def test_timestamped_rtsp_camera_rejects_non_integer_lifecycle_config(field: str, value) -> None:
    with pytest.raises(ValueError):
        TimestampedRTSPCamera(make_camera_config(), **{field: value})


def test_timestamped_rtsp_camera_selects_buffered_frame_nearest_target() -> None:
    monotonic_now = [1_000_000_000]
    camera = TimestampedRTSPCamera(
        make_camera_config(),
        monotonic_ns=lambda: monotonic_now[0],
    )
    camera._thread = DummyThread()
    camera._stop_event.clear()
    for sequence, receive_monotonic_ns in enumerate(
        (1_000_000_000, 1_100_000_000, 1_200_000_000),
        start=1,
    ):
        monotonic_now[0] = receive_monotonic_ns
        image = np.full((3, 4, 3), sequence, dtype=np.uint8)
        camera._store_decoded_frame(FakeVideoFrame(image), generation=1)

    selected = camera.read_timed_nearest(1_125_000_000)

    assert selected.decoder_sequence == 2
    assert np.all(selected.image == 2)
    assert camera.last_read_timing["decoder_sequence"] == 2
    camera.disconnect()


def test_timestamped_rtsp_camera_connect_cleans_up_after_warmup_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    container = FakeContainer([FakeVideoFrame(image)], block_after_frames=True)
    camera = TimestampedRTSPCamera(
        make_camera_config(timeout_ms=20, warmup_frames=1),
        reconnect_delay_ms=0,
    )
    monkeypatch.setattr(camera, "_open_container", lambda: container)

    with pytest.raises(TimeoutError, match="warming up"):
        camera.connect()

    assert container.closed.is_set()
    assert camera._thread is None
    assert not camera.is_connected


def test_timestamped_rtsp_camera_reconnects_and_tracks_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = FakeContainer([FakeVideoFrame(np.full((3, 4, 3), 1, dtype=np.uint8))])
    second = FakeContainer(
        [FakeVideoFrame(np.full((3, 4, 3), 2, dtype=np.uint8))],
        block_after_frames=True,
    )
    containers = iter((first, second))
    camera = TimestampedRTSPCamera(make_camera_config(), reconnect_delay_ms=0)
    monkeypatch.setattr(camera, "_open_container", lambda: next(containers))

    camera.connect()
    try:
        deadline = time.monotonic() + 1.0
        while camera.diagnostics["reconnect_count"] < 1 and time.monotonic() < deadline:
            time.sleep(0.001)

        frame = camera.read_timed()

        assert frame.reconnect_generation == 2
        assert np.all(frame.image == 2)
        assert camera.diagnostics["reconnect_count"] == 1
        assert first.closed.is_set()
    finally:
        camera.disconnect()

    assert second.closed.is_set()
    assert camera._thread is None


def test_timestamped_rtsp_camera_parses_opencv_ffmpeg_options() -> None:
    assert TimestampedRTSPCamera._parse_capture_options("rtsp_transport;tcp|buffer_size;1024") == {
        "rtsp_transport": "tcp",
        "buffer_size": "1024",
    }
    with pytest.raises(ValueError, match="key;value"):
        TimestampedRTSPCamera._parse_capture_options("invalid")


def test_jz_robot_pin_timed_aligns_camera_to_state_receive_time() -> None:
    camera_config = make_camera_config()
    robot = JZRobotPinTimed(
        make_robot_config(
            rtsp_cameras={"camera_test": camera_config},
            max_camera_state_receive_skew_ms=20,
            timing_log_every_n=0,
        )
    )
    camera = robot.cameras["camera_test"]
    camera._thread = DummyThread()
    camera._stop_event.clear()
    robot._state_cache.update(sample_state_packet(), sender=("192.168.1.81", 39010))
    state = robot._state_cache.latest()
    state_receive_monotonic_ns = int(state.received_monotonic_s * 1_000_000_000)

    camera._monotonic_ns = lambda: state_receive_monotonic_ns - 10_000_000
    camera._store_decoded_frame(
        FakeVideoFrame(np.full((3, 4, 3), 1, dtype=np.uint8)),
        generation=1,
    )
    camera._monotonic_ns = lambda: state_receive_monotonic_ns + 50_000_000
    camera._store_decoded_frame(
        FakeVideoFrame(np.full((3, 4, 3), 2, dtype=np.uint8)),
        generation=1,
    )
    camera._monotonic_ns = lambda: state_receive_monotonic_ns + 60_000_000
    robot._is_connected = True

    try:
        observation = robot.get_observation()
        timing = robot.last_observation_timing
    finally:
        robot.disconnect()

    assert np.all(observation["camera_test"] == 1)
    assert timing["cameras"]["camera_test"]["decoder_sequence"] == 1
    assert timing["cameras"]["camera_test"]["state_receive_delta_ms"] == -10.0
    assert timing["cameras"]["camera_test"]["state_receive_skew_ms"] == 10.0


def test_rejected_timing_does_not_mark_camera_frame_as_accepted() -> None:
    camera_config = make_camera_config()
    robot = JZRobotPinTimed(
        make_robot_config(
            rtsp_cameras={"camera_test": camera_config},
            max_camera_state_receive_skew_ms=5,
            enforce_camera_state_receive_skew=True,
            reject_reused_camera_frames=True,
            timing_log_every_n=0,
        )
    )
    camera = robot.cameras["camera_test"]
    camera._thread = DummyThread()
    camera._stop_event.clear()
    now = [1_000_000_000]
    camera._monotonic_ns = lambda: now[0]
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    camera._store_decoded_frame(FakeVideoFrame(image), generation=1)
    camera.read_timed()
    robot._after_observation(
        CachedState(sample_state_packet(seq=1), ("192.168.1.81", 39010), 1.0),
        {},
    )

    now[0] = 2_000_000_000
    camera._store_decoded_frame(FakeVideoFrame(image), generation=1)
    camera.read_timed()
    with pytest.raises(TimeoutError, match="receive skew"):
        robot._after_observation(
            CachedState(sample_state_packet(seq=2), ("192.168.1.81", 39010), 1.0),
            {},
        )

    robot._after_observation(
        CachedState(sample_state_packet(seq=3), ("192.168.1.81", 39010), 2.0),
        {},
    )

    assert robot.last_observation_timing["observation_sequence"] == 2
    assert not robot.last_observation_timing["cameras"]["camera_test"]["reused_by_observation_loop"]
    camera.disconnect()


def test_jz_robot_pin_timed_observation_and_sidecar_keep_18d_schema(tmp_path) -> None:
    camera_config = make_camera_config()
    robot = JZRobotPinTimed(
        make_robot_config(
            rtsp_cameras={"camera_test": camera_config},
            max_camera_state_receive_skew_ms=1000,
            timing_log_every_n=0,
        )
    )
    camera = robot.cameras["camera_test"]
    camera._thread = DummyThread()
    camera._stop_event.clear()
    camera._store_decoded_frame(
        FakeVideoFrame(np.zeros((camera_config.height, camera_config.width, 3), dtype=np.uint8)),
        generation=1,
    )
    robot._state_cache.update(sample_state_packet(), sender=("192.168.1.81", 39010))
    robot._is_connected = True

    observation = robot.get_observation()
    numeric_observation = {key: value for key, value in observation.items() if key != "camera_test"}
    robot.send_action(numeric_observation)
    robot.save_frame_timing(
        dataset_root=tmp_path,
        episode_index=2,
        frame_index=4,
        action_timing={"packet_seq": 11},
    )
    robot._close_timing_files()

    assert len(numeric_observation) == 18
    assert observation["camera_test"].shape == (3, 4, 3)
    assert robot.last_observation_timing["state"]["packet_seq"] == 7
    assert isinstance(robot.last_observation_timing["state"]["receive_wall_ns"], int)
    assert robot.last_observation_timing["cameras"]["camera_test"]["decoder_sequence"] == 1

    sidecar = tmp_path / "meta/timing/episode-000002.jsonl"
    saved = json.loads(sidecar.read_text(encoding="utf-8"))
    assert len(saved["session_id"]) == 32
    assert saved["episode_index"] == 2
    assert saved["frame_index"] == 4
    assert saved["action"]["packet_seq"] == 11
    assert saved["command"]["observation_sequence"] == 1
    assert saved["command"]["packet_seq"] == 1
    assert saved["command"]["mode"] == "dry_run"
    assert saved["command"]["transport"] == "local"
    assert isinstance(saved["command"]["send_completed_wall_ns"], int)
    assert isinstance(saved["command"]["send_completed_monotonic_ns"], int)
    assert saved["command"]["action_key_count"] == 18
    assert saved["cameras"]["camera_test"]["decoder_sequence"] == 1

    robot._is_connected = False
    camera.disconnect()


def test_timing_sidecar_restarts_episode_without_duplicate_frames(tmp_path) -> None:
    robot = JZRobotPinTimed(make_robot_config(timing_log_every_n=0))
    robot._last_observation_timing = {
        "session_id": robot._timing_session_id,
        "observation_sequence": 1,
        "state": {},
        "cameras": {},
    }
    robot._last_command_timing = {"observation_sequence": 1, "packet_seq": 1}

    robot.save_frame_timing(
        dataset_root=tmp_path,
        episode_index=0,
        frame_index=0,
        action_timing={"packet_seq": 1},
    )
    robot.save_frame_timing(
        dataset_root=tmp_path,
        episode_index=0,
        frame_index=1,
        action_timing={"packet_seq": 2},
    )
    robot.save_frame_timing(
        dataset_root=tmp_path,
        episode_index=0,
        frame_index=0,
        action_timing={"packet_seq": 3},
    )
    robot._close_timing_files()

    lines = (tmp_path / "meta/timing/episode-000000.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["action"]["packet_seq"] == 3


def test_timing_sidecar_requires_command_from_matching_observation(tmp_path) -> None:
    robot = JZRobotPinTimed(make_robot_config(timing_log_every_n=0))
    robot._last_observation_timing = {
        "session_id": robot._timing_session_id,
        "observation_sequence": 2,
        "state": {},
        "cameras": {},
    }

    with pytest.raises(RuntimeError, match="before sending"):
        robot.save_frame_timing(
            dataset_root=tmp_path,
            episode_index=0,
            frame_index=0,
            action_timing={"packet_seq": 1},
        )

    robot._last_command_timing = {"observation_sequence": 1, "packet_seq": 1}
    with pytest.raises(RuntimeError, match="another observation"):
        robot.save_frame_timing(
            dataset_root=tmp_path,
            episode_index=0,
            frame_index=0,
            action_timing={"packet_seq": 1},
        )

    assert not (tmp_path / "meta/timing/episode-000000.jsonl").exists()


@pytest.mark.parametrize("action_timing", [None, []])
def test_timing_sidecar_requires_target_action_timing(tmp_path, action_timing) -> None:
    robot = JZRobotPinTimed(make_robot_config(timing_log_every_n=0))
    robot._last_observation_timing = {
        "session_id": robot._timing_session_id,
        "observation_sequence": 1,
        "state": {},
        "cameras": {},
    }
    robot._last_command_timing = {"observation_sequence": 1, "packet_seq": 1}

    with pytest.raises(RuntimeError, match="target-action timing"):
        robot.save_frame_timing(
            dataset_root=tmp_path,
            episode_index=0,
            frame_index=0,
            action_timing=action_timing,
        )

    assert not (tmp_path / "meta/timing/episode-000000.jsonl").exists()


def test_state_and_action_timing_include_backward_compatible_receive_wall_time() -> None:
    legacy_state = CachedState(sample_state_packet(), ("127.0.0.1", 1), 1.0)
    assert legacy_state.received_wall_ns is None

    cache = StateCache()
    cache.update(sample_state_packet(), sender=("127.0.0.1", 1))
    assert isinstance(cache.latest().received_wall_ns, int)

    config = JZRobotPinTargetActionTeleopConfig(target_action_port=39030)
    teleop = JZRobotPinTargetActionTeleop(config)
    teleop._is_connected = True
    packet = sample_state_packet()
    target_packet = make_jz_robot_udp_target_action_packet(
        robot=packet["robot"],
        seq=packet["seq"],
        stamp_ns=time.time_ns(),
        actions={
            "left": packet["joints"]["left"],
            "right": packet["joints"]["right"],
            "grippers": packet["grippers"],
        },
    )
    teleop._target_action_cache.update(target_packet, sender=("127.0.0.1", 39030))
    cached_receive_wall_ns = teleop._target_action_cache.latest().received_wall_ns

    teleop.get_action()

    assert isinstance(cached_receive_wall_ns, int)
    assert teleop.last_action_timing["receive_wall_ns"] == cached_receive_wall_ns
