#!/usr/bin/env python

from __future__ import annotations

import ast
import json
import math
import os
import socket
import sys
import threading
import time
import types
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

import pytest

from lerobot.robots.jz_robot_udp import JZRobotUDP, JZRobotUDPConfig
from lerobot.robots.jz_robot_udp.config_jz_robot_udp import RTSPCameraConfig
from lerobot.robots.jz_robot_udp.protocol import (
    COMMAND_MESSAGE_TYPE,
    COMMAND_MODE_ARMED,
    COMMAND_MODE_DRY_RUN,
    PROTOCOL_VERSION,
    ProtocolError,
    STATE_MESSAGE_TYPE,
    TARGET_ACTION_MESSAGE_TYPE,
    decode_jz_robot_udp_command_packet,
    decode_state_packet,
    decode_target_action_packet,
    encode_jz_robot_udp_command_packet,
    encode_state_packet,
    encode_target_action_packet,
    make_jz_robot_udp_command_packet,
    make_jz_robot_udp_target_action_packet,
    validate_source_timing,
)
from lerobot.robots.jz_robot_udp.rtsp_camera import RTSPCamera, configure_opencv_rtsp_environment
from lerobot.robots.jz_robot_udp.state_cache import StateCache
from lerobot.robots.jz_robot_udp.udp_client import UDPStateReceiver, UDPTargetActionReceiver
from lerobot.scripts.lerobot_record import _get_teleop_action
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.utils import make_teleoperator_from_config
from udp_test.test_scripts.x86_side.x86_jz_robot_udp_replay_action_check import action_vector_to_dict
from udp_test.test_scripts.x86_side.x86_jz_robot_udp_send_action_check import (
    make_observation_delta_action,
    validate_args,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ORIN_COMMAND_RECEIVER = REPO_ROOT / "udp_test/test_scripts/arm_side/orin_udp_command_receiver.py"
SOURCE_TIMING_SCHEMA = REPO_ROOT / "my_devs/jz_robot_pin_timed/schema/source_timing_v1.schema.json"
SOURCE_TIMING_EXAMPLE = REPO_ROOT / "my_devs/jz_robot_pin_timed/schema/source_timing_v1.example.json"


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


def sample_source_timing(generation: int = 12) -> dict:
    source_names = ("left_joints", "right_joints", "left_gripper", "right_gripper")
    receive_offsets_ms = (0, 2, 5, 9)
    sources = {}
    for index, (source_name, offset_ms) in enumerate(zip(source_names, receive_offsets_ms, strict=True)):
        sources[source_name] = {
            "generation": generation + index,
            "recv_wall_ns": 1_783_737_600_000_000_000 + offset_ms * 1_000_000,
            "recv_monotonic_ns": 123_456_789_000_000 + offset_ms * 1_000_000,
            "header_stamp_ns": (
                1_783_737_599_999_000_000 + offset_ms * 1_000_000 if source_name.endswith("joints") else None
            ),
            "age_ms": float(9 - offset_ms),
        }
    return {
        "schema_version": 1,
        "source_skew_ms": 9.0,
        "sources": sources,
    }


def sample_action() -> dict[str, float]:
    action = {
        **{f"left_left_joint{i}.pos": float(i) for i in range(1, 8)},
        **{f"right_right_joint{i}.pos": float(i + 10) for i in range(1, 8)},
        "left_gripper.width": 0.01,
        "left_gripper.force": 1.0,
        "right_gripper.width": 0.02,
        "right_gripper.force": 2.0,
    }
    return action


def sample_command_actions() -> dict:
    return {
        "left": {f"left_joint{i}": float(i) for i in range(1, 8)},
        "right": {f"right_joint{i}": float(i + 10) for i in range(1, 8)},
        "grippers": {
            "left": {"width": 0.01, "force": 1.0},
            "right": {"width": 0.02, "force": 2.0},
        },
    }


def sample_target_action_packet(seq: int = 8) -> dict:
    return make_jz_robot_udp_target_action_packet(
        robot="robot1",
        seq=seq,
        stamp_ns=987654321,
        actions=sample_command_actions(),
    )


def sample_command_packet(seq: int = 1) -> dict:
    return make_jz_robot_udp_command_packet(
        robot="robot1",
        seq=seq,
        stamp_ns=123,
        mode=COMMAND_MODE_DRY_RUN,
        actions=sample_command_actions(),
    )


STRICT_DECODER_CASES = (
    ("state", decode_state_packet, sample_state_packet),
    ("command", decode_jz_robot_udp_command_packet, sample_command_packet),
    ("target_action", decode_target_action_packet, sample_target_action_packet),
)


def make_config(**overrides) -> JZRobotUDPConfig:
    values = {
        "id": "test_jz_robot_udp",
        "rtsp_cameras": {},
        "use_gripper": True,
        "state_timeout_s": 1.0,
        "connect_timeout_s": 0.01,
    }
    values.update(overrides)
    return JZRobotUDPConfig(
        **values,
    )


def test_jz_robot_udp_command_config_defaults_are_safe() -> None:
    cfg = JZRobotUDPConfig()

    assert cfg.command_target_ip == "192.168.1.81"
    assert cfg.command_target_port == 39020
    assert cfg.send_action_transport == "local"
    assert cfg.send_action_execution == "dry_run"
    assert cfg.command_robot == "robot1"
    assert cfg.command_timeout_s == 0.2


def test_rtsp_camera_configures_tcp_capture_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
    cfg = RTSPCameraConfig(url="rtsp://192.168.1.81:8554/robot_camera/camera_head", transport="tcp")

    configure_opencv_rtsp_environment(cfg)

    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "rtsp_transport;tcp"


def test_rtsp_camera_configures_custom_capture_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
    cfg = RTSPCameraConfig(
        url="rtsp://192.168.1.81:8554/robot_camera/camera_head",
        transport="tcp",
        ffmpeg_capture_options="rtsp_transport;tcp|stimeout;5000000",
    )

    configure_opencv_rtsp_environment(cfg)

    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "rtsp_transport;tcp|stimeout;5000000"


def test_rtsp_camera_threaded_reader_drains_to_latest_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    class FakeVideoCapture:
        def __init__(self, _url: str):
            self.read_count = 0
            self.released = False
            self.options = []

        def isOpened(self) -> bool:  # noqa: N802 - mirrors the OpenCV API
            return not self.released

        def set(self, prop, value) -> None:
            self.options.append((prop, value))

        def read(self):
            time.sleep(0.001)
            self.read_count += 1
            frame = np.full((4, 6, 3), self.read_count % 255, dtype=np.uint8)
            return True, frame

        def release(self) -> None:
            self.released = True

    fake_cv2 = types.SimpleNamespace(
        VideoCapture=FakeVideoCapture,
        CAP_PROP_BUFFERSIZE=1,
        CAP_PROP_OPEN_TIMEOUT_MSEC=2,
        CAP_PROP_READ_TIMEOUT_MSEC=3,
        COLOR_BGR2RGB=4,
        cvtColor=lambda frame, _code: frame[:, :, ::-1],
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    camera = RTSPCamera(
        RTSPCameraConfig(
            url="rtsp://192.168.1.81:8554/robot_camera/camera_head",
            width=6,
            height=4,
            warmup_frames=2,
            threaded_reader=True,
            stale_frame_timeout_ms=1000,
            read_retry_sleep_ms=1,
        )
    )

    camera.connect()
    try:
        first_read_count = camera.diagnostics["frames_read"]
        time.sleep(0.01)
        frame = camera.async_read()
        later_read_count = camera.diagnostics["frames_read"]
    finally:
        camera.disconnect()

    assert frame.shape == (4, 6, 3)
    assert first_read_count >= 3
    assert later_read_count > first_read_count
    assert camera.diagnostics["read_failures"] == 0


def test_jz_robot_udp_command_config_accepts_explicit_armed_execution() -> None:
    cfg = JZRobotUDPConfig(send_action_execution=COMMAND_MODE_ARMED)

    assert cfg.send_action_execution == COMMAND_MODE_ARMED


@pytest.mark.parametrize("execution", ["active", "execute", "publish", ""])
def test_jz_robot_udp_command_config_rejects_unknown_execution(execution: str) -> None:
    with pytest.raises(ValueError, match="send_action_execution"):
        JZRobotUDPConfig(send_action_execution=execution)


@pytest.mark.parametrize("transport", ["active", "armed", "execute", "publish", ""])
def test_jz_robot_udp_command_config_rejects_unknown_transport(transport: str) -> None:
    with pytest.raises(ValueError, match="send_action_transport"):
        JZRobotUDPConfig(send_action_transport=transport)


def test_state_packet_round_trip_validates_schema() -> None:
    encoded = encode_state_packet(sample_state_packet())
    decoded = decode_state_packet(encoded)

    assert decoded["version"] == 1
    assert decoded["type"] == "state"
    assert decoded["seq"] == 7
    assert decoded["joints"]["left"]["left_joint1"] == 1.0
    assert decoded["grippers"]["right"]["force"] == 2.0
    assert "source_timing" not in decoded


@pytest.mark.parametrize(("_name", "decoder", "packet_factory"), STRICT_DECODER_CASES)
@pytest.mark.parametrize("version", [True, False, 1.0])
def test_external_packet_decoders_require_integer_protocol_version(
    _name: str, decoder, packet_factory, version
) -> None:
    packet = packet_factory()
    packet["version"] = version

    with pytest.raises(ProtocolError, match="version must be integer 1"):
        decoder(json.dumps(packet).encode("utf-8"))


@pytest.mark.parametrize(("_name", "decoder", "packet_factory"), STRICT_DECODER_CASES)
@pytest.mark.parametrize("field", ["seq", "stamp_ns"])
def test_external_packet_decoders_reject_bool_integer_fields(
    _name: str, decoder, packet_factory, field: str
) -> None:
    packet = packet_factory()
    packet[field] = True

    with pytest.raises(ProtocolError, match=field):
        decoder(json.dumps(packet).encode("utf-8"))


@pytest.mark.parametrize(("_name", "decoder", "packet_factory"), STRICT_DECODER_CASES)
@pytest.mark.parametrize("field", ["seq", "stamp_ns"])
def test_external_packet_decoders_reject_negative_sequence_and_time_fields(
    _name: str, decoder, packet_factory, field: str
) -> None:
    packet = packet_factory()
    packet[field] = -1

    with pytest.raises(ProtocolError, match=f"{field} must be a non-negative integer"):
        decoder(json.dumps(packet).encode("utf-8"))


@pytest.mark.parametrize(("_name", "decoder", "packet_factory"), STRICT_DECODER_CASES)
@pytest.mark.parametrize("constant", [math.nan, math.inf, -math.inf])
def test_external_packet_decoders_reject_nonstandard_nonfinite_json_constants(
    _name: str, decoder, packet_factory, constant: float
) -> None:
    packet = packet_factory()
    packet["seq"] = constant

    with pytest.raises(ProtocolError, match="non-standard JSON constant is forbidden"):
        decoder(json.dumps(packet).encode("utf-8"))


@pytest.mark.parametrize(("_name", "decoder", "packet_factory"), STRICT_DECODER_CASES)
def test_external_packet_decoders_reject_duplicate_root_keys(
    _name: str, decoder, packet_factory
) -> None:
    payload = json.dumps(packet_factory(), separators=(",", ":"))
    payload = payload.replace('"version":1', '"version":1,"version":1', 1)

    with pytest.raises(ProtocolError, match="duplicate key: 'version'"):
        decoder(payload.encode("utf-8"))


@pytest.mark.parametrize(("_name", "decoder", "packet_factory"), STRICT_DECODER_CASES)
def test_external_packet_decoders_reject_duplicate_nested_keys(
    _name: str, decoder, packet_factory
) -> None:
    payload = json.dumps(packet_factory(), separators=(",", ":"))
    payload = payload.replace('"width":0.01', '"width":0.01,"width":0.02', 1)

    with pytest.raises(ProtocolError, match="duplicate key: 'width'"):
        decoder(payload.encode("utf-8"))


@pytest.mark.parametrize(("_name", "decoder", "packet_factory"), STRICT_DECODER_CASES)
def test_external_packet_decoders_reject_extra_top_level_fields(
    _name: str, decoder, packet_factory
) -> None:
    packet = packet_factory()
    packet["unexpected"] = 1

    with pytest.raises(ProtocolError, match="extra=.*unexpected"):
        decoder(json.dumps(packet).encode("utf-8"))


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda packet: packet["joints"].__setitem__("middle", {}), "state packet joints.*extra=.*middle"),
        (
            lambda packet: packet["grippers"].__setitem__(
                "middle", {"width": 0.0, "force": 0.0}
            ),
            "state packet grippers.*extra=.*middle",
        ),
        (
            lambda packet: packet["grippers"]["left"].__setitem__("temperature", 20.0),
            "state packet grippers.left.*extra=.*temperature",
        ),
    ],
)
def test_external_state_decoder_rejects_extra_structural_fields(mutation, match: str) -> None:
    packet = sample_state_packet()
    mutation(packet)

    with pytest.raises(ProtocolError, match=match):
        decode_state_packet(json.dumps(packet).encode("utf-8"))


@pytest.mark.parametrize(
    ("decoder", "packet_factory"),
    [
        (decode_jz_robot_udp_command_packet, sample_command_packet),
        (decode_target_action_packet, sample_target_action_packet),
    ],
)
@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda packet: packet["actions"].__setitem__("middle", {}), "extra=.*middle"),
        (
            lambda packet: packet["actions"]["grippers"]["left"].__setitem__("velocity", 0.1),
            "extra=.*velocity",
        ),
    ],
)
def test_external_action_decoders_reject_extra_structural_fields(
    decoder, packet_factory, mutation, match: str
) -> None:
    packet = packet_factory()
    mutation(packet)

    with pytest.raises(ProtocolError, match=match):
        decoder(json.dumps(packet).encode("utf-8"))


class _OneShotPacketSocket:
    def __init__(self, receiver: UDPStateReceiver, payload: bytes):
        self.receiver = receiver
        self.payload = payload

    def recvfrom(self, _buffer_size: int) -> tuple[bytes, tuple[str, int]]:
        self.receiver._stop_event.set()
        return self.payload, ("127.0.0.1", 39010)


@pytest.mark.parametrize("target_action", [False, True])
def test_udp_receivers_do_not_cache_packets_rejected_by_strict_decoder(target_action: bool) -> None:
    cache = StateCache()
    receiver = (
        UDPTargetActionReceiver("127.0.0.1", 0, cache)
        if target_action
        else UDPStateReceiver("127.0.0.1", 0, cache)
    )
    packet = sample_target_action_packet() if target_action else sample_state_packet()
    payload = json.dumps(packet, separators=(",", ":"))
    payload = payload.replace('"version":1', '"version":1,"version":1', 1).encode("utf-8")
    receiver._socket = _OneShotPacketSocket(receiver, payload)

    receiver._run()

    assert cache.latest() is None


def test_state_packet_source_timing_round_trip_is_additive() -> None:
    packet = sample_state_packet()
    packet["source_timing"] = sample_source_timing()

    decoded = decode_state_packet(encode_state_packet(packet))

    assert decoded["version"] == PROTOCOL_VERSION
    assert decoded["joints"] == packet["joints"]
    assert decoded["grippers"] == packet["grippers"]
    assert decoded["source_timing"] == packet["source_timing"]
    assert decoded["source_timing"]["sources"]["left_joints"]["header_stamp_ns"] is not None
    assert decoded["source_timing"]["sources"]["left_gripper"]["header_stamp_ns"] is None


def test_source_timing_example_matches_schema_root_and_public_validator() -> None:
    schema = json.loads(SOURCE_TIMING_SCHEMA.read_text())
    example = json.loads(SOURCE_TIMING_EXAMPLE.read_text())

    assert set(example) == {"schema_version", "source_skew_ms", "sources"}
    assert schema["properties"]["schema_version"] == {"type": "integer", "const": 1}
    validate_source_timing(example)


def test_state_packet_calls_source_timing_validator() -> None:
    packet = sample_state_packet()
    packet["source_timing"] = sample_source_timing()
    packet["source_timing"]["schema_version"] = True

    with pytest.raises(ValueError, match="schema_version must be integer 1"):
        encode_state_packet(packet)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda timing: timing.pop("sources"), "missing=.*sources"),
        (lambda timing: timing.__setitem__("unexpected", 1), "extra=.*unexpected"),
        (lambda timing: timing["sources"].pop("right_gripper"), "missing=.*right_gripper"),
        (lambda timing: timing["sources"].__setitem__("other", {}), "extra=.*other"),
        (lambda timing: timing["sources"]["left_joints"].pop("age_ms"), "missing=.*age_ms"),
        (
            lambda timing: timing["sources"]["left_joints"].__setitem__("unexpected", 1),
            "extra=.*unexpected",
        ),
    ],
)
def test_source_timing_rejects_missing_and_extra_fields(mutation, match: str) -> None:
    source_timing = sample_source_timing()
    mutation(source_timing)

    with pytest.raises(ValueError, match=match):
        validate_source_timing(source_timing)


@pytest.mark.parametrize("schema_version", [True, False, 1.0, 0, 2])
def test_source_timing_requires_integer_schema_version_one(schema_version) -> None:
    source_timing = sample_source_timing()
    source_timing["schema_version"] = schema_version

    with pytest.raises(ValueError, match="schema_version must be integer 1"):
        validate_source_timing(source_timing)


@pytest.mark.parametrize("value", [True, -0.1, math.nan, math.inf])
def test_source_timing_rejects_invalid_source_skew(value) -> None:
    source_timing = sample_source_timing()
    source_timing["source_skew_ms"] = value

    with pytest.raises(ValueError, match="source_skew_ms"):
        validate_source_timing(source_timing)


@pytest.mark.parametrize(
    ("source_name", "field", "value"),
    [
        ("left_joints", "generation", True),
        ("left_joints", "generation", 0),
        ("right_joints", "recv_wall_ns", False),
        ("right_joints", "recv_wall_ns", -1),
        ("left_gripper", "recv_monotonic_ns", True),
        ("left_gripper", "recv_monotonic_ns", -1),
        ("right_gripper", "age_ms", False),
        ("right_gripper", "age_ms", -0.1),
        ("right_gripper", "age_ms", math.nan),
    ],
)
def test_source_timing_rejects_invalid_source_values(source_name: str, field: str, value) -> None:
    source_timing = sample_source_timing()
    source_timing["sources"][source_name][field] = value

    with pytest.raises(ValueError, match=field):
        validate_source_timing(source_timing)


@pytest.mark.parametrize(
    ("source_name", "header_stamp_ns"),
    [
        ("left_joints", None),
        ("left_joints", True),
        ("right_joints", -1),
        ("left_gripper", 0),
        ("right_gripper", False),
    ],
)
def test_source_timing_rejects_header_type_for_source_kind(source_name: str, header_stamp_ns) -> None:
    source_timing = sample_source_timing()
    source_timing["sources"][source_name]["header_stamp_ns"] = header_stamp_ns

    with pytest.raises(ValueError, match="header_stamp_ns"):
        validate_source_timing(source_timing)


def test_typical_state_packet_with_source_timing_fits_single_ipv4_udp_payload() -> None:
    packet = sample_state_packet(seq=1195)
    packet["stamp_ns"] = 1_783_737_600_123_456_789
    packet["joints"] = {
        "left": {f"left_joint{i}": -2.123456789012345 + i * 0.123456789012345 for i in range(1, 8)},
        "right": {f"right_joint{i}": 2.123456789012345 - i * 0.123456789012345 for i in range(1, 8)},
    }
    packet["grippers"] = {
        "left": {"width": 50.1234567890123, "force": 70.1234567890123},
        "right": {"width": 49.9876543210987, "force": 69.9876543210987},
    }
    packet["source_timing"] = sample_source_timing(generation=1195)

    encoded = encode_state_packet(packet)

    assert len(encoded) <= 1472, f"state packet would require IPv4 fragmentation: {len(encoded)} bytes"
    assert decode_state_packet(encoded)["source_timing"] == packet["source_timing"]


@pytest.mark.parametrize("mode", [COMMAND_MODE_DRY_RUN, COMMAND_MODE_ARMED])
def test_command_packet_accepts_dry_run_and_armed_modes(mode: str) -> None:
    packet = make_jz_robot_udp_command_packet(
        robot="robot1",
        seq=1,
        stamp_ns=123,
        mode=mode,
        actions=sample_command_actions(),
    )

    decoded = decode_jz_robot_udp_command_packet(encode_jz_robot_udp_command_packet(packet))

    assert decoded["version"] == PROTOCOL_VERSION
    assert decoded["type"] == COMMAND_MESSAGE_TYPE
    assert decoded["robot"] == "robot1"
    assert decoded["seq"] == 1
    assert decoded["stamp_ns"] == 123
    assert decoded["mode"] == mode
    assert decoded["actions"]["left"]["left_joint1"] == 1.0
    assert decoded["actions"]["grippers"]["right"]["force"] == 2.0


def test_target_action_packet_round_trip_validates_schema() -> None:
    packet = sample_target_action_packet()

    decoded = decode_target_action_packet(encode_target_action_packet(packet))

    assert decoded["version"] == PROTOCOL_VERSION
    assert decoded["type"] == TARGET_ACTION_MESSAGE_TYPE
    assert decoded["robot"] == "robot1"
    assert decoded["seq"] == 8
    assert decoded["stamp_ns"] == 987654321
    assert decoded["actions"]["left"]["left_joint1"] == 1.0
    assert decoded["actions"]["right"]["right_joint7"] == 17.0
    assert decoded["actions"]["grippers"]["left"]["width"] == 0.01


@pytest.mark.parametrize("mode", ["active", "execute", "publish", ""])
def test_command_packet_rejects_unknown_modes(mode: str) -> None:
    with pytest.raises(Exception, match="mode"):
        make_jz_robot_udp_command_packet(
            robot="robot1",
            seq=1,
            stamp_ns=123,
            mode=mode,
            actions=sample_command_actions(),
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda packet: packet.__setitem__("type", "state"), "message type"),
        (lambda packet: packet.pop("actions"), "actions"),
        (lambda packet: packet.pop("stamp_ns"), "stamp_ns"),
        (lambda packet: packet.__setitem__("seq", True), "seq"),
        (lambda packet: packet.__setitem__("stamp_ns", False), "stamp_ns"),
        (lambda packet: packet.__setitem__("mode", "execute"), "mode"),
        (lambda packet: packet["actions"]["grippers"]["left"].__setitem__("velocity", 0.1), "velocity"),
        (lambda packet: packet["actions"]["left"].__setitem__("left_joint1", math.nan), "finite"),
        (lambda packet: packet["actions"]["right"].__setitem__("right_joint1", math.inf), "finite"),
        (lambda packet: packet["actions"]["left"].__setitem__("left_joint2", True), "numeric"),
        (lambda packet: packet["actions"]["grippers"]["right"].__setitem__("force", False), "numeric"),
    ],
)
def test_command_packet_rejects_invalid_cases(mutation, match: str) -> None:
    packet = make_jz_robot_udp_command_packet(
        robot="robot1",
        seq=1,
        stamp_ns=123,
        mode="dry_run",
        actions=sample_command_actions(),
    )
    mutation(packet)

    with pytest.raises(Exception, match=match):
        decode_jz_robot_udp_command_packet(encode_jz_robot_udp_command_packet(packet))


def test_state_cache_waits_for_latest_state() -> None:
    cache = StateCache()
    assert cache.latest() is None

    cache.update(sample_state_packet(seq=1), sender=("192.168.1.81", 39010))

    latest = cache.wait(timeout_s=0.01)
    assert latest is not None
    assert latest.packet["seq"] == 1
    assert latest.sender == ("192.168.1.81", 39010)


def test_state_cache_waits_for_strictly_newer_local_revision_across_packet_seq_reset() -> None:
    cache = StateCache()
    cache.update(sample_state_packet(seq=99), sender=("192.168.1.81", 39010))
    first = cache.latest()
    assert first is not None

    timer = threading.Timer(
        0.01,
        lambda: cache.update(sample_state_packet(seq=1), sender=("192.168.1.81", 39010)),
    )
    timer.start()
    try:
        second = cache.wait_after_revision(timeout_s=0.2, after_revision=first.revision)
    finally:
        timer.join(timeout=1.0)

    assert second is not None
    assert second.revision == first.revision + 1
    assert second.packet["seq"] == 1


def test_udp_state_receiver_fails_fast_when_port_is_already_owned() -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    first = UDPStateReceiver("127.0.0.1", port, StateCache())
    second = UDPStateReceiver("127.0.0.1", port, StateCache())

    first.start()
    try:
        with pytest.raises(OSError, match="another recorder, probe, or control process"):
            second.start()
        assert second._socket is None
        assert first.is_running
    finally:
        first.stop()


def test_jz_robot_udp_observation_features_match_jz_robot_style() -> None:
    robot = JZRobotUDP(make_config())

    assert robot.observation_features["left_left_joint1.pos"] is float
    assert robot.observation_features["right_right_joint7.pos"] is float
    assert robot.observation_features["left_gripper.width"] is float
    assert robot.observation_features["right_gripper.force"] is float
    assert robot.action_features["left_left_joint1.pos"] is float
    assert robot.action_features["right_right_joint7.pos"] is float
    assert robot.action_features["left_gripper.width"] is float
    assert robot.action_features["right_gripper.force"] is float


def test_jz_robot_udp_get_observation_from_cached_state() -> None:
    robot = JZRobotUDP(make_config())
    robot._is_connected = True
    robot._state_cache.update(sample_state_packet(), sender=("192.168.1.81", 39010))

    obs = robot.get_observation()

    assert obs["left_left_joint1.pos"] == 1.0
    assert obs["right_right_joint7.pos"] == 17.0
    assert obs["left_gripper.width"] == 0.01
    assert obs["right_gripper.force"] == 2.0


def test_jz_robot_udp_local_dry_run_send_action_returns_float_copy_without_udp_sender() -> None:
    robot = JZRobotUDP(make_config(send_action_transport="local", send_action_execution="dry_run"))
    robot._is_connected = True

    returned = robot.send_action(sample_action())

    assert returned == sample_action()
    assert all(isinstance(value, float) for value in returned.values())
    assert robot._command_seq == 1
    assert robot._command_sender is None


def test_jz_robot_udp_send_action_rejects_missing_or_extra_keys() -> None:
    robot = JZRobotUDP(make_config(send_action_transport="local", send_action_execution="dry_run"))
    robot._is_connected = True
    missing_action = sample_action()
    del missing_action["left_left_joint1.pos"]
    extra_action = {**sample_action(), "left_left_joint99.pos": 99.0}

    with pytest.raises(ValueError, match="missing"):
        robot.send_action(missing_action)
    with pytest.raises(ValueError, match="unexpected"):
        robot.send_action(extra_action)


def test_jz_robot_udp_send_action_rejects_bool_values() -> None:
    robot = JZRobotUDP(make_config(send_action_transport="local", send_action_execution="dry_run"))
    robot._is_connected = True
    action = sample_action()
    action["left_left_joint1.pos"] = True

    with pytest.raises(ValueError, match="numeric"):
        robot.send_action(action)


def test_jz_robot_udp_udp_dry_run_send_action_sends_command_packet() -> None:
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(1.0)
    _, port = receiver.getsockname()
    robot = JZRobotUDP(
        make_config(
            command_target_ip="127.0.0.1",
            command_target_port=port,
            send_action_transport="udp",
            send_action_execution="dry_run",
        )
    )
    robot._is_connected = True

    try:
        returned = robot.send_action(sample_action())
        data, sender = receiver.recvfrom(65535)
    finally:
        robot.disconnect()
        receiver.close()

    decoded = decode_jz_robot_udp_command_packet(data)
    assert returned == sample_action()
    assert sender[0] == "127.0.0.1"
    assert decoded["type"] == "command"
    assert decoded["mode"] == "dry_run"
    assert decoded["robot"] == "robot1"
    assert decoded["seq"] == 1
    assert decoded["actions"]["left"]["left_joint1"] == 1.0
    assert decoded["actions"]["right"]["right_joint7"] == 17.0
    assert decoded["actions"]["grippers"]["left"]["width"] == 0.01


def test_jz_robot_udp_udp_armed_send_action_sends_armed_command_packet_only() -> None:
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(1.0)
    _, port = receiver.getsockname()
    robot = JZRobotUDP(
        make_config(
            command_target_ip="127.0.0.1",
            command_target_port=port,
            send_action_transport="udp",
            send_action_execution=COMMAND_MODE_ARMED,
        )
    )
    robot._is_connected = True

    try:
        robot.send_action(sample_action())
        data, _sender = receiver.recvfrom(65535)
    finally:
        robot.disconnect()
        receiver.close()

    decoded = decode_jz_robot_udp_command_packet(data)
    assert decoded["mode"] == COMMAND_MODE_ARMED
    assert robot._command_seq == 1


def test_x86_send_action_check_builds_observation_delta_action() -> None:
    robot = JZRobotUDP(make_config())
    observation = {key: float(index) for index, key in enumerate(robot.action_features)}

    action = make_observation_delta_action(
        robot.action_features,
        observation,
        delta_key="left_left_joint1.pos",
        delta_value=0.001,
    )

    assert action["left_left_joint1.pos"] == observation["left_left_joint1.pos"] + 0.001
    unchanged_keys = set(robot.action_features) - {"left_left_joint1.pos"}
    assert all(action[key] == observation[key] for key in unchanged_keys)


def test_x86_send_action_check_observation_delta_requires_real_observation_connection() -> None:
    args = Namespace(
        count=1,
        hz=1.0,
        command_target_port=39020,
        action_source="observation_delta",
        command_only=True,
        delta_key="left_left_joint1.pos",
    )

    with pytest.raises(ValueError, match="--no-command-only"):
        validate_args(args)


def test_orin_command_receiver_has_no_ros_publish_path() -> None:
    tree = ast.parse(ORIN_COMMAND_RECEIVER.read_text())
    forbidden_import_roots = {"rclpy"}
    forbidden_import_parts = {"cmd_vel"}
    forbidden_attributes = {"create_publisher", "publish"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", maxsplit=1)[0]
                parts = set(alias.name.lower().split("."))
                assert root not in forbidden_import_roots
                assert not (parts & forbidden_import_parts)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".", maxsplit=1)[0]
            parts = set(module.lower().split("."))
            assert root not in forbidden_import_roots
            assert not (parts & forbidden_import_parts)
        elif isinstance(node, ast.Attribute):
            assert node.attr not in forbidden_attributes


def test_jz_robot_udp_stale_state_fails() -> None:
    robot = JZRobotUDP(make_config(state_timeout_s=0.01))
    robot._is_connected = True
    robot._state_cache.update(sample_state_packet(), sender=("192.168.1.81", 39010))
    time.sleep(0.02)

    with pytest.raises(TimeoutError, match="stale"):
        robot.get_observation()


def test_state_packet_requires_gripper_fields_when_grippers_are_present() -> None:
    packet = sample_state_packet()
    del packet["grippers"]["left"]["force"]

    with pytest.raises(Exception, match="force"):
        decode_state_packet(encode_state_packet(packet))


def test_connect_waits_for_fresh_state_after_receiver_start() -> None:
    robot = JZRobotUDP(make_config(connect_timeout_s=0.01))
    robot._state_cache.update(sample_state_packet(), sender=("192.168.1.81", 39010))
    robot._receiver = Mock()
    robot._receiver.start.side_effect = lambda: None
    robot._receiver.stop.side_effect = lambda: None

    with pytest.raises(TimeoutError, match="first JZRobot UDP state packet"):
        robot.connect()


def test_jz_robot_udp_sender_filter_rejects_unexpected_sender() -> None:
    robot = JZRobotUDP(make_config(allowed_sender_ip="192.168.1.81"))
    robot._is_connected = True
    robot._state_cache.update(sample_state_packet(), sender=("192.168.1.200", 39010))

    with pytest.raises(RuntimeError, match="unexpected sender"):
        robot.get_observation()


def test_jz_robot_udp_constant_teleop_matches_robot_action_features() -> None:
    from lerobot.teleoperators.jz_robot_udp_constant import JZRobotUDPConstantTeleopConfig

    robot = JZRobotUDP(make_config())
    teleop = make_teleoperator_from_config(JZRobotUDPConstantTeleopConfig())

    assert teleop.action_features == robot.action_features
    assert teleop.feedback_features == {}
    assert teleop.is_calibrated
    assert not teleop.is_connected

    teleop.connect()
    try:
        action = teleop.get_action()
    finally:
        teleop.disconnect()

    assert action == dict.fromkeys(robot.action_features, 0.0)
    assert not teleop.is_connected


def test_jz_robot_udp_constant_teleop_is_registered_for_draccus() -> None:
    from lerobot.teleoperators.jz_robot_udp_constant import JZRobotUDPConstantTeleopConfig

    assert TeleoperatorConfig.get_choice_name(JZRobotUDPConstantTeleopConfig) == "jz_robot_udp_constant"


def test_jz_robot_udp_send_action_accepts_numpy_scalar_values() -> None:
    import numpy as np

    robot = JZRobotUDP(make_config(send_action_transport="local"))
    robot._is_connected = True
    action = {key: np.float32(index) for index, key in enumerate(robot.action_features)}

    returned = robot.send_action(action)

    assert returned == {key: float(index) for index, key in enumerate(robot.action_features)}


def test_jz_robot_udp_send_action_accepts_single_value_tensors() -> None:
    torch = pytest.importorskip("torch")

    robot = JZRobotUDP(make_config(send_action_transport="local"))
    robot._is_connected = True
    action = {key: torch.tensor([float(index)]) for index, key in enumerate(robot.action_features)}

    returned = robot.send_action(action)

    assert returned == {key: float(index) for index, key in enumerate(robot.action_features)}


def test_jz_robot_udp_send_action_rejects_non_scalar_tensors() -> None:
    torch = pytest.importorskip("torch")

    robot = JZRobotUDP(make_config(send_action_transport="local"))
    robot._is_connected = True
    action = sample_action()
    action["left_left_joint1.pos"] = torch.tensor([1.0, 2.0])

    with pytest.raises(ValueError, match="scalar numeric"):
        robot.send_action(action)


def test_replay_action_check_maps_action_vector_to_robot_action() -> None:
    import numpy as np

    names = ["left_left_joint1.pos", "left_left_joint2.pos"]
    action = action_vector_to_dict(np.array([1.25, 2.5], dtype=np.float32), names)

    assert action == {"left_left_joint1.pos": pytest.approx(1.25), "left_left_joint2.pos": pytest.approx(2.5)}


def test_jz_robot_udp_hold_teleop_maps_observation_to_action() -> None:
    from lerobot.teleoperators.jz_robot_udp_hold import JZRobotUDPHoldTeleopConfig

    robot = JZRobotUDP(make_config())
    teleop = make_teleoperator_from_config(JZRobotUDPHoldTeleopConfig())
    observation = {key: float(index) for index, key in enumerate(robot.action_features)}

    action = teleop.get_action_from_observation(observation)

    assert teleop.action_features == robot.action_features
    assert action == observation


def test_jz_robot_udp_hold_teleop_rejects_missing_observation_keys() -> None:
    from lerobot.teleoperators.jz_robot_udp_hold import JZRobotUDPHoldTeleopConfig

    teleop = make_teleoperator_from_config(JZRobotUDPHoldTeleopConfig())

    with pytest.raises(RuntimeError, match="missing"):
        teleop.get_action_from_observation({"left_left_joint1.pos": 0.0})


def test_record_action_getter_uses_observation_aware_teleop() -> None:
    from lerobot.teleoperators.jz_robot_udp_hold import JZRobotUDPHoldTeleopConfig

    teleop = make_teleoperator_from_config(JZRobotUDPHoldTeleopConfig())
    observation = {key: float(index) for index, key in enumerate(teleop.action_features)}

    assert _get_teleop_action(teleop, observation) == observation
