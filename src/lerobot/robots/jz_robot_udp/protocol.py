#!/usr/bin/env python

from __future__ import annotations

import json
import math
from typing import Any

PROTOCOL_VERSION = 1
STATE_MESSAGE_TYPE = "state"
COMMAND_MESSAGE_TYPE = "command"
TARGET_ACTION_MESSAGE_TYPE = "target_action"
COMMAND_MODE_DRY_RUN = "dry_run"
COMMAND_MODE_ARMED = "armed"
COMMAND_MODES = (COMMAND_MODE_DRY_RUN, COMMAND_MODE_ARMED)
COMMAND_ACTION_SIDES = ("left", "right")
COMMAND_GRIPPER_SIDES = ("left", "right")
COMMAND_GRIPPER_FIELDS = ("width", "force")
STATE_SOURCE_NAMES = ("left_joints", "right_joints", "left_gripper", "right_gripper")
SOURCE_TIMING_SCHEMA_VERSION = 1

_STATE_REQUIRED_KEYS = {"version", "type", "robot", "seq", "stamp_ns", "joints", "grippers"}
_STATE_OPTIONAL_KEYS = {"source_timing"}
_COMMAND_KEYS = {"version", "type", "robot", "seq", "stamp_ns", "mode", "actions"}
_TARGET_ACTION_KEYS = {"version", "type", "robot", "seq", "stamp_ns", "actions"}
_SIDES = {"left", "right"}
_GRIPPER_FIELDS = {"width", "force"}


class ProtocolError(ValueError):
    """Raised when a UDP packet does not match a JZRobot UDP schema."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ProtocolError(f"JSON object contains duplicate key: {key!r}")
        value[key] = item
    return value


def _reject_nonfinite_json_constant(value: str) -> Any:
    raise ProtocolError(f"non-standard JSON constant is forbidden: {value}")


def _decode_json_packet(data: bytes, packet_name: str) -> Any:
    if not isinstance(data, bytes):
        raise ProtocolError(f"{packet_name} packet data must be bytes")
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except ProtocolError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ProtocolError(f"failed to decode {packet_name} packet as strict JSON: {exc}") from exc


def _validate_exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        extra = sorted(set(value) - expected)
        missing = sorted(expected - set(value))
        raise ProtocolError(f"{name} keys mismatch: missing={missing}, extra={extra}")


def _validate_protocol_version(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != PROTOCOL_VERSION:
        raise ProtocolError(f"{name} version must be integer {PROTOCOL_VERSION}")


def encode_state_packet(packet: dict[str, Any]) -> bytes:
    validate_state_packet(packet)
    return json.dumps(packet, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_state_packet(data: bytes) -> dict[str, Any]:
    packet = _decode_json_packet(data, "state")
    validate_state_packet(packet)
    return packet


def make_jz_robot_udp_command_packet(
    *,
    robot: str,
    seq: int,
    stamp_ns: int,
    mode: str,
    actions: dict[str, Any],
) -> dict[str, Any]:
    packet = {
        "version": PROTOCOL_VERSION,
        "type": COMMAND_MESSAGE_TYPE,
        "robot": robot,
        "seq": seq,
        "stamp_ns": stamp_ns,
        "mode": mode,
        "actions": actions,
    }
    validate_jz_robot_udp_command_packet(packet)
    return packet


def make_jz_robot_udp_target_action_packet(
    *,
    robot: str,
    seq: int,
    stamp_ns: int,
    actions: dict[str, Any],
) -> dict[str, Any]:
    packet = {
        "version": PROTOCOL_VERSION,
        "type": TARGET_ACTION_MESSAGE_TYPE,
        "robot": robot,
        "seq": seq,
        "stamp_ns": stamp_ns,
        "actions": actions,
    }
    validate_target_action_packet(packet)
    return packet


def encode_jz_robot_udp_command_packet(packet: dict[str, Any]) -> bytes:
    validate_jz_robot_udp_command_packet(packet)
    return json.dumps(packet, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_jz_robot_udp_command_packet(data: bytes) -> dict[str, Any]:
    packet = _decode_json_packet(data, "command")
    validate_jz_robot_udp_command_packet(packet)
    return packet


def encode_target_action_packet(packet: dict[str, Any]) -> bytes:
    validate_target_action_packet(packet)
    return json.dumps(packet, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_target_action_packet(data: bytes) -> dict[str, Any]:
    packet = _decode_json_packet(data, "target action")
    validate_target_action_packet(packet)
    return packet


def validate_state_packet(packet: Any) -> None:
    if not isinstance(packet, dict):
        raise ProtocolError("state packet must be a JSON object")
    keys = set(packet)
    missing = sorted(_STATE_REQUIRED_KEYS - keys)
    extra = sorted(keys - (_STATE_REQUIRED_KEYS | _STATE_OPTIONAL_KEYS))
    if missing or extra:
        raise ProtocolError(f"state packet keys mismatch: missing={missing}, extra={extra}")
    _validate_protocol_version(packet.get("version"), "state packet")
    if packet.get("type") != STATE_MESSAGE_TYPE:
        raise ProtocolError(f"unsupported message type: {packet.get('type')}")
    if not isinstance(packet.get("robot"), str) or not packet["robot"]:
        raise ProtocolError("state packet robot must be a non-empty string")
    _validate_nonnegative_integer(packet.get("seq"), "state packet seq")
    _validate_nonnegative_integer(packet.get("stamp_ns"), "state packet stamp_ns")

    joints = packet.get("joints")
    if not isinstance(joints, dict):
        raise ProtocolError("state packet joints must be an object")
    _validate_exact_keys(joints, _SIDES, "state packet joints")
    for side in ("left", "right"):
        if not isinstance(joints.get(side), dict):
            raise ProtocolError(f"state packet joints.{side} must be an object")
        _validate_number_map(joints[side], f"joints.{side}")

    grippers = packet.get("grippers")
    if not isinstance(grippers, dict):
        raise ProtocolError("state packet grippers must be an object")
    _validate_exact_keys(grippers, _SIDES, "state packet grippers")
    for side in ("left", "right"):
        if not isinstance(grippers.get(side), dict):
            raise ProtocolError(f"state packet grippers.{side} must be an object")
        _validate_exact_keys(grippers[side], _GRIPPER_FIELDS, f"state packet grippers.{side}")
        _validate_number_map(grippers[side], f"grippers.{side}")

    if "source_timing" in packet:
        validate_source_timing(packet["source_timing"])


def validate_jz_robot_udp_command_packet(packet: Any) -> None:
    if not isinstance(packet, dict):
        raise ProtocolError("command packet must be a JSON object")
    _validate_exact_keys(packet, _COMMAND_KEYS, "command packet")
    _validate_protocol_version(packet.get("version"), "command packet")
    if packet.get("type") != COMMAND_MESSAGE_TYPE:
        raise ProtocolError(f"unsupported command message type: {packet.get('type')}")
    if not isinstance(packet.get("robot"), str) or not packet["robot"]:
        raise ProtocolError("command packet robot must be a non-empty string")
    _validate_nonnegative_integer(packet.get("seq"), "command packet seq")
    _validate_nonnegative_integer(packet.get("stamp_ns"), "command packet stamp_ns")
    if packet.get("mode") not in COMMAND_MODES:
        raise ProtocolError(f"command packet mode must be one of {COMMAND_MODES}")

    _validate_actions_object(packet.get("actions"), "command packet actions")


def validate_target_action_packet(packet: Any) -> None:
    if not isinstance(packet, dict):
        raise ProtocolError("target action packet must be a JSON object")
    _validate_exact_keys(packet, _TARGET_ACTION_KEYS, "target action packet")
    _validate_protocol_version(packet.get("version"), "target action packet")
    if packet.get("type") != TARGET_ACTION_MESSAGE_TYPE:
        raise ProtocolError(f"unsupported target action message type: {packet.get('type')}")
    if not isinstance(packet.get("robot"), str) or not packet["robot"]:
        raise ProtocolError("target action packet robot must be a non-empty string")
    _validate_nonnegative_integer(packet.get("seq"), "target action packet seq")
    _validate_nonnegative_integer(packet.get("stamp_ns"), "target action packet stamp_ns")

    _validate_actions_object(packet.get("actions"), "target action packet actions")


def _validate_actions_object(actions: Any, name: str) -> None:
    if not isinstance(actions, dict):
        raise ProtocolError(f"{name} must be an object")
    allowed_action_keys = {"left", "right", "grippers"}
    if set(actions) != allowed_action_keys:
        extra = sorted(set(actions) - allowed_action_keys)
        missing = sorted(allowed_action_keys - set(actions))
        raise ProtocolError(f"{name} keys mismatch: missing={missing}, extra={extra}")

    for side in COMMAND_ACTION_SIDES:
        if not isinstance(actions.get(side), dict):
            raise ProtocolError(f"{name}.{side} must be an object")
        _validate_number_map(actions[side], f"{name}.{side}")

    grippers = actions.get("grippers")
    if not isinstance(grippers, dict):
        raise ProtocolError(f"{name}.grippers must be an object")
    if set(grippers) != set(COMMAND_GRIPPER_SIDES):
        extra = sorted(set(grippers) - set(COMMAND_GRIPPER_SIDES))
        missing = sorted(set(COMMAND_GRIPPER_SIDES) - set(grippers))
        raise ProtocolError(f"{name}.grippers keys mismatch: missing={missing}, extra={extra}")
    for side in COMMAND_GRIPPER_SIDES:
        if not isinstance(grippers.get(side), dict):
            raise ProtocolError(f"{name}.grippers.{side} must be an object")
        fields = grippers[side]
        if set(fields) != set(COMMAND_GRIPPER_FIELDS):
            extra = sorted(set(fields) - set(COMMAND_GRIPPER_FIELDS))
            missing = sorted(set(COMMAND_GRIPPER_FIELDS) - set(fields))
            raise ProtocolError(f"{name}.grippers.{side} keys mismatch: missing={missing}, extra={extra}")
        _validate_number_map(fields, f"{name}.grippers.{side}")


def _validate_number_map(values: dict[str, Any], name: str) -> None:
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ProtocolError(f"{name} keys must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ProtocolError(f"{name}.{key} must be numeric")
        try:
            finite = math.isfinite(float(value))
        except OverflowError:
            finite = False
        if not finite:
            raise ProtocolError(f"{name}.{key} must be finite")


def validate_source_timing(source_timing: Any) -> None:
    if not isinstance(source_timing, dict):
        raise ProtocolError("state packet source_timing must be an object")
    expected_keys = {"schema_version", "source_skew_ms", "sources"}
    if set(source_timing) != expected_keys:
        extra = sorted(set(source_timing) - expected_keys)
        missing = sorted(expected_keys - set(source_timing))
        raise ProtocolError(f"source_timing keys mismatch: missing={missing}, extra={extra}")
    schema_version = source_timing.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != SOURCE_TIMING_SCHEMA_VERSION
    ):
        raise ProtocolError(f"source_timing.schema_version must be integer {SOURCE_TIMING_SCHEMA_VERSION}")
    _validate_nonnegative_number(source_timing.get("source_skew_ms"), "source_timing.source_skew_ms")

    sources = source_timing.get("sources")
    if not isinstance(sources, dict):
        raise ProtocolError("source_timing.sources must be an object")
    if set(sources) != set(STATE_SOURCE_NAMES):
        extra = sorted(set(sources) - set(STATE_SOURCE_NAMES))
        missing = sorted(set(STATE_SOURCE_NAMES) - set(sources))
        raise ProtocolError(f"source_timing.sources keys mismatch: missing={missing}, extra={extra}")

    expected_source_keys = {
        "generation",
        "recv_wall_ns",
        "recv_monotonic_ns",
        "header_stamp_ns",
        "age_ms",
    }
    for source_name in STATE_SOURCE_NAMES:
        source = sources[source_name]
        if not isinstance(source, dict):
            raise ProtocolError(f"source_timing.sources.{source_name} must be an object")
        if set(source) != expected_source_keys:
            extra = sorted(set(source) - expected_source_keys)
            missing = sorted(expected_source_keys - set(source))
            raise ProtocolError(
                f"source_timing.sources.{source_name} keys mismatch: missing={missing}, extra={extra}"
            )
        generation = source.get("generation")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
            raise ProtocolError(f"source_timing.sources.{source_name}.generation must be a positive integer")
        _validate_nonnegative_integer(
            source.get("recv_wall_ns"),
            f"source_timing.sources.{source_name}.recv_wall_ns",
        )
        _validate_nonnegative_integer(
            source.get("recv_monotonic_ns"),
            f"source_timing.sources.{source_name}.recv_monotonic_ns",
        )
        header_stamp_ns = source.get("header_stamp_ns")
        if source_name.endswith("_joints"):
            _validate_nonnegative_integer(
                header_stamp_ns,
                f"source_timing.sources.{source_name}.header_stamp_ns",
            )
        elif header_stamp_ns is not None:
            raise ProtocolError(f"source_timing.sources.{source_name}.header_stamp_ns must be null")
        _validate_nonnegative_number(
            source.get("age_ms"),
            f"source_timing.sources.{source_name}.age_ms",
        )


def _validate_nonnegative_integer(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProtocolError(f"{name} must be a non-negative integer")


def _validate_nonnegative_number(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ProtocolError(f"{name} must be numeric")
    try:
        finite = math.isfinite(float(value))
    except OverflowError:
        finite = False
    if not finite or value < 0:
        raise ProtocolError(f"{name} must be finite and non-negative")
