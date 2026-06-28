#!/usr/bin/env python

from __future__ import annotations

import json
import math
from typing import Any

PROTOCOL_VERSION = 1
STATE_MESSAGE_TYPE = "state"
COMMAND_MESSAGE_TYPE = "command"
COMMAND_MODE_DRY_RUN = "dry_run"
COMMAND_ACTION_SIDES = ("left", "right")
COMMAND_GRIPPER_SIDES = ("left", "right")
COMMAND_GRIPPER_FIELDS = ("width", "force")


class ProtocolError(ValueError):
    """Raised when a UDP packet does not match the readonly JZRobot state schema."""


def encode_state_packet(packet: dict[str, Any]) -> bytes:
    validate_state_packet(packet)
    return json.dumps(packet, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_state_packet(data: bytes) -> dict[str, Any]:
    try:
        packet = json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise ProtocolError(f"failed to decode state packet as JSON: {exc}") from exc
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


def encode_jz_robot_udp_command_packet(packet: dict[str, Any]) -> bytes:
    validate_jz_robot_udp_command_packet(packet)
    return json.dumps(packet, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_jz_robot_udp_command_packet(data: bytes) -> dict[str, Any]:
    try:
        packet = json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise ProtocolError(f"failed to decode command packet as JSON: {exc}") from exc
    validate_jz_robot_udp_command_packet(packet)
    return packet


def validate_state_packet(packet: Any) -> None:
    if not isinstance(packet, dict):
        raise ProtocolError("state packet must be a JSON object")
    if packet.get("version") != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version: {packet.get('version')}")
    if packet.get("type") != STATE_MESSAGE_TYPE:
        raise ProtocolError(f"unsupported message type: {packet.get('type')}")
    if not isinstance(packet.get("robot"), str) or not packet["robot"]:
        raise ProtocolError("state packet robot must be a non-empty string")
    if isinstance(packet.get("seq"), bool) or not isinstance(packet.get("seq"), int):
        raise ProtocolError("state packet seq must be an integer")
    if isinstance(packet.get("stamp_ns"), bool) or not isinstance(packet.get("stamp_ns"), int):
        raise ProtocolError("state packet stamp_ns must be an integer")

    joints = packet.get("joints")
    if not isinstance(joints, dict):
        raise ProtocolError("state packet joints must be an object")
    for side in ("left", "right"):
        if not isinstance(joints.get(side), dict):
            raise ProtocolError(f"state packet joints.{side} must be an object")
        _validate_number_map(joints[side], f"joints.{side}")

    grippers = packet.get("grippers", {})
    if grippers is None:
        raise ProtocolError("state packet grippers must be an object")
    if not isinstance(grippers, dict):
        raise ProtocolError("state packet grippers must be an object")
    for side in ("left", "right"):
        if not isinstance(grippers.get(side), dict):
            raise ProtocolError(f"state packet grippers.{side} must be an object")
        _validate_number_map(grippers[side], f"grippers.{side}")
        for field in ("width", "force"):
            if field not in grippers[side]:
                raise ProtocolError(f"state packet grippers.{side}.{field} is required")


def validate_jz_robot_udp_command_packet(packet: Any) -> None:
    if not isinstance(packet, dict):
        raise ProtocolError("command packet must be a JSON object")
    if set(packet) != {"version", "type", "robot", "seq", "stamp_ns", "mode", "actions"}:
        extra = sorted(set(packet) - {"version", "type", "robot", "seq", "stamp_ns", "mode", "actions"})
        missing = sorted({"version", "type", "robot", "seq", "stamp_ns", "mode", "actions"} - set(packet))
        raise ProtocolError(f"command packet keys mismatch: missing={missing}, extra={extra}")
    if packet.get("version") != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported command protocol version: {packet.get('version')}")
    if packet.get("type") != COMMAND_MESSAGE_TYPE:
        raise ProtocolError(f"unsupported command message type: {packet.get('type')}")
    if not isinstance(packet.get("robot"), str) or not packet["robot"]:
        raise ProtocolError("command packet robot must be a non-empty string")
    if isinstance(packet.get("seq"), bool) or not isinstance(packet.get("seq"), int):
        raise ProtocolError("command packet seq must be an integer")
    if isinstance(packet.get("stamp_ns"), bool) or not isinstance(packet.get("stamp_ns"), int):
        raise ProtocolError("command packet stamp_ns must be an integer")
    if packet.get("mode") != COMMAND_MODE_DRY_RUN:
        raise ProtocolError("command packet mode must be dry_run")

    actions = packet.get("actions")
    if not isinstance(actions, dict):
        raise ProtocolError("command packet actions must be an object")
    allowed_action_keys = {"left", "right", "grippers"}
    if set(actions) != allowed_action_keys:
        extra = sorted(set(actions) - allowed_action_keys)
        missing = sorted(allowed_action_keys - set(actions))
        raise ProtocolError(f"command packet actions keys mismatch: missing={missing}, extra={extra}")

    for side in COMMAND_ACTION_SIDES:
        if not isinstance(actions.get(side), dict):
            raise ProtocolError(f"command packet actions.{side} must be an object")
        _validate_number_map(actions[side], f"actions.{side}")

    grippers = actions.get("grippers")
    if not isinstance(grippers, dict):
        raise ProtocolError("command packet actions.grippers must be an object")
    if set(grippers) != set(COMMAND_GRIPPER_SIDES):
        extra = sorted(set(grippers) - set(COMMAND_GRIPPER_SIDES))
        missing = sorted(set(COMMAND_GRIPPER_SIDES) - set(grippers))
        raise ProtocolError(f"command packet actions.grippers keys mismatch: missing={missing}, extra={extra}")
    for side in COMMAND_GRIPPER_SIDES:
        if not isinstance(grippers.get(side), dict):
            raise ProtocolError(f"command packet actions.grippers.{side} must be an object")
        fields = grippers[side]
        if set(fields) != set(COMMAND_GRIPPER_FIELDS):
            extra = sorted(set(fields) - set(COMMAND_GRIPPER_FIELDS))
            missing = sorted(set(COMMAND_GRIPPER_FIELDS) - set(fields))
            raise ProtocolError(
                f"command packet actions.grippers.{side} keys mismatch: missing={missing}, extra={extra}"
            )
        _validate_number_map(fields, f"actions.grippers.{side}")


def _validate_number_map(values: dict[str, Any], name: str) -> None:
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ProtocolError(f"{name} keys must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ProtocolError(f"{name}.{key} must be numeric")
        if not math.isfinite(float(value)):
            raise ProtocolError(f"{name}.{key} must be finite")
