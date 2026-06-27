#!/usr/bin/env python

from __future__ import annotations

import json
from typing import Any

PROTOCOL_VERSION = 1
STATE_MESSAGE_TYPE = "state"


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


def validate_state_packet(packet: Any) -> None:
    if not isinstance(packet, dict):
        raise ProtocolError("state packet must be a JSON object")
    if packet.get("version") != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version: {packet.get('version')}")
    if packet.get("type") != STATE_MESSAGE_TYPE:
        raise ProtocolError(f"unsupported message type: {packet.get('type')}")
    if not isinstance(packet.get("robot"), str) or not packet["robot"]:
        raise ProtocolError("state packet robot must be a non-empty string")
    if not isinstance(packet.get("seq"), int):
        raise ProtocolError("state packet seq must be an integer")
    if not isinstance(packet.get("stamp_ns"), int):
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


def _validate_number_map(values: dict[str, Any], name: str) -> None:
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ProtocolError(f"{name} keys must be non-empty strings")
        if not isinstance(value, int | float):
            raise ProtocolError(f"{name}.{key} must be numeric")
