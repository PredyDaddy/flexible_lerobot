#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ipaddress
import os
import signal
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(SRC_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from lerobot.robots.jz_robot_udp.protocol import (  # noqa: E402
    COMMAND_MODE_ARMED,
    COMMAND_MODE_DRY_RUN,
    COMMAND_MODES,
    ProtocolError,
    decode_jz_robot_udp_command_packet,
    validate_jz_robot_udp_command_packet,
)

DEFAULT_CONFIG = REPO_ROOT / "udp_test/test_scripts/arm_side/orin_phase3_executor_config.yaml"
DEFAULT_LEFT_JOINT_NAMES = [f"left_joint{i}" for i in range(1, 8)]
DEFAULT_RIGHT_JOINT_NAMES = [f"right_joint{i}" for i in range(1, 8)]
ALLOWED_COMMAND_TOPICS = {
    "left_arm_command_topic": "/robot1/telecon/arm_left/joint_commands_input",
    "right_arm_command_topic": "/robot1/telecon/arm_right/joint_commands_input",
    "left_gripper_command_topic": "/robot1/left_gripper/gripper_commands",
    "right_gripper_command_topic": "/robot1/right_gripper/gripper_commands",
}
ARMED_ENV_VAR = "JZ_UDP_EXECUTOR_ARMED"
MAX_REASONABLE_JOINT_ABS_LIMIT = 100.0
MAX_REASONABLE_GRIPPER_SPAN = 1000.0
MAX_REASONABLE_GRIPPER_DELTA_PER_STEP = 20.0

_SHUTDOWN_REQUESTED = False


@dataclass
class GripperLimits:
    width: tuple[Any, Any] | list[Any] | None = None
    force: tuple[Any, Any] | list[Any] | None = None
    max_width_delta_per_step: Any = None
    max_force_delta_per_step: Any = None


@dataclass
class Phase3ExecutorConfig:
    execution: str = COMMAND_MODE_DRY_RUN
    robot: str = "robot1"
    bind_ip: str = "192.168.1.81"
    allowed_sender_ip: str | None = "192.168.1.106"
    command_port: int = 39020
    buffer_size: int = 65535
    socket_timeout_s: float = 0.2
    max_command_age_s: float = 0.25
    max_clock_skew_s: float = 1.0
    command_timeout_s: float = 0.3
    max_publish_hz: float = 10.0
    qos_depth: int = 10
    armed_env_var: str = ARMED_ENV_VAR

    left_arm_command_topic: str = "/robot1/telecon/arm_left/joint_commands_input"
    right_arm_command_topic: str = "/robot1/telecon/arm_right/joint_commands_input"
    left_gripper_command_topic: str = "/robot1/left_gripper/gripper_commands"
    right_gripper_command_topic: str = "/robot1/right_gripper/gripper_commands"

    left_joint_names: list[str] = field(default_factory=lambda: DEFAULT_LEFT_JOINT_NAMES.copy())
    right_joint_names: list[str] = field(default_factory=lambda: DEFAULT_RIGHT_JOINT_NAMES.copy())
    joint_position_limits: dict[str, tuple[Any, Any] | list[Any]] = field(default_factory=dict)
    allow_joint_position_limit_bypass: bool = False
    allow_joint_delta_limit_bypass: bool = False
    max_joint_delta_per_step: Any = 0.02
    max_joint_delta_overrides: dict[str, Any] = field(default_factory=dict)
    initial_joint_positions: dict[str, Any] = field(default_factory=dict)
    gripper_limits: dict[str, GripperLimits] = field(default_factory=dict)
    initial_gripper_state: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def all_joint_names(self) -> list[str]:
        return [*self.left_joint_names, *self.right_joint_names]


@dataclass
class ValidatedCommand:
    seq: int
    mode: str
    robot: str
    stamp_ns: int
    left_positions: list[float]
    right_positions: list[float]
    left_gripper: tuple[float, float]
    right_gripper: tuple[float, float]


@dataclass
class RosCommandMessages:
    left_arm: Any
    right_arm: Any
    left_gripper: Any
    right_gripper: Any


@dataclass
class GateDecision:
    accepted: bool
    publish: bool
    reason: str
    seq: int | None = None
    command: ValidatedCommand | None = None


@dataclass
class ExecutorCounters:
    received: int = 0
    invalid: int = 0
    unexpected_sender: int = 0
    rejected_by_gate: int = 0
    published: int = 0
    dry_run: int = 0
    rate_limited: int = 0
    timeout: int = 0
    seq_gap_count: int = 0
    last_seq: int | None = None
    last_published_seq: int | None = None
    reject_reasons: dict[str, int] = field(default_factory=dict)


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return value == value and value not in (float("inf"), float("-inf"))


def _finite_float(value: Any) -> float | None:
    if not _is_finite_number(value):
        return None
    return float(value)


def _finite_range(values: Any) -> tuple[float, float] | None:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        return None
    lower = _finite_float(values[0])
    upper = _finite_float(values[1])
    if lower is None or upper is None or lower >= upper:
        return None
    return lower, upper


def _positive_finite(value: Any) -> float | None:
    numeric = _finite_float(value)
    if numeric is None or numeric <= 0:
        return None
    return numeric


def _coerce_gripper_limits(value: Any) -> GripperLimits:
    if isinstance(value, GripperLimits):
        return value
    if not isinstance(value, dict):
        return GripperLimits()
    return GripperLimits(
        width=value.get("width"),
        force=value.get("force"),
        max_width_delta_per_step=value.get("max_width_delta_per_step"),
        max_force_delta_per_step=value.get("max_force_delta_per_step"),
    )


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def load_config(path: str | Path = DEFAULT_CONFIG) -> Phase3ExecutorConfig:
    config_path = Path(path)
    if not config_path.exists():
        return Phase3ExecutorConfig()

    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to load Phase 3 executor YAML config") from exc

    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Phase 3 executor config must be a mapping: {config_path}")

    cfg = Phase3ExecutorConfig()
    network = _as_mapping(raw.get("network"))
    topics = _as_mapping(raw.get("topics"))
    timeouts = _as_mapping(raw.get("timeouts"))
    rate = _as_mapping(raw.get("rate_limits"))

    cfg.execution = raw.get("execution", raw.get("execution_mode", cfg.execution))
    cfg.robot = raw.get("robot", cfg.robot)
    cfg.bind_ip = raw.get("bind_ip", network.get("bind_ip", cfg.bind_ip))
    cfg.allowed_sender_ip = raw.get("allowed_sender_ip", network.get("allowed_sender_ip", cfg.allowed_sender_ip))
    cfg.command_port = raw.get("command_port", network.get("command_port", cfg.command_port))
    cfg.buffer_size = raw.get("buffer_size", network.get("buffer_size", cfg.buffer_size))
    cfg.socket_timeout_s = raw.get("socket_timeout_s", timeouts.get("socket_timeout_s", cfg.socket_timeout_s))
    cfg.max_command_age_s = raw.get("max_command_age_s", timeouts.get("max_command_age_s", cfg.max_command_age_s))
    cfg.max_clock_skew_s = raw.get("max_clock_skew_s", timeouts.get("max_clock_skew_s", cfg.max_clock_skew_s))
    cfg.command_timeout_s = raw.get("command_timeout_s", timeouts.get("command_timeout_s", cfg.command_timeout_s))
    cfg.max_publish_hz = raw.get("max_publish_hz", rate.get("max_publish_hz", cfg.max_publish_hz))
    cfg.qos_depth = raw.get("qos_depth", cfg.qos_depth)

    cfg.left_arm_command_topic = topics.get(
        "left_arm_command_topic", raw.get("left_arm_command_topic", cfg.left_arm_command_topic)
    )
    cfg.right_arm_command_topic = topics.get(
        "right_arm_command_topic", raw.get("right_arm_command_topic", cfg.right_arm_command_topic)
    )
    cfg.left_gripper_command_topic = topics.get(
        "left_gripper_command_topic", raw.get("left_gripper_command_topic", cfg.left_gripper_command_topic)
    )
    cfg.right_gripper_command_topic = topics.get(
        "right_gripper_command_topic", raw.get("right_gripper_command_topic", cfg.right_gripper_command_topic)
    )

    cfg.left_joint_names = list(raw.get("left_joint_names", cfg.left_joint_names))
    cfg.right_joint_names = list(raw.get("right_joint_names", cfg.right_joint_names))
    cfg.joint_position_limits = dict(raw.get("joint_position_limits", cfg.joint_position_limits))
    cfg.allow_joint_position_limit_bypass = bool(
        raw.get("allow_joint_position_limit_bypass", cfg.allow_joint_position_limit_bypass)
    )
    cfg.allow_joint_delta_limit_bypass = bool(
        raw.get("allow_joint_delta_limit_bypass", cfg.allow_joint_delta_limit_bypass)
    )
    cfg.initial_joint_positions = dict(raw.get("initial_joint_positions", cfg.initial_joint_positions))
    cfg.initial_gripper_state = dict(raw.get("initial_gripper_state", cfg.initial_gripper_state))

    joint_delta = raw.get("max_joint_delta_per_step", cfg.max_joint_delta_per_step)
    if isinstance(joint_delta, dict):
        cfg.max_joint_delta_per_step = joint_delta.get("default", cfg.max_joint_delta_per_step)
        cfg.max_joint_delta_overrides = dict(joint_delta.get("overrides", {}))
    else:
        cfg.max_joint_delta_per_step = joint_delta

    cfg.gripper_limits = {
        side: _coerce_gripper_limits(value) for side, value in _as_mapping(raw.get("gripper_limits")).items()
    }
    return cfg


def apply_args_to_config(cfg: Phase3ExecutorConfig, args: argparse.Namespace) -> Phase3ExecutorConfig:
    cfg.execution = args.execution
    if args.bind_ip is not None:
        cfg.bind_ip = args.bind_ip
    if args.port is not None:
        cfg.command_port = args.port
    if args.allowed_sender_ip is not None:
        cfg.allowed_sender_ip = args.allowed_sender_ip
    if args.buffer_size is not None:
        cfg.buffer_size = args.buffer_size
    if args.socket_timeout_s is not None:
        cfg.socket_timeout_s = args.socket_timeout_s
    return cfg


def _validate_common_config(cfg: Phase3ExecutorConfig) -> None:
    if cfg.execution not in COMMAND_MODES:
        raise RuntimeError(f"execution must be one of {COMMAND_MODES}")
    if not isinstance(cfg.command_port, int) or not 0 < cfg.command_port <= 65535:
        raise RuntimeError("command_port must be in 1..65535")
    if not isinstance(cfg.buffer_size, int) or cfg.buffer_size <= 0:
        raise RuntimeError("buffer_size must be positive")
    for name in ("socket_timeout_s", "max_command_age_s", "max_clock_skew_s", "command_timeout_s"):
        if _positive_finite(getattr(cfg, name)) is None:
            raise RuntimeError(f"{name} must be a positive finite number")
    if _positive_finite(cfg.max_publish_hz) is None:
        raise RuntimeError("max_publish_hz must be a positive finite number")
    if len(cfg.left_joint_names) != 7 or len(cfg.right_joint_names) != 7:
        raise RuntimeError("left_joint_names and right_joint_names must each contain 7 joints")
    if len(set(cfg.all_joint_names)) != len(cfg.all_joint_names):
        raise RuntimeError("joint names must be unique")


def _validate_armed_limits(cfg: Phase3ExecutorConfig) -> None:
    if not isinstance(cfg.allowed_sender_ip, str) or not cfg.allowed_sender_ip.strip():
        raise RuntimeError("allowed_sender_ip must be a non-empty IP address for armed execution")
    try:
        allowed_sender_ip = ipaddress.ip_address(cfg.allowed_sender_ip)
    except ValueError as exc:
        raise RuntimeError("allowed_sender_ip must be a valid IP address for armed execution") from exc
    if allowed_sender_ip.is_unspecified:
        raise RuntimeError("allowed_sender_ip must not be a wildcard address for armed execution")

    for field_name, allowed_topic in ALLOWED_COMMAND_TOPICS.items():
        configured_topic = getattr(cfg, field_name)
        if configured_topic != allowed_topic:
            raise RuntimeError(
                f"{field_name} must remain {allowed_topic!r} in Phase 3 first-version armed mode; "
                f"got {configured_topic!r}"
            )

    missing_limits = [joint for joint in cfg.all_joint_names if joint not in cfg.joint_position_limits]
    if missing_limits:
        raise RuntimeError(f"joint_position_limits missing joints: {missing_limits}")

    for joint in cfg.all_joint_names:
        limits = _finite_range(cfg.joint_position_limits[joint])
        if limits is None:
            raise RuntimeError(f"joint_position_limits invalid for {joint}")
        lower, upper = limits
        if abs(lower) > MAX_REASONABLE_JOINT_ABS_LIMIT or abs(upper) > MAX_REASONABLE_JOINT_ABS_LIMIT:
            raise RuntimeError(f"joint_position_limits too wide for {joint}")

    default_delta = _positive_finite(cfg.max_joint_delta_per_step)
    if default_delta is None or default_delta > 1.0:
        raise RuntimeError("max_joint_delta_per_step must be finite, positive, and conservative")
    for joint, value in cfg.max_joint_delta_overrides.items():
        delta = _positive_finite(value)
        if joint not in cfg.all_joint_names or delta is None or delta > 1.0:
            raise RuntimeError(f"max_joint_delta_per_step override invalid for {joint}")

    for side in ("left", "right"):
        limits = cfg.gripper_limits.get(side)
        if limits is None:
            raise RuntimeError(f"gripper_limits missing side: {side}")
        width = _finite_range(limits.width)
        force = _finite_range(limits.force)
        width_delta = _positive_finite(limits.max_width_delta_per_step)
        force_delta = _positive_finite(limits.max_force_delta_per_step)
        if width is None or force is None or width_delta is None or force_delta is None:
            raise RuntimeError(f"gripper_limits invalid for {side}")
        if width_delta > MAX_REASONABLE_GRIPPER_DELTA_PER_STEP:
            raise RuntimeError(f"gripper_limits width delta too wide for {side}")
        if force_delta > MAX_REASONABLE_GRIPPER_DELTA_PER_STEP:
            raise RuntimeError(f"gripper_limits force delta too wide for {side}")
        if width[1] - width[0] > MAX_REASONABLE_GRIPPER_SPAN:
            raise RuntimeError(f"gripper_limits width too wide for {side}")
        if force[1] - force[0] > MAX_REASONABLE_GRIPPER_SPAN:
            raise RuntimeError(f"gripper_limits force too wide for {side}")


def validate_startup_gates(cfg: Phase3ExecutorConfig, *, cli_ack: bool) -> bool:
    _validate_common_config(cfg)
    if cfg.execution == COMMAND_MODE_DRY_RUN:
        return False

    if os.environ.get(cfg.armed_env_var) != "1":
        raise RuntimeError(f"{cfg.armed_env_var}=1 is required for armed execution")
    if not cli_ack:
        raise RuntimeError("CLI acknowledgement is required for armed execution")
    _validate_armed_limits(cfg)
    return True


def startup_banner(cfg: Phase3ExecutorConfig, *, publish_enabled: bool) -> str:
    lines = []
    if publish_enabled:
        lines.extend(
            [
                "PHASE3 COMMAND EXECUTOR ARMED",
                "WILL publish ROS command topics",
                "operator must be next to emergency stop",
                "emergency stop is physical fallback, not a replacement for software limits",
                "required acknowledgements present",
            ]
        )
    else:
        lines.extend(
            [
                "PHASE3 COMMAND EXECUTOR DRY-RUN",
                "NOT publishing ROS command topics",
                "robot should not move",
            ]
        )
    lines.extend(
        [
            f"bind_ip={cfg.bind_ip}",
            f"port={cfg.command_port}",
            f"allowed_sender_ip={cfg.allowed_sender_ip}",
            f"left_arm_topic={cfg.left_arm_command_topic}",
            f"right_arm_topic={cfg.right_arm_command_topic}",
            f"left_gripper_topic={cfg.left_gripper_command_topic}",
            f"right_gripper_topic={cfg.right_gripper_command_topic}",
            f"max_publish_hz={cfg.max_publish_hz}",
            f"command_timeout_s={cfg.command_timeout_s}",
            f"allow_joint_position_limit_bypass={cfg.allow_joint_position_limit_bypass}",
            f"allow_joint_delta_limit_bypass={cfg.allow_joint_delta_limit_bypass}",
        ]
    )
    return "\n".join(lines)


def build_ros_command_messages(
    cfg: Phase3ExecutorConfig,
    command: ValidatedCommand,
    *,
    joint_state_cls: type,
    float64_multi_array_cls: type,
) -> RosCommandMessages:
    left_arm = joint_state_cls()
    left_arm.name = list(cfg.left_joint_names)
    left_arm.position = list(command.left_positions)

    right_arm = joint_state_cls()
    right_arm.name = list(cfg.right_joint_names)
    right_arm.position = list(command.right_positions)

    left_gripper = float64_multi_array_cls()
    left_gripper.data = [command.left_gripper[0], command.left_gripper[1]]

    right_gripper = float64_multi_array_cls()
    right_gripper.data = [command.right_gripper[0], command.right_gripper[1]]

    return RosCommandMessages(
        left_arm=left_arm,
        right_arm=right_arm,
        left_gripper=left_gripper,
        right_gripper=right_gripper,
    )


class RosPublisherAdapter:
    def __init__(self, cfg: Phase3ExecutorConfig) -> None:
        import rclpy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray

        self._rclpy = rclpy
        self._joint_state_cls = JointState
        self._float64_multi_array_cls = Float64MultiArray

        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("jz_udp_phase3_command_executor")
        self._left_arm_pub = self._node.create_publisher(JointState, cfg.left_arm_command_topic, cfg.qos_depth)
        self._right_arm_pub = self._node.create_publisher(JointState, cfg.right_arm_command_topic, cfg.qos_depth)
        self._left_gripper_pub = self._node.create_publisher(
            Float64MultiArray, cfg.left_gripper_command_topic, cfg.qos_depth
        )
        self._right_gripper_pub = self._node.create_publisher(
            Float64MultiArray, cfg.right_gripper_command_topic, cfg.qos_depth
        )
        self._cfg = cfg

    def publish_command(self, command: ValidatedCommand) -> None:
        messages = build_ros_command_messages(
            self._cfg,
            command,
            joint_state_cls=self._joint_state_cls,
            float64_multi_array_cls=self._float64_multi_array_cls,
        )
        self._left_arm_pub.publish(messages.left_arm)
        self._right_arm_pub.publish(messages.right_arm)
        self._left_gripper_pub.publish(messages.left_gripper)
        self._right_gripper_pub.publish(messages.right_gripper)

    def close(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None


class Phase3CommandExecutor:
    def __init__(self, cfg: Phase3ExecutorConfig, publisher_adapter: Any | None = None) -> None:
        self.cfg = cfg
        self.publisher_adapter = publisher_adapter
        self.counters = ExecutorCounters()
        self.active = False
        self.shutdown_requested = False
        self.last_joint_positions: dict[str, float] | None = None
        self.last_gripper_state: dict[str, dict[str, float]] | None = None
        self.last_publish_monotonic_s: float | None = None
        self.last_valid_command_monotonic_s: float | None = None

    def process_packet(
        self,
        packet: dict[str, Any],
        *,
        sender: tuple[str, int],
        now_ns: int | None = None,
        monotonic_s: float | None = None,
    ) -> GateDecision:
        self.counters.received += 1
        now_ns = time.time_ns() if now_ns is None else now_ns
        monotonic_s = time.monotonic() if monotonic_s is None else monotonic_s

        sender_ip, _sender_port = sender
        if self.cfg.allowed_sender_ip and sender_ip != self.cfg.allowed_sender_ip:
            self.counters.unexpected_sender += 1
            return self._reject("unexpected_sender", packet.get("seq"))

        try:
            validate_jz_robot_udp_command_packet(packet)
        except ProtocolError:
            self.counters.invalid += 1
            return self._reject("invalid_packet", packet.get("seq"))

        seq = int(packet["seq"])
        if packet["robot"] != self.cfg.robot:
            return self._reject("robot_mismatch", seq)

        age_s = (now_ns - int(packet["stamp_ns"])) / 1_000_000_000.0
        if age_s > float(self.cfg.max_command_age_s):
            return self._reject("stale_command", seq)
        if age_s < -float(self.cfg.max_clock_skew_s):
            return self._reject("future_command", seq)

        if self.counters.last_seq is not None and seq <= self.counters.last_seq:
            return self._reject("non_monotonic_seq", seq)

        actions = packet["actions"]
        command_or_reason = self._build_validated_command(packet)
        if isinstance(command_or_reason, str):
            return self._reject(command_or_reason, seq)
        command = command_or_reason

        joint_limit_reason = self._joint_limit_reason(command)
        gripper_limit_reason = self._gripper_limit_reason(actions)
        limit_reason = joint_limit_reason or gripper_limit_reason
        if limit_reason is not None:
            if self.cfg.execution == COMMAND_MODE_ARMED:
                return self._reject(limit_reason, seq)
            if self._dry_run_missing_limits_reason(limit_reason):
                self._remember_seq_for_dry_run(command)
                self.counters.dry_run += 1
                return GateDecision(
                    accepted=True,
                    publish=False,
                    reason="dry_run_limits_unconfigured",
                    seq=seq,
                    command=command,
                )
            return self._reject(limit_reason, seq)

        if self.cfg.execution == COMMAND_MODE_ARMED and command.mode != COMMAND_MODE_ARMED:
            return self._reject("packet_not_armed", seq)

        if self.cfg.execution == COMMAND_MODE_ARMED:
            period_s = 1.0 / float(self.cfg.max_publish_hz)
            if (
                self.last_publish_monotonic_s is not None
                and monotonic_s - self.last_publish_monotonic_s < period_s
            ):
                self.counters.rate_limited += 1
                return self._reject("rate_limited", seq)

        if self.cfg.execution != COMMAND_MODE_ARMED:
            self._remember_command(command, monotonic_s)
            self.counters.dry_run += 1
            reason = "executor_not_armed" if command.mode == COMMAND_MODE_ARMED else "dry_run"
            return GateDecision(accepted=True, publish=False, reason=reason, seq=seq, command=command)

        if self.publisher_adapter is None:
            return self._reject("publisher_unavailable", seq)

        self._remember_command(command, monotonic_s)
        self.publisher_adapter.publish_command(command)
        self.counters.published += 1
        self.counters.last_published_seq = seq
        self.last_publish_monotonic_s = monotonic_s
        return GateDecision(accepted=True, publish=True, reason="published", seq=seq, command=command)

    def _dry_run_missing_limits_reason(self, reason: str) -> bool:
        return reason in {
            "joint_position_limits",
            "max_joint_delta_unconfigured",
            "initial_joint_state_required",
            "gripper_limits",
            "initial_gripper_state_required",
        }

    def _build_validated_command(self, packet: dict[str, Any]) -> ValidatedCommand | str:
        actions = packet["actions"]
        left_actions = actions["left"]
        right_actions = actions["right"]
        if set(left_actions) != set(self.cfg.left_joint_names) or set(right_actions) != set(self.cfg.right_joint_names):
            return "joint_name_mismatch"

        grippers = actions["grippers"]
        try:
            left_positions = [float(left_actions[joint]) for joint in self.cfg.left_joint_names]
            right_positions = [float(right_actions[joint]) for joint in self.cfg.right_joint_names]
            left_gripper = (
                float(grippers["left"]["width"]),
                float(grippers["left"]["force"]),
            )
            right_gripper = (
                float(grippers["right"]["width"]),
                float(grippers["right"]["force"]),
            )
        except Exception:
            return "invalid_packet"

        return ValidatedCommand(
            seq=int(packet["seq"]),
            mode=str(packet["mode"]),
            robot=str(packet["robot"]),
            stamp_ns=int(packet["stamp_ns"]),
            left_positions=left_positions,
            right_positions=right_positions,
            left_gripper=left_gripper,
            right_gripper=right_gripper,
        )

    def _joint_limit_reason(self, command: ValidatedCommand) -> str | None:
        positions_by_joint = {
            **dict(zip(self.cfg.left_joint_names, command.left_positions, strict=True)),
            **dict(zip(self.cfg.right_joint_names, command.right_positions, strict=True)),
        }

        for joint, position in positions_by_joint.items():
            limits = _finite_range(self.cfg.joint_position_limits.get(joint))
            if limits is None:
                return "joint_position_limits"
            lower, upper = limits
            if not self.cfg.allow_joint_position_limit_bypass and (position < lower or position > upper):
                return "joint_position_limit"

        baseline = self.last_joint_positions
        if baseline is None:
            baseline = {
                joint: float(self.cfg.initial_joint_positions[joint])
                for joint in self.cfg.all_joint_names
                if _finite_float(self.cfg.initial_joint_positions.get(joint)) is not None
            }
            if set(baseline) != set(self.cfg.all_joint_names):
                return "initial_joint_state_required"

        for joint, position in positions_by_joint.items():
            max_delta = self._max_joint_delta(joint)
            if max_delta is None:
                return "max_joint_delta_unconfigured"
            if not self.cfg.allow_joint_delta_limit_bypass and abs(position - baseline[joint]) > max_delta:
                return "joint_delta_limit"
        return None

    def _max_joint_delta(self, joint: str) -> float | None:
        value = self.cfg.max_joint_delta_overrides.get(joint, self.cfg.max_joint_delta_per_step)
        return _positive_finite(value)

    def _gripper_limit_reason(self, actions: dict[str, Any]) -> str | None:
        requested = {
            side: {
                "width": float(actions["grippers"][side]["width"]),
                "force": float(actions["grippers"][side]["force"]),
            }
            for side in ("left", "right")
        }
        for side, fields in requested.items():
            limits = self.cfg.gripper_limits.get(side)
            if limits is None:
                return "gripper_limits"
            width_range = _finite_range(limits.width)
            force_range = _finite_range(limits.force)
            if width_range is None or force_range is None:
                return "gripper_limits"
            if fields["width"] < width_range[0] or fields["width"] > width_range[1]:
                return "gripper_limit"
            if fields["force"] < force_range[0] or fields["force"] > force_range[1]:
                return "gripper_limit"

        baseline = self.last_gripper_state
        if baseline is None:
            baseline = {}
            for side in ("left", "right"):
                side_state = _as_mapping(self.cfg.initial_gripper_state.get(side))
                width = _finite_float(side_state.get("width"))
                force = _finite_float(side_state.get("force"))
                if width is not None and force is not None:
                    baseline[side] = {"width": width, "force": force}
            if set(baseline) != {"left", "right"}:
                return "initial_gripper_state_required"

        for side, fields in requested.items():
            limits = self.cfg.gripper_limits[side]
            width_delta = _positive_finite(limits.max_width_delta_per_step)
            force_delta = _positive_finite(limits.max_force_delta_per_step)
            if width_delta is None or force_delta is None:
                return "gripper_limits"
            if abs(fields["width"] - baseline[side]["width"]) > width_delta:
                return "gripper_delta_limit"
            if abs(fields["force"] - baseline[side]["force"]) > force_delta:
                return "gripper_delta_limit"
        return None

    def _remember_command(self, command: ValidatedCommand, monotonic_s: float) -> None:
        self._remember_seq_for_dry_run(command)
        self.last_joint_positions = {
            **dict(zip(self.cfg.left_joint_names, command.left_positions, strict=True)),
            **dict(zip(self.cfg.right_joint_names, command.right_positions, strict=True)),
        }
        self.last_gripper_state = {
            "left": {"width": command.left_gripper[0], "force": command.left_gripper[1]},
            "right": {"width": command.right_gripper[0], "force": command.right_gripper[1]},
        }
        self.last_valid_command_monotonic_s = monotonic_s
        self.active = True

    def _remember_seq_for_dry_run(self, command: ValidatedCommand) -> None:
        if self.counters.last_seq is not None and command.seq > self.counters.last_seq + 1:
            self.counters.seq_gap_count += command.seq - self.counters.last_seq - 1
        self.counters.last_seq = command.seq

    def _reject(self, reason: str, seq: Any = None) -> GateDecision:
        self.counters.rejected_by_gate += 1
        self.counters.reject_reasons[reason] = self.counters.reject_reasons.get(reason, 0) + 1
        return GateDecision(
            accepted=False,
            publish=False,
            reason=reason,
            seq=seq if isinstance(seq, int) and not isinstance(seq, bool) else None,
        )

    def check_command_timeout(self, *, monotonic_s: float | None = None) -> bool:
        if not self.active or self.last_valid_command_monotonic_s is None:
            return False
        monotonic_s = time.monotonic() if monotonic_s is None else monotonic_s
        if monotonic_s - self.last_valid_command_monotonic_s <= float(self.cfg.command_timeout_s):
            return False
        self.active = False
        self.counters.timeout += 1
        return True

    def request_shutdown(self) -> None:
        self.shutdown_requested = True

    def close(self) -> None:
        if self.publisher_adapter is not None:
            self.publisher_adapter.close()

    def summary(self) -> str:
        return (
            "SUMMARY: "
            f"received={self.counters.received} invalid={self.counters.invalid} "
            f"unexpected_sender={self.counters.unexpected_sender} rejected_by_gate={self.counters.rejected_by_gate} "
            f"published={self.counters.published} dry_run={self.counters.dry_run} "
            f"rate_limited={self.counters.rate_limited} timeout={self.counters.timeout} "
            f"seq_gap_count={self.counters.seq_gap_count} last_seq={self.counters.last_seq} "
            f"last_published_seq={self.counters.last_published_seq} reject_reasons={self.counters.reject_reasons}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 Orin UDP command executor.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--bind-ip", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--allowed-sender-ip", default=None)
    parser.add_argument("--execution", choices=COMMAND_MODES, default=COMMAND_MODE_DRY_RUN)
    parser.add_argument("--count", type=int, default=0, help="Stop after N accepted packets. 0 means run forever.")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--socket-timeout-s", type=float, default=None)
    parser.add_argument("--i-understand-this-publishes-robot-commands", action="store_true")
    return parser.parse_args()


def request_shutdown(signum: int, _frame: object) -> None:
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = True
    print(f"[orin phase3 executor] signal={signum} shutdown requested", flush=True)


def install_signal_handlers() -> None:
    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)


def _print_decision(idx: int, decision: GateDecision) -> None:
    print(
        "[orin phase3 executor] "
        f"idx={idx} seq={decision.seq} accepted={decision.accepted} "
        f"publish={decision.publish} reason={decision.reason}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    if args.count < 0:
        raise ValueError("--count must be non-negative")
    if args.print_every < 0:
        raise ValueError("--print-every must be non-negative")

    cfg = apply_args_to_config(load_config(args.config), args)
    publish_enabled = validate_startup_gates(
        cfg,
        cli_ack=args.i_understand_this_publishes_robot_commands,
    )
    print(startup_banner(cfg, publish_enabled=publish_enabled), flush=True)

    install_signal_handlers()
    publisher = RosPublisherAdapter(cfg) if publish_enabled else None
    executor = Phase3CommandExecutor(cfg, publisher_adapter=publisher)
    accepted_count = 0

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((cfg.bind_ip, cfg.command_port))
            sock.settimeout(float(cfg.socket_timeout_s))
            while not _SHUTDOWN_REQUESTED and not executor.shutdown_requested:
                if args.count and accepted_count >= args.count:
                    break
                if executor.check_command_timeout():
                    print("[orin phase3 executor] COMMAND_TIMEOUT: stopped accepting stale activity", flush=True)
                try:
                    data, sender = sock.recvfrom(int(cfg.buffer_size))
                except socket.timeout:
                    continue

                try:
                    packet = decode_jz_robot_udp_command_packet(data)
                except ProtocolError as exc:
                    executor.counters.received += 1
                    executor.counters.invalid += 1
                    executor._reject("invalid_packet")
                    print(
                        f"[orin phase3 executor] WARN invalid packet bytes={len(data)} reason={exc}",
                        flush=True,
                    )
                    continue

                decision = executor.process_packet(packet, sender=sender)
                if decision.accepted:
                    accepted_count += 1
                if accepted_count == 1 or (args.print_every > 0 and accepted_count % args.print_every == 0):
                    _print_decision(accepted_count, decision)
    finally:
        executor.close()
        print(executor.summary(), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
