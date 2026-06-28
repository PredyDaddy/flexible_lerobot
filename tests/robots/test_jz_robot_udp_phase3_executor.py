#!/usr/bin/env python

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from lerobot.robots.jz_robot_udp.protocol import COMMAND_MODE_ARMED, COMMAND_MODE_DRY_RUN

from udp_test.test_scripts.arm_side import orin_phase3_command_executor as phase3

REPO_ROOT = Path(__file__).resolve().parents[2]
EXECUTOR_PATH = REPO_ROOT / "udp_test/test_scripts/arm_side/orin_phase3_command_executor.py"
EXECUTOR_CONFIG_PATH = REPO_ROOT / "udp_test/test_scripts/arm_side/orin_phase3_executor_config.yaml"
PHASE2_RECEIVER_PATH = REPO_ROOT / "udp_test/test_scripts/arm_side/orin_udp_command_receiver.py"
X86_ROBOT_PATH = REPO_ROOT / "src/lerobot/robots/jz_robot_udp/jz_robot_udp.py"
START_PHASE3_PATH = REPO_ROOT / "udp_test/server_bash/orin_arm/start_phase3_executor.sh"
STOP_PHASE3_PATH = REPO_ROOT / "udp_test/server_bash/orin_arm/stop_phase3_executor.sh"
STATUS_PATH = REPO_ROOT / "udp_test/server_bash/orin_arm/status.sh"


class StubJointState:
    def __init__(self) -> None:
        self.name = []
        self.position = []
        self.velocity = []
        self.effort = []


class StubFloat64MultiArray:
    def __init__(self) -> None:
        self.data = []


class FakePublisherAdapter:
    def __init__(self, cfg: phase3.Phase3ExecutorConfig) -> None:
        self.cfg = cfg
        self.published = []
        self.closed = False

    def publish_command(self, command: phase3.ValidatedCommand) -> None:
        self.published.append(
            phase3.build_ros_command_messages(
                self.cfg,
                command,
                joint_state_cls=StubJointState,
                float64_multi_array_cls=StubFloat64MultiArray,
            )
        )

    def close(self) -> None:
        self.closed = True


def joint_names() -> list[str]:
    return [f"left_joint{i}" for i in range(1, 8)] + [f"right_joint{i}" for i in range(1, 8)]


def complete_config(**overrides) -> phase3.Phase3ExecutorConfig:
    cfg = phase3.Phase3ExecutorConfig(
        joint_position_limits={joint: (-1.0, 1.0) for joint in joint_names()},
        initial_joint_positions={joint: 0.0 for joint in joint_names()},
        gripper_limits={
            "left": phase3.GripperLimits(
                width=(0.0, 100.0),
                force=(0.0, 100.0),
                max_width_delta_per_step=5.0,
                max_force_delta_per_step=5.0,
            ),
            "right": phase3.GripperLimits(
                width=(0.0, 100.0),
                force=(0.0, 100.0),
                max_width_delta_per_step=5.0,
                max_force_delta_per_step=5.0,
            ),
        },
        initial_gripper_state={
            "left": {"width": 0.0, "force": 0.0},
            "right": {"width": 0.0, "force": 0.0},
        },
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def command_packet(
    *,
    seq: int = 1,
    mode: str = COMMAND_MODE_ARMED,
    stamp_ns: int = 1_000_000_000,
    left_value: float = 0.0,
    right_value: float = 0.0,
    gripper_width: float = 0.0,
    gripper_force: float = 0.0,
) -> dict:
    return {
        "version": 1,
        "type": "command",
        "robot": "robot1",
        "seq": seq,
        "stamp_ns": stamp_ns,
        "mode": mode,
        "actions": {
            "left": {f"left_joint{i}": left_value for i in range(1, 8)},
            "right": {f"right_joint{i}": right_value for i in range(1, 8)},
            "grippers": {
                "left": {"width": gripper_width, "force": gripper_force},
                "right": {"width": gripper_width, "force": gripper_force},
            },
        },
    }


def make_executor(
    cfg: phase3.Phase3ExecutorConfig,
) -> tuple[phase3.Phase3CommandExecutor, FakePublisherAdapter]:
    publisher = FakePublisherAdapter(cfg)
    return phase3.Phase3CommandExecutor(cfg, publisher_adapter=publisher), publisher


def process(
    executor: phase3.Phase3CommandExecutor,
    packet: dict,
    *,
    sender_ip: str = "192.168.1.106",
    now_ns: int = 1_050_000_000,
    monotonic_s: float = 10.0,
) -> phase3.GateDecision:
    return executor.process_packet(
        packet,
        sender=(sender_ip, 39020),
        now_ns=now_ns,
        monotonic_s=monotonic_s,
    )


def test_default_config_and_startup_banner_are_dry_run() -> None:
    cfg = phase3.Phase3ExecutorConfig()

    assert cfg.execution == COMMAND_MODE_DRY_RUN
    banner = phase3.startup_banner(cfg, publish_enabled=False)
    assert "PHASE3 COMMAND EXECUTOR DRY-RUN" in banner
    assert "NOT publishing ROS command topics" in banner
    assert "robot should not move" in banner


def test_dry_run_executor_never_publishes_even_for_armed_packet() -> None:
    cfg = complete_config(execution=COMMAND_MODE_DRY_RUN)
    executor, publisher = make_executor(cfg)

    decision = process(executor, command_packet(mode=COMMAND_MODE_ARMED))

    assert decision.accepted
    assert not decision.publish
    assert decision.reason == "executor_not_armed"
    assert publisher.published == []
    assert executor.counters.dry_run == 1


def test_dry_run_allows_unconfigured_limits_for_regression_without_publish() -> None:
    cfg = phase3.Phase3ExecutorConfig()
    executor, publisher = make_executor(cfg)

    decision = process(executor, command_packet(mode=COMMAND_MODE_DRY_RUN))

    assert decision.accepted
    assert not decision.publish
    assert decision.reason == "dry_run_limits_unconfigured"
    assert publisher.published == []


def test_armed_startup_requires_env_and_cli_ack(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = complete_config(execution=COMMAND_MODE_ARMED)

    monkeypatch.delenv("JZ_UDP_EXECUTOR_ARMED", raising=False)
    with pytest.raises(RuntimeError, match="JZ_UDP_EXECUTOR_ARMED"):
        phase3.validate_startup_gates(cfg, cli_ack=True)

    monkeypatch.setenv("JZ_UDP_EXECUTOR_ARMED", "1")
    with pytest.raises(RuntimeError, match="CLI acknowledgement"):
        phase3.validate_startup_gates(cfg, cli_ack=False)

    assert phase3.validate_startup_gates(cfg, cli_ack=True)


def test_packet_dry_run_does_not_publish_when_executor_is_armed() -> None:
    cfg = complete_config(execution=COMMAND_MODE_ARMED)
    executor, publisher = make_executor(cfg)

    decision = process(executor, command_packet(mode=COMMAND_MODE_DRY_RUN))

    assert not decision.accepted
    assert not decision.publish
    assert decision.reason == "packet_not_armed"
    assert publisher.published == []


def test_armed_packet_publishes_only_four_expected_messages_with_ros_mapping() -> None:
    cfg = complete_config(execution=COMMAND_MODE_ARMED)
    executor, publisher = make_executor(cfg)
    packet = command_packet(left_value=0.01, right_value=-0.01, gripper_width=1.0, gripper_force=2.0)

    decision = process(executor, packet)

    assert decision.accepted
    assert decision.publish
    assert len(publisher.published) == 1
    messages = publisher.published[0]
    assert messages.left_arm.name == cfg.left_joint_names
    assert messages.left_arm.position == [0.01] * 7
    assert messages.left_arm.velocity == []
    assert messages.left_arm.effort == []
    assert messages.right_arm.name == cfg.right_joint_names
    assert messages.right_arm.position == [-0.01] * 7
    assert messages.right_arm.velocity == []
    assert messages.right_arm.effort == []
    assert messages.left_gripper.data == [1.0, 2.0]
    assert messages.right_gripper.data == [1.0, 2.0]
    assert executor.counters.published == 1
    assert executor.counters.last_published_seq == 1


def test_sender_ip_gate_rejects_unexpected_sender_without_publish() -> None:
    cfg = complete_config(execution=COMMAND_MODE_ARMED)
    executor, publisher = make_executor(cfg)

    decision = process(executor, command_packet(), sender_ip="192.168.1.200")

    assert not decision.accepted
    assert decision.reason == "unexpected_sender"
    assert executor.counters.unexpected_sender == 1
    assert publisher.published == []


def test_seq_gate_requires_monotonic_seq_and_counts_gaps() -> None:
    cfg = complete_config(execution=COMMAND_MODE_ARMED, max_publish_hz=100.0)
    executor, publisher = make_executor(cfg)

    assert process(executor, command_packet(seq=2), monotonic_s=10.0).accepted
    repeated = process(executor, command_packet(seq=2), monotonic_s=10.1)
    assert not repeated.accepted
    assert repeated.reason == "non_monotonic_seq"
    assert process(executor, command_packet(seq=5), monotonic_s=10.2).accepted
    assert executor.counters.seq_gap_count == 2
    assert len(publisher.published) == 2


def test_stamp_age_gate_rejects_stale_and_future_packets() -> None:
    cfg = complete_config(execution=COMMAND_MODE_ARMED)
    executor, publisher = make_executor(cfg)

    stale = process(executor, command_packet(stamp_ns=1_000_000_000), now_ns=1_400_000_000)
    future = process(executor, command_packet(seq=2, stamp_ns=3_000_000_000), now_ns=1_400_000_000)

    assert stale.reason == "stale_command"
    assert future.reason == "future_command"
    assert publisher.published == []


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda packet: packet["actions"]["left"].pop("left_joint1"), "joint_name_mismatch"),
        (lambda packet: packet["actions"]["right"].__setitem__("right_joint99", 0.0), "joint_name_mismatch"),
        (lambda packet: packet["actions"]["left"].__setitem__("left_joint1", math.nan), "invalid_packet"),
        (lambda packet: packet["actions"]["right"].__setitem__("right_joint1", math.inf), "invalid_packet"),
        (lambda packet: packet["actions"]["grippers"]["left"].__setitem__("width", True), "invalid_packet"),
    ],
)
def test_finite_numeric_and_joint_name_gates_reject_bad_packets(mutate, reason: str) -> None:
    cfg = complete_config(execution=COMMAND_MODE_ARMED)
    executor, publisher = make_executor(cfg)
    packet = command_packet()
    mutate(packet)

    decision = process(executor, packet)

    assert not decision.accepted
    assert decision.reason == reason
    assert publisher.published == []


def test_joint_absolute_range_and_delta_gates_reject_publish() -> None:
    cfg = complete_config(execution=COMMAND_MODE_ARMED, max_publish_hz=100.0)
    executor, publisher = make_executor(cfg)

    out_of_range = process(executor, command_packet(left_value=2.0), monotonic_s=10.0)
    assert out_of_range.reason == "joint_position_limit"

    assert process(executor, command_packet(seq=2, left_value=0.01), monotonic_s=10.1).accepted
    too_large_delta = process(executor, command_packet(seq=3, left_value=0.5), monotonic_s=10.2)
    assert too_large_delta.reason == "joint_delta_limit"
    assert len(publisher.published) == 1


def test_joint_absolute_range_bypass_still_enforces_delta_gate() -> None:
    cfg = complete_config(
        execution=COMMAND_MODE_ARMED,
        max_publish_hz=100.0,
        allow_joint_position_limit_bypass=True,
    )
    cfg.initial_joint_positions = {joint: 1.99 for joint in joint_names()}
    executor, publisher = make_executor(cfg)

    bypassed = process(executor, command_packet(left_value=2.0, right_value=2.0), monotonic_s=10.0)
    assert bypassed.accepted
    assert bypassed.publish
    assert bypassed.reason == "published"

    too_large_delta = process(
        executor,
        command_packet(seq=2, left_value=2.5, right_value=2.5),
        monotonic_s=10.1,
    )
    assert not too_large_delta.accepted
    assert too_large_delta.reason == "joint_delta_limit"
    assert len(publisher.published) == 1


def test_joint_delta_bypass_allows_large_step_when_explicitly_enabled() -> None:
    cfg = complete_config(
        execution=COMMAND_MODE_ARMED,
        max_publish_hz=100.0,
        allow_joint_position_limit_bypass=True,
        allow_joint_delta_limit_bypass=True,
    )
    executor, publisher = make_executor(cfg)

    decision = process(executor, command_packet(left_value=2.5, right_value=2.5), monotonic_s=10.0)

    assert decision.accepted
    assert decision.publish
    assert decision.reason == "published"
    assert len(publisher.published) == 1


def test_first_armed_command_requires_initial_joint_and_gripper_state() -> None:
    cfg = complete_config(execution=COMMAND_MODE_ARMED)
    cfg.initial_joint_positions = {}
    executor, publisher = make_executor(cfg)

    decision = process(executor, command_packet())

    assert not decision.accepted
    assert decision.reason == "initial_joint_state_required"
    assert publisher.published == []


def test_gripper_range_and_delta_gates_reject_publish() -> None:
    cfg = complete_config(execution=COMMAND_MODE_ARMED, max_publish_hz=100.0)
    executor, publisher = make_executor(cfg)

    out_of_range = process(executor, command_packet(gripper_width=200.0), monotonic_s=10.0)
    assert out_of_range.reason == "gripper_limit"

    assert process(executor, command_packet(seq=2, gripper_width=1.0, gripper_force=1.0), monotonic_s=10.1).accepted
    too_large_delta = process(executor, command_packet(seq=3, gripper_width=20.0, gripper_force=1.0), monotonic_s=10.2)
    assert too_large_delta.reason == "gripper_delta_limit"
    assert len(publisher.published) == 1


def test_publish_rate_limit_drops_fast_packets_without_bursting() -> None:
    cfg = complete_config(execution=COMMAND_MODE_ARMED, max_publish_hz=10.0)
    executor, publisher = make_executor(cfg)

    assert process(executor, command_packet(seq=1), monotonic_s=10.0).publish
    limited = process(executor, command_packet(seq=2), monotonic_s=10.01)

    assert limited.reason == "rate_limited"
    assert executor.counters.rate_limited == 1
    assert len(publisher.published) == 1


def test_command_timeout_marks_inactive_without_publishing_stop_pose() -> None:
    cfg = complete_config(execution=COMMAND_MODE_ARMED)
    executor, publisher = make_executor(cfg)

    assert process(executor, command_packet(), monotonic_s=10.0).publish
    executor.check_command_timeout(monotonic_s=10.31)

    assert not executor.active
    assert executor.counters.timeout == 1
    assert len(publisher.published) == 1


def test_shutdown_closes_adapter_without_extra_publish() -> None:
    cfg = complete_config(execution=COMMAND_MODE_ARMED)
    executor, publisher = make_executor(cfg)

    executor.request_shutdown()
    executor.close()

    assert executor.shutdown_requested
    assert publisher.closed
    assert publisher.published == []


def test_armed_startup_rejects_missing_invalid_or_overwide_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JZ_UDP_EXECUTOR_ARMED", "1")

    missing_sender_ip = complete_config(execution=COMMAND_MODE_ARMED, allowed_sender_ip=None)
    with pytest.raises(RuntimeError, match="allowed_sender_ip"):
        phase3.validate_startup_gates(missing_sender_ip, cli_ack=True)

    wildcard_sender_ip = complete_config(execution=COMMAND_MODE_ARMED, allowed_sender_ip="0.0.0.0")
    with pytest.raises(RuntimeError, match="allowed_sender_ip"):
        phase3.validate_startup_gates(wildcard_sender_ip, cli_ack=True)

    wrong_topic = complete_config(execution=COMMAND_MODE_ARMED)
    wrong_topic.left_arm_command_topic = "/robot1/other/joint_commands_input"
    with pytest.raises(RuntimeError, match="left_arm_command_topic"):
        phase3.validate_startup_gates(wrong_topic, cli_ack=True)

    missing_joint = complete_config(execution=COMMAND_MODE_ARMED)
    missing_joint.joint_position_limits.pop("left_joint1")
    with pytest.raises(RuntimeError, match="joint_position_limits"):
        phase3.validate_startup_gates(missing_joint, cli_ack=True)

    invalid_joint = complete_config(execution=COMMAND_MODE_ARMED)
    invalid_joint.joint_position_limits["left_joint1"] = ("TODO_MIN", "TODO_MAX")
    with pytest.raises(RuntimeError, match="joint_position_limits"):
        phase3.validate_startup_gates(invalid_joint, cli_ack=True)

    overwide_joint = complete_config(execution=COMMAND_MODE_ARMED)
    overwide_joint.joint_position_limits["left_joint1"] = (-1_000_000.0, 1_000_000.0)
    with pytest.raises(RuntimeError, match="joint_position_limits"):
        phase3.validate_startup_gates(overwide_joint, cli_ack=True)

    missing_gripper = complete_config(execution=COMMAND_MODE_ARMED)
    missing_gripper.gripper_limits.pop("left")
    with pytest.raises(RuntimeError, match="gripper_limits"):
        phase3.validate_startup_gates(missing_gripper, cli_ack=True)

    overwide_gripper_delta = complete_config(execution=COMMAND_MODE_ARMED)
    overwide_gripper_delta.gripper_limits["left"].max_width_delta_per_step = 999_999.0
    with pytest.raises(RuntimeError, match="gripper_limits width delta"):
        phase3.validate_startup_gates(overwide_gripper_delta, cli_ack=True)


def test_armed_rejects_without_publisher_before_remembering_command_state() -> None:
    cfg = complete_config(execution=COMMAND_MODE_ARMED)
    executor = phase3.Phase3CommandExecutor(cfg, publisher_adapter=None)

    decision = process(executor, command_packet(), monotonic_s=10.0)

    assert not decision.accepted
    assert decision.reason == "publisher_unavailable"
    assert not executor.active
    assert executor.counters.last_seq is None
    assert executor.last_joint_positions is None
    assert executor.last_gripper_state is None


def test_phase3_executor_static_safety_surface() -> None:
    text = EXECUTOR_PATH.read_text()
    assert "cmd_vel" not in text
    assert "waist" not in text
    assert "body" not in text
    assert "/robot1/telecon/arm_left/joint_commands_input" in text
    assert "/robot1/telecon/arm_right/joint_commands_input" in text
    assert "/robot1/left_gripper/gripper_commands" in text
    assert "/robot1/right_gripper/gripper_commands" in text


def test_phase2_receiver_still_has_no_ros_publish_path() -> None:
    tree = ast.parse(PHASE2_RECEIVER_PATH.read_text())

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".", maxsplit=1)[0] != "rclpy" for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".", maxsplit=1)[0] != "rclpy"
        elif isinstance(node, ast.Attribute):
            assert node.attr not in {"create_publisher", "publish"}


def test_x86_udp_robot_has_no_ros_publish_path() -> None:
    tree = ast.parse(X86_ROBOT_PATH.read_text())

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = {alias.name.split(".", maxsplit=1)[0] for alias in node.names}
            assert not (imported & {"rclpy", "sensor_msgs", "std_msgs"})
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", maxsplit=1)[0]
            assert root not in {"rclpy", "sensor_msgs", "std_msgs"}
        elif isinstance(node, ast.Attribute):
            assert node.attr not in {"create_publisher", "publish"}


def test_phase3_config_file_defaults_to_dry_run_and_documents_required_limits() -> None:
    text = EXECUTOR_CONFIG_PATH.read_text()

    assert 'execution: "dry_run"' in text
    assert "allowed_sender_ip" in text
    assert "command_port: 39020" in text
    assert "joint_position_limits" in text
    assert "gripper_limits" in text
    assert "TODO_MIN" in text


def test_phase3_server_scripts_default_dry_run_and_require_armed_env() -> None:
    start_text = START_PHASE3_PATH.read_text()
    stop_text = STOP_PHASE3_PATH.read_text()
    status_text = STATUS_PATH.read_text()

    assert 'EXECUTION="${EXECUTION:-dry_run}"' in start_text
    assert "JZ_UDP_EXECUTOR_ARMED" in start_text
    assert "--execution" in start_text
    assert "--i-understand-this-publishes-robot-commands" in start_text
    assert "PHASE3 COMMAND EXECUTOR DRY-RUN" in start_text
    assert "PHASE3 COMMAND EXECUTOR ARMED" in start_text
    assert "orin_phase3_command_executor.py" in stop_text
    assert "orin_phase3_command_executor.py" in status_text
    assert "cmd_vel" not in start_text
    assert "cmd_vel" not in stop_text
