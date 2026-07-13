from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest

from lerobot.robots.jz_robot_udp.protocol import COMMAND_MODE_ARMED
from udp_test.test_scripts.arm_side import orin_phase3_command_executor as phase3
from udp_test.test_scripts.arm_side.jz_pin_reset_control import (
    ARMCONTROL_FINAL_POSE_STABLE_S,
    ARMCONTROL_FINAL_POSE_TOLERANCE_RAD,
    ChoreographyResult,
    ResetControlConfig,
    ResetControlServer,
    ResetCoordinator,
    ResetState,
    load_choreography_targets,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
START_PHASE3 = REPO_ROOT / "udp_test/server_bash/orin_arm/start_phase3_executor.sh"
ACTIONS_PATH = Path(
    "/home/data/test/workspace/teleop_ws/install/multi_robot_choreographer/share/"
    "multi_robot_choreographer/config/actions/actions.yaml"
)


class FakePublisher:
    def __init__(self) -> None:
        self.published: list[phase3.ValidatedCommand] = []

    def publish_command(self, command: phase3.ValidatedCommand) -> None:
        self.published.append(command)

    def close(self) -> None:
        pass


def executor_config() -> phase3.Phase3ExecutorConfig:
    joint_names = [f"left_joint{i}" for i in range(1, 8)] + [f"right_joint{i}" for i in range(1, 8)]
    return phase3.Phase3ExecutorConfig(
        execution=COMMAND_MODE_ARMED,
        command_timeout_s=0.3,
        max_publish_hz=100.0,
        joint_position_limits={joint: (-2.0, 2.0) for joint in joint_names},
        initial_joint_positions={joint: 0.0 for joint in joint_names},
        gripper_limits={
            side: phase3.GripperLimits(
                width=(0.0, 100.0),
                force=(0.0, 100.0),
                max_width_delta_per_step=5.0,
                max_force_delta_per_step=5.0,
            )
            for side in ("left", "right")
        },
        initial_gripper_state={
            "left": {"width": 0.0, "force": 0.0},
            "right": {"width": 0.0, "force": 0.0},
        },
    )


def command_packet(seq: int = 1) -> dict:
    return {
        "version": 1,
        "type": "command",
        "robot": "robot1",
        "seq": seq,
        "stamp_ns": 1_000_000_000,
        "mode": COMMAND_MODE_ARMED,
        "actions": {
            "left": {f"left_joint{i}": 0.0 for i in range(1, 8)},
            "right": {f"right_joint{i}": 0.0 for i in range(1, 8)},
            "grippers": {
                "left": {"width": 0.0, "force": 0.0},
                "right": {"width": 0.0, "force": 0.0},
            },
        },
    }


@dataclass
class FakeClock:
    wall_ns: int = 10_000_000_000
    monotonic_s: float = 10.0

    def wait(self, seconds: float) -> None:
        self.wall_ns += int(seconds * 1_000_000_000)
        self.monotonic_s += seconds


class FakeAdapter:
    def __init__(self, result: ChoreographyResult, callback=None) -> None:
        self.result = result
        self.callback = callback
        self.calls: list[str] = []
        self.closed = False

    def execute_and_wait(self, choreography_name: str) -> ChoreographyResult:
        self.calls.append(choreography_name)
        if self.callback is not None:
            self.callback()
        return self.result

    def close(self) -> None:
        self.closed = True


def successful_result() -> ChoreographyResult:
    return ChoreographyResult(
        started=True,
        completed=True,
        success=True,
        execution_id=17,
        status_message="执行完成",
        final_pose_check={"passed": True, "max_error_rad": 0.001},
    )


def reset_request(request_id: str | None = None, **overrides) -> dict:
    request = {
        "protocol": "jz_pin_reset_control",
        "protocol_version": 1,
        "request_id": request_id or str(uuid.uuid4()),
        "operation": "reset",
        "choreography_name": "VR_inital_no_waist",
        "stamp_ns": 10_000_000_000,
    }
    request.update(overrides)
    return request


def resume_request(reset_request_id: str, **overrides) -> dict:
    request = {
        "protocol": "jz_pin_reset_control",
        "protocol_version": 1,
        "request_id": str(uuid.uuid4()),
        "operation": "resume",
        "reset_request_id": reset_request_id,
        "stamp_ns": 10_300_000_000,
    }
    request.update(overrides)
    return request


def coordinator(result: ChoreographyResult, *, callback=None):
    clock = FakeClock()
    executor = phase3.Phase3CommandExecutor(executor_config(), publisher_adapter=FakePublisher())
    adapter = FakeAdapter(result, callback=callback)
    cfg = ResetControlConfig(actions_path=ACTIONS_PATH)
    control = ResetCoordinator(
        cfg,
        executor,
        adapter,
        wall_time_ns=lambda: clock.wall_ns,
        monotonic=lambda: clock.monotonic_s,
        wait=clock.wait,
    )
    return control, executor, adapter, clock


def test_choreography_targets_are_loaded_from_deployed_definition() -> None:
    targets = load_choreography_targets(ACTIONS_PATH, "VR_inital_no_waist")

    assert list(targets) == ["left", "right"]
    assert targets["left"]["left_joint1"] == pytest.approx(-0.050562189728566756)
    assert targets["left"]["left_joint7"] == pytest.approx(0.0009773844112854462)
    assert targets["right"]["right_joint1"] == pytest.approx(0.05035274875358883)
    assert targets["right"]["right_joint7"] == pytest.approx(-0.0016057029149556751)
    assert ARMCONTROL_FINAL_POSE_TOLERANCE_RAD == pytest.approx(0.03490658503988659)
    assert ARMCONTROL_FINAL_POSE_STABLE_S == 0.5


@pytest.mark.parametrize(
    ("source_ip", "overrides", "reason"),
    [
        ("192.168.1.99", {}, "unexpected_sender"),
        ("192.168.1.106", {"choreography_name": "not_allowed"}, "choreography_not_allowed"),
        ("192.168.1.106", {"stamp_ns": 1_000_000_000}, "stale_request"),
        ("192.168.1.106", {"stamp_ns": 12_000_000_000}, "future_request"),
    ],
)
def test_invalid_reset_requests_are_rejected_without_inhibiting(source_ip, overrides, reason) -> None:
    control, executor, adapter, _clock = coordinator(successful_result())

    response = control.handle(reset_request(**overrides), source_ip=source_ip)

    assert response["status"] == "rejected"
    assert response["message"] == reason
    assert control.state == ResetState.ACTIVE
    assert not executor.commands_inhibited
    assert adapter.calls == []


def test_duplicate_request_id_is_rejected() -> None:
    control, _executor, adapter, clock = coordinator(successful_result())
    request_id = str(uuid.uuid4())
    first = control.handle(reset_request(request_id), source_ip="192.168.1.106")
    clock.wait(0.1)
    duplicate = control.handle(reset_request(request_id, stamp_ns=clock.wall_ns), source_ip="192.168.1.106")

    assert first["status"] == "completed"
    assert duplicate["status"] == "rejected"
    assert duplicate["message"] == "duplicate_request_id"
    assert adapter.calls == ["VR_inital_no_waist"]


def test_reset_inhibits_before_service_and_drops_commands_without_caching() -> None:
    holder = {}

    def during_reset() -> None:
        executor = holder["executor"]
        assert holder["control"].state == ResetState.RESETTING
        decision = executor.process_packet(
            command_packet(2),
            sender=("192.168.1.106", 39020),
            now_ns=1_050_000_000,
            monotonic_s=10.1,
        )
        holder["decision"] = decision

    control, executor, _adapter, clock = coordinator(successful_result(), callback=during_reset)
    holder["executor"] = executor
    holder["control"] = control
    pre_reset = executor.process_packet(
        command_packet(1),
        sender=("192.168.1.106", 39020),
        now_ns=1_050_000_000,
        monotonic_s=9.9,
    )
    assert pre_reset.publish

    response = control.handle(reset_request(), source_ip="192.168.1.106")

    assert response["status"] == "completed"
    assert response["state"] == ResetState.RESET_COMPLETE_INHIBITED.value
    assert holder["decision"].reason == "command_inhibited"
    assert executor.counters.command_inhibited_drops == 1
    assert executor.latest_target is None
    assert not executor.active
    assert executor.commands_inhibited
    assert clock.monotonic_s == pytest.approx(10.3)


@pytest.mark.parametrize(
    "result",
    [
        ChoreographyResult(started=True, completed=False, execution_id=3, status_message="still_running"),
        ChoreographyResult(
            started=True,
            completed=True,
            success=False,
            execution_id=4,
            status_message="movement timeout",
            final_pose_check={"passed": False},
        ),
        ChoreographyResult(started=False, completed=False, status_message="service unavailable"),
    ],
)
def test_started_only_failure_and_timeout_remain_inhibited(result) -> None:
    control, executor, _adapter, _clock = coordinator(result)

    response = control.handle(reset_request(), source_ip="192.168.1.106")

    assert response["status"] == "failed"
    assert control.state == ResetState.RESET_FAILED_INHIBITED
    assert executor.commands_inhibited


def test_only_resume_linked_to_successful_reset_restores_command_acceptance() -> None:
    control, executor, _adapter, clock = coordinator(successful_result())
    reset_id = str(uuid.uuid4())
    assert control.handle(reset_request(reset_id), source_ip="192.168.1.106")["status"] == "completed"

    mismatch = control.handle(
        resume_request(str(uuid.uuid4()), stamp_ns=clock.wall_ns), source_ip="192.168.1.106"
    )
    assert mismatch["status"] == "rejected"
    assert mismatch["message"] == "reset_request_id_mismatch"
    assert executor.commands_inhibited

    clock.wait(0.1)
    resumed = control.handle(
        resume_request(reset_id, stamp_ns=clock.wall_ns), source_ip="192.168.1.106"
    )
    assert resumed["status"] == "resumed"
    assert resumed["state"] == ResetState.ACTIVE.value
    assert not executor.commands_inhibited
    assert executor.latest_target is None
    assert executor.counters.resume_transitions == 1


def test_failed_reset_cannot_be_resumed() -> None:
    control, executor, _adapter, clock = coordinator(
        ChoreographyResult(started=True, completed=False, execution_id=9, status_message="timeout")
    )
    reset_id = str(uuid.uuid4())
    control.handle(reset_request(reset_id), source_ip="192.168.1.106")
    response = control.handle(
        resume_request(reset_id, stamp_ns=clock.wall_ns), source_ip="192.168.1.106"
    )
    assert response["status"] == "rejected"
    assert response["message"] == "resume_not_allowed_in_RESET_FAILED_INHIBITED"
    assert executor.commands_inhibited


def test_tcp_server_stops_without_a_thread_residue() -> None:
    control, _executor, adapter, _clock = coordinator(successful_result())
    control.cfg.bind_ip = "127.0.0.1"
    control.cfg.allowed_sender_ip = "127.0.0.1"
    control.cfg.port = 0
    server = ResetControlServer(control.cfg, control)
    server.start()

    server.close()

    assert not server._thread.is_alive()
    assert adapter.closed


def test_tcp_wire_response_is_json_line_and_protocol_v1() -> None:
    control, _executor, _adapter, clock = coordinator(successful_result())
    request = reset_request(stamp_ns=clock.wall_ns)
    response = control.handle(json.loads(json.dumps(request)), source_ip="192.168.1.106")

    encoded = json.dumps(response, separators=(",", ":")) + "\n"
    decoded = json.loads(encoded)
    assert decoded["protocol"] == "jz_pin_reset_control"
    assert decoded["protocol_version"] == 1
    assert decoded["request_id"] == request["request_id"]
    assert decoded["status"] == "completed"
    assert decoded["execution_id"] == 17
    assert decoded["final_pose_check"]["passed"]


def test_armed_start_wrapper_enables_reset_endpoint_fail_closed() -> None:
    text = START_PHASE3.read_text(encoding="utf-8")

    assert 'RESET_CONTROL_PORT="${RESET_CONTROL_PORT:-39040}"' in text
    assert 'RESET_CONTROL_ENABLED="${RESET_CONTROL_ENABLED:-1}"' in text
    assert "--reset-control-enabled" in text
    assert "--reset-control-allowed-sender-ip" in text
    assert "RESET_CONTROL_ENABLED=1 is required" in text
