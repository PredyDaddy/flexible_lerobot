#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

RESET_PROTOCOL = "jz_pin_reset_control"
RESET_PROTOCOL_VERSION = 1
RESET_OPERATION = "reset"
RESUME_OPERATION = "resume"
DEFAULT_RESET_PORT = 39040
DEFAULT_CHOREOGRAPHY = "VR_inital_no_waist"
DEFAULT_ACTIONS_PATH = Path(
    "/home/data/test/workspace/teleop_ws/install/multi_robot_choreographer/share/"
    "multi_robot_choreographer/config/actions/actions.yaml"
)

# armcontrol registers its SDK completion monitor with these exact values.
ARMCONTROL_FINAL_POSE_TOLERANCE_RAD = math.radians(2.0)
ARMCONTROL_FINAL_POSE_STABLE_S = 0.5


class ResetState(str, Enum):
    ACTIVE = "ACTIVE"
    COMMAND_QUIET = "COMMAND_QUIET"
    RESETTING = "RESETTING"
    RESET_COMPLETE_INHIBITED = "RESET_COMPLETE_INHIBITED"
    RESET_FAILED_INHIBITED = "RESET_FAILED_INHIBITED"


@dataclass
class ResetControlConfig:
    bind_ip: str = "192.168.1.81"
    port: int = DEFAULT_RESET_PORT
    allowed_sender_ip: str = "192.168.1.106"
    allowed_choreographies: tuple[str, ...] = (DEFAULT_CHOREOGRAPHY,)
    max_request_age_s: float = 5.0
    max_clock_skew_s: float = 1.0
    choreography_service_timeout_s: float = 5.0
    choreography_completion_timeout_s: float = 35.0
    status_poll_interval_s: float = 0.1
    final_pose_timeout_s: float = 3.0
    final_pose_tolerance_rad: float = ARMCONTROL_FINAL_POSE_TOLERANCE_RAD
    final_pose_stable_s: float = ARMCONTROL_FINAL_POSE_STABLE_S
    actions_path: Path = DEFAULT_ACTIONS_PATH
    execute_service: str = "/robot1/choreographer/execute"
    status_service: str = "/robot1/choreographer/execution_status"
    left_joint_state_topic: str = "/robot1/arm_left/joint_states"
    right_joint_state_topic: str = "/robot1/arm_right/joint_states"
    max_request_bytes: int = 64 * 1024
    socket_timeout_s: float = 0.2
    seen_request_limit: int = 1024


@dataclass
class ChoreographyResult:
    started: bool = False
    completed: bool = False
    success: bool = False
    execution_id: int | None = None
    status_message: str = ""
    final_pose_check: dict[str, Any] = field(default_factory=lambda: {"passed": False})


class ResetProtocolError(ValueError):
    pass


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _canonical_uuid(value: Any) -> str:
    if not isinstance(value, str):
        raise ResetProtocolError("request_id must be a UUID string")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ResetProtocolError("request_id must be a valid UUID") from exc
    if str(parsed) != value.lower():
        raise ResetProtocolError("request_id must use canonical UUID form")
    return str(parsed)


def validate_reset_request(
    request: Any,
    *,
    cfg: ResetControlConfig,
    source_ip: str,
    now_ns: int,
) -> dict[str, Any]:
    if source_ip != cfg.allowed_sender_ip:
        raise ResetProtocolError("unexpected_sender")
    if not isinstance(request, dict):
        raise ResetProtocolError("request must be a JSON object")
    if request.get("protocol") != RESET_PROTOCOL:
        raise ResetProtocolError("protocol mismatch")
    if request.get("protocol_version") != RESET_PROTOCOL_VERSION:
        raise ResetProtocolError("protocol_version mismatch")

    request_id = _canonical_uuid(request.get("request_id"))
    operation = request.get("operation")
    if operation not in {RESET_OPERATION, RESUME_OPERATION}:
        raise ResetProtocolError("operation must be reset or resume")
    stamp_ns = request.get("stamp_ns")
    if not _is_int(stamp_ns) or stamp_ns <= 0:
        raise ResetProtocolError("stamp_ns must be a positive integer")
    age_s = (now_ns - stamp_ns) / 1_000_000_000.0
    if age_s > cfg.max_request_age_s:
        raise ResetProtocolError("stale_request")
    if age_s < -cfg.max_clock_skew_s:
        raise ResetProtocolError("future_request")

    normalized = dict(request)
    normalized["request_id"] = request_id
    if operation == RESET_OPERATION:
        name = request.get("choreography_name")
        if name not in cfg.allowed_choreographies:
            raise ResetProtocolError("choreography_not_allowed")
    else:
        normalized["reset_request_id"] = _canonical_uuid(request.get("reset_request_id"))
    return normalized


def load_choreography_targets(actions_path: Path, choreography_name: str) -> dict[str, dict[str, float]]:
    import yaml

    document = yaml.safe_load(actions_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise RuntimeError(f"invalid choreography actions file: {actions_path}")
    actions = document.get("actions")
    choreographies = document.get("choreographies")
    if not isinstance(actions, dict) or not isinstance(choreographies, dict):
        raise RuntimeError(f"missing actions/choreographies in {actions_path}")
    steps = choreographies.get(choreography_name)
    if not isinstance(steps, list) or not steps:
        raise RuntimeError(f"missing choreography {choreography_name!r} in {actions_path}")

    targets: dict[str, dict[str, float]] = {}
    robot_mapping = {
        2: [f"left_joint{i}" for i in range(1, 8)],
        1: [f"right_joint{i}" for i in range(1, 8)],
    }
    for step in steps:
        if not isinstance(step, dict):
            raise RuntimeError(f"invalid step in choreography {choreography_name!r}")
        for action_name in step.get("actions", []):
            action = actions.get(action_name)
            if not isinstance(action, dict):
                raise RuntimeError(f"missing action {action_name!r}")
            robot_id = action.get("robot_id")
            if robot_id not in robot_mapping:
                raise RuntimeError(
                    f"choreography {choreography_name!r} controls unsupported robot_id={robot_id}"
                )
            joints = action.get("joints")
            names = robot_mapping[robot_id]
            if not isinstance(joints, list) or len(joints) != len(names):
                raise RuntimeError(f"invalid joints for action {action_name!r}")
            targets.update({name: float(value) for name, value in zip(names, joints, strict=True)})
    if set(targets) != set(robot_mapping[1] + robot_mapping[2]):
        raise RuntimeError(f"choreography {choreography_name!r} must define both seven-joint arms")
    return {
        "left": {name: targets[name] for name in robot_mapping[2]},
        "right": {name: targets[name] for name in robot_mapping[1]},
    }


class RosChoreographyAdapter:
    def __init__(self, cfg: ResetControlConfig) -> None:
        import rclpy
        from multi_robot_choreographer_interfaces.srv import ExecuteChoreography, GetExecutionStatus
        from sensor_msgs.msg import JointState

        self.cfg = cfg
        self._rclpy = rclpy
        self._execute_type = ExecuteChoreography
        self._status_type = GetExecutionStatus
        self._targets = {
            name: load_choreography_targets(cfg.actions_path, name) for name in cfg.allowed_choreographies
        }
        self._joint_lock = threading.Lock()
        self._joint_states: dict[str, dict[str, float]] = {}
        self._stop = threading.Event()
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("jz_pin_reset_control")
        self._execute_client = self._node.create_client(ExecuteChoreography, cfg.execute_service)
        self._status_client = self._node.create_client(GetExecutionStatus, cfg.status_service)
        self._left_sub = self._node.create_subscription(
            JointState, cfg.left_joint_state_topic, lambda msg: self._on_joint_state("left", msg), 1
        )
        self._right_sub = self._node.create_subscription(
            JointState, cfg.right_joint_state_topic, lambda msg: self._on_joint_state("right", msg), 1
        )

    def _on_joint_state(self, side: str, message: Any) -> None:
        if len(message.name) != len(message.position):
            return
        values = {str(name): float(value) for name, value in zip(message.name, message.position, strict=True)}
        with self._joint_lock:
            self._joint_states[side] = values

    def _call(self, client: Any, request: Any, timeout_s: float) -> Any | None:
        future = client.call_async(request)
        self._rclpy.spin_until_future_complete(self._node, future, timeout_sec=timeout_s)
        if not future.done() or future.exception() is not None:
            return None
        return future.result()

    def execute_and_wait(self, choreography_name: str) -> ChoreographyResult:
        if not self._execute_client.wait_for_service(timeout_sec=self.cfg.choreography_service_timeout_s):
            return ChoreographyResult(status_message="execute_service_unavailable")
        if not self._status_client.wait_for_service(timeout_sec=self.cfg.choreography_service_timeout_s):
            return ChoreographyResult(status_message="status_service_unavailable")

        execute_request = self._execute_type.Request()
        execute_request.choreography_name = choreography_name
        response = self._call(
            self._execute_client, execute_request, self.cfg.choreography_service_timeout_s
        )
        if response is None:
            return ChoreographyResult(status_message="execute_service_timeout")
        execution_id = int(response.execution_id)
        if not response.success or execution_id <= 0:
            return ChoreographyResult(
                started=False,
                execution_id=execution_id or None,
                status_message=str(response.message),
            )

        deadline = time.monotonic() + self.cfg.choreography_completion_timeout_s
        while time.monotonic() < deadline:
            if self._stop.is_set():
                return ChoreographyResult(
                    started=True,
                    execution_id=execution_id,
                    status_message="reset_adapter_shutdown",
                )
            status = self._call(
                self._status_client,
                self._status_type.Request(),
                self.cfg.choreography_service_timeout_s,
            )
            if status is None:
                return ChoreographyResult(
                    started=True,
                    execution_id=execution_id,
                    status_message="execution_status_timeout",
                )
            if int(status.last_finished_execution_id) == execution_id:
                completed_successfully = bool(status.last_execution_success)
                if str(status.last_finished_choreography) != choreography_name:
                    completed_successfully = False
                pose_check = (
                    self._wait_for_final_pose(choreography_name)
                    if completed_successfully
                    else {"passed": False, "reason": "choreography_failed"}
                )
                return ChoreographyResult(
                    started=True,
                    completed=True,
                    success=completed_successfully and bool(pose_check.get("passed")),
                    execution_id=execution_id,
                    status_message=str(status.status_message),
                    final_pose_check=pose_check,
                )
            if int(status.last_finished_execution_id) > execution_id:
                return ChoreographyResult(
                    started=True,
                    execution_id=execution_id,
                    status_message=(
                        "execution_status_lost:"
                        f"last_finished_execution_id={int(status.last_finished_execution_id)}"
                    ),
                )
            active_id = int(status.active_execution_id)
            if active_id not in (0, execution_id):
                return ChoreographyResult(
                    started=True,
                    execution_id=execution_id,
                    status_message=f"execution_superseded_by_{active_id}",
                )
            time.sleep(self.cfg.status_poll_interval_s)
        return ChoreographyResult(
            started=True,
            execution_id=execution_id,
            status_message="choreography_completion_timeout",
        )

    def _wait_for_final_pose(self, choreography_name: str) -> dict[str, Any]:
        targets = self._targets[choreography_name]
        deadline = time.monotonic() + self.cfg.final_pose_timeout_s
        stable_since: float | None = None
        max_error = float("inf")
        while time.monotonic() < deadline:
            if self._stop.is_set():
                return {"passed": False, "reason": "reset_adapter_shutdown"}
            self._rclpy.spin_once(self._node, timeout_sec=min(0.05, self.cfg.final_pose_timeout_s))
            with self._joint_lock:
                states = {side: dict(values) for side, values in self._joint_states.items()}
            errors: list[float] = []
            complete = True
            for side in ("left", "right"):
                state = states.get(side, {})
                for name, target in targets[side].items():
                    if name not in state:
                        complete = False
                        break
                    errors.append(abs(state[name] - target))
            max_error = max(errors, default=float("inf")) if complete else float("inf")
            now = time.monotonic()
            if complete and max_error <= self.cfg.final_pose_tolerance_rad:
                stable_since = now if stable_since is None else stable_since
                if now - stable_since >= self.cfg.final_pose_stable_s:
                    return {
                        "passed": True,
                        "max_error_rad": max_error,
                        "tolerance_rad": self.cfg.final_pose_tolerance_rad,
                        "stable_s": self.cfg.final_pose_stable_s,
                        "source": "armcontrol_setEnableTolerance",
                    }
            else:
                stable_since = None
        return {
            "passed": False,
            "reason": "final_pose_timeout",
            "max_error_rad": max_error if math.isfinite(max_error) else None,
            "tolerance_rad": self.cfg.final_pose_tolerance_rad,
            "stable_s": self.cfg.final_pose_stable_s,
            "source": "armcontrol_setEnableTolerance",
        }

    def close(self) -> None:
        self._stop.set()
        if self._node is not None:
            self._node.destroy_node()
            self._node = None

    def request_stop(self) -> None:
        self._stop.set()


class ResetCoordinator:
    def __init__(
        self,
        cfg: ResetControlConfig,
        executor: Any,
        choreography_adapter: Any,
        *,
        wall_time_ns: Callable[[], int] = time.time_ns,
        monotonic: Callable[[], float] = time.monotonic,
        wait: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg = cfg
        self.executor = executor
        self.choreography_adapter = choreography_adapter
        self.wall_time_ns = wall_time_ns
        self.monotonic = monotonic
        self.wait = wait
        self.state = ResetState.ACTIVE
        self.last_reset_request_id: str | None = None
        self._seen_ids: list[str] = []
        self._seen_set: set[str] = set()
        self._lock = threading.Lock()

    def _remember_request_id(self, request_id: str) -> None:
        if request_id in self._seen_set:
            raise ResetProtocolError("duplicate_request_id")
        self._seen_ids.append(request_id)
        self._seen_set.add(request_id)
        if len(self._seen_ids) > self.cfg.seen_request_limit:
            expired = self._seen_ids.pop(0)
            self._seen_set.remove(expired)

    def _response(
        self,
        request_id: str | None,
        status: str,
        *,
        execution_id: int | None = None,
        started_wall_ns: int | None = None,
        completed_wall_ns: int | None = None,
        final_pose_check: dict[str, Any] | None = None,
        message: str = "",
    ) -> dict[str, Any]:
        return {
            "protocol": RESET_PROTOCOL,
            "protocol_version": RESET_PROTOCOL_VERSION,
            "request_id": request_id,
            "status": status,
            "state": self.state.value,
            "execution_id": execution_id,
            "started_wall_ns": started_wall_ns,
            "completed_wall_ns": completed_wall_ns,
            "final_pose_check": final_pose_check,
            "message": message,
            "command_inhibited_drops": int(self.executor.counters.command_inhibited_drops),
        }

    def handle(self, request: Any, *, source_ip: str) -> dict[str, Any]:
        request_id = request.get("request_id") if isinstance(request, dict) else None
        try:
            normalized = validate_reset_request(
                request, cfg=self.cfg, source_ip=source_ip, now_ns=self.wall_time_ns()
            )
            request_id = normalized["request_id"]
            with self._lock:
                self._remember_request_id(request_id)
                if normalized["operation"] == RESUME_OPERATION:
                    return self._resume(normalized)
                return self._reset(normalized)
        except ResetProtocolError as exc:
            return self._response(request_id, "rejected", message=str(exc))
        except Exception as exc:
            self.executor.inhibit_commands()
            self.state = ResetState.RESET_FAILED_INHIBITED
            return self._response(request_id, "failed", message=f"internal_error:{type(exc).__name__}:{exc}")

    def _reset(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.state not in {ResetState.ACTIVE, ResetState.RESET_FAILED_INHIBITED}:
            raise ResetProtocolError(f"reset_not_allowed_in_{self.state.value}")
        request_id = request["request_id"]
        choreography_name = request["choreography_name"]
        started_wall_ns = self.wall_time_ns()
        self.executor.inhibit_commands()
        self.state = ResetState.COMMAND_QUIET
        quiet_started = self.monotonic()
        self.wait(float(self.executor.cfg.command_timeout_s))
        remaining = float(self.executor.cfg.command_timeout_s) - (self.monotonic() - quiet_started)
        if remaining > 0:
            self.wait(remaining)
        self.state = ResetState.RESETTING
        self.last_reset_request_id = request_id

        result: ChoreographyResult = self.choreography_adapter.execute_and_wait(choreography_name)
        completed_wall_ns = self.wall_time_ns()
        if not result.started:
            self.state = ResetState.RESET_FAILED_INHIBITED
            return self._response(
                request_id,
                "failed",
                execution_id=result.execution_id,
                started_wall_ns=started_wall_ns,
                completed_wall_ns=completed_wall_ns,
                final_pose_check=result.final_pose_check,
                message=result.status_message or "choreography_not_started",
            )
        if not result.completed:
            self.state = ResetState.RESET_FAILED_INHIBITED
            return self._response(
                request_id,
                "failed",
                execution_id=result.execution_id,
                started_wall_ns=started_wall_ns,
                completed_wall_ns=completed_wall_ns,
                final_pose_check=result.final_pose_check,
                message=result.status_message or "choreography_not_completed",
            )
        if not result.success or not result.final_pose_check.get("passed"):
            self.state = ResetState.RESET_FAILED_INHIBITED
            return self._response(
                request_id,
                "failed",
                execution_id=result.execution_id,
                started_wall_ns=started_wall_ns,
                completed_wall_ns=completed_wall_ns,
                final_pose_check=result.final_pose_check,
                message=result.status_message or "final_pose_check_failed",
            )

        self.state = ResetState.RESET_COMPLETE_INHIBITED
        return self._response(
            request_id,
            "completed",
            execution_id=result.execution_id,
            started_wall_ns=started_wall_ns,
            completed_wall_ns=completed_wall_ns,
            final_pose_check=result.final_pose_check,
            message=result.status_message,
        )

    def _resume(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.state != ResetState.RESET_COMPLETE_INHIBITED:
            raise ResetProtocolError(f"resume_not_allowed_in_{self.state.value}")
        if request["reset_request_id"] != self.last_reset_request_id:
            raise ResetProtocolError("reset_request_id_mismatch")
        self.executor.resume_commands()
        self.state = ResetState.ACTIVE
        return self._response(
            request["request_id"],
            "resumed",
            completed_wall_ns=self.wall_time_ns(),
            message="command acceptance resumed; no cached command was replayed",
        )

    def close(self) -> None:
        self.choreography_adapter.close()

    def request_stop(self) -> None:
        request_stop = getattr(self.choreography_adapter, "request_stop", None)
        if request_stop is not None:
            request_stop()


class ResetControlServer:
    def __init__(self, cfg: ResetControlConfig, coordinator: ResetCoordinator) -> None:
        self.cfg = cfg
        self.coordinator = coordinator
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="jz-pin-reset-control", daemon=False)
        self._socket: socket.socket | None = None
        self._connection: socket.socket | None = None
        self.error: BaseException | None = None

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("reset control server did not become ready")
        if self.error is not None:
            raise RuntimeError(f"reset control server failed: {self.error}")

    def _run(self) -> None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                self._socket = server
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((self.cfg.bind_ip, self.cfg.port))
                server.listen(4)
                server.settimeout(self.cfg.socket_timeout_s)
                self._ready.set()
                while not self._stop.is_set():
                    try:
                        connection, address = server.accept()
                    except socket.timeout:
                        continue
                    self._connection = connection
                    try:
                        with connection:
                            connection.settimeout(self.cfg.choreography_completion_timeout_s + 15.0)
                            response = self._handle_connection(connection, address[0])
                            try:
                                connection.sendall(
                                    json.dumps(response, separators=(",", ":")).encode() + b"\n"
                                )
                            except OSError:
                                pass
                    finally:
                        self._connection = None
        except OSError as exc:
            if not self._stop.is_set():
                self.error = exc
        except BaseException as exc:
            self.error = exc
            self._ready.set()
        finally:
            self._socket = None

    def _handle_connection(self, connection: socket.socket, source_ip: str) -> dict[str, Any]:
        payload = bytearray()
        while len(payload) <= self.cfg.max_request_bytes:
            chunk = connection.recv(min(4096, self.cfg.max_request_bytes + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
            if b"\n" in chunk:
                break
        if len(payload) > self.cfg.max_request_bytes:
            return self.coordinator._response(None, "rejected", message="request_too_large")
        try:
            request = json.loads(bytes(payload).split(b"\n", 1)[0])
        except Exception:
            request = None
        return self.coordinator.handle(request, source_ip=source_ip)

    def close(self) -> None:
        self._stop.set()
        self.coordinator.request_stop()
        connection = self._connection
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        sock = self._socket
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        self._thread.join(timeout=max(7.0, self.cfg.choreography_service_timeout_s + 2.0))
        if self._thread.is_alive():
            raise RuntimeError("reset control server thread did not stop")
        self.coordinator.close()

    @property
    def healthy(self) -> bool:
        return self._thread.is_alive() and self.error is None
