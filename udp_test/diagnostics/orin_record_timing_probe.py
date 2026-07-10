#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
MY_DEVS_ROOT = REPO_ROOT / "my_devs"
for path in (str(SRC_ROOT), str(MY_DEVS_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from my_devs.jz_robot.common import DEFAULT_ROBOT_CONFIG, load_robot_config


LEFT = "left"
RIGHT = "right"
STOP_REQUESTED = False
PROBE_VERSION = 1


def request_stop(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def monotonic_ns() -> int:
    return time.monotonic_ns()


def ros_stamp_ns(msg: Any) -> int | None:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def encode_packet(packet: dict[str, Any]) -> bytes:
    return json.dumps(packet, separators=(",", ":"), sort_keys=True).encode("utf-8")


def import_ros_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise RuntimeError(
            "Missing ROS2 Python dependency "
            f"{missing!r}. Run this Orin-side probe from a ROS2-sourced shell/environment where "
            "`python -c 'import rclpy'` works. For example, source the ROS setup used by the "
            "existing Orin bridge scripts before running this command. This probe is readonly and "
            "does not publish robot commands."
        ) from exc
    return rclpy, SingleThreadedExecutor, JointState, Float64MultiArray


class TimingCollector:
    def __init__(self, robot_cfg: Any) -> None:
        self.robot_cfg = robot_cfg
        self.state_joints = {LEFT: {}, RIGHT: {}}
        self.state_grippers = {LEFT: {}, RIGHT: {}}
        self.action_joints = {LEFT: {}, RIGHT: {}}
        self.action_grippers = {LEFT: {}, RIGHT: {}}
        self.last_update_ns = {
            "state_left_joints": None,
            "state_right_joints": None,
            "state_left_gripper": None,
            "state_right_gripper": None,
            "action_left_joints": None,
            "action_right_joints": None,
            "action_left_gripper": None,
            "action_right_gripper": None,
        }
        self.last_ros_stamp_ns = {key: None for key in self.last_update_ns}
        self.counts = {key: 0 for key in self.last_update_ns}

    def _update_joints(self, prefix: str, side: str, msg: JointState) -> None:
        target = self.state_joints if prefix == "state" else self.action_joints
        target[side].update(
            {name: float(msg.position[idx]) for idx, name in enumerate(msg.name) if idx < len(msg.position)}
        )
        key = f"{prefix}_{side}_joints"
        self.counts[key] += 1
        self.last_update_ns[key] = monotonic_ns()
        self.last_ros_stamp_ns[key] = ros_stamp_ns(msg)

    def _update_gripper(self, prefix: str, side: str, msg: Float64MultiArray) -> None:
        target = self.state_grippers if prefix == "state" else self.action_grippers
        if len(msg.data) > 0:
            target[side]["width"] = float(msg.data[0])
        if len(msg.data) > 1:
            target[side]["force"] = float(msg.data[1])
        key = f"{prefix}_{side}_gripper"
        self.counts[key] += 1
        self.last_update_ns[key] = monotonic_ns()
        self.last_ros_stamp_ns[key] = ros_stamp_ns(msg)

    def update_state_joints(self, side: str, msg: JointState) -> None:
        self._update_joints("state", side, msg)

    def update_state_gripper(self, side: str, msg: Float64MultiArray) -> None:
        self._update_gripper("state", side, msg)

    def update_action_joints(self, side: str, msg: JointState) -> None:
        self._update_joints("action", side, msg)

    def update_action_gripper(self, side: str, msg: Float64MultiArray) -> None:
        self._update_gripper("action", side, msg)

    def state_ready(self) -> bool:
        left_ready = all(name in self.state_joints[LEFT] for name in self.robot_cfg.left_joint_names)
        right_ready = all(name in self.state_joints[RIGHT] for name in self.robot_cfg.right_joint_names)
        if not self.robot_cfg.use_gripper:
            return left_ready and right_ready
        gripper_ready = all(
            field in self.state_grippers[side] for side in (LEFT, RIGHT) for field in ("width", "force")
        )
        return left_ready and right_ready and gripper_ready

    def action_ready(self) -> bool:
        left_ready = all(name in self.action_joints[LEFT] for name in self.robot_cfg.left_joint_names)
        right_ready = all(name in self.action_joints[RIGHT] for name in self.robot_cfg.right_joint_names)
        return left_ready and right_ready

    def packet(self, stream: str, seq: int) -> dict[str, Any]:
        if stream == "state":
            joints = self.state_joints
            grippers = self.state_grippers
        elif stream == "target_action":
            joints = self.action_joints
            grippers = self.action_grippers
        else:
            raise ValueError(f"unsupported stream: {stream}")

        source_prefix = "action" if stream == "target_action" else "state"
        source_keys = [key for key in self.last_update_ns if key.startswith(source_prefix)]
        sample_monotonic_ns = monotonic_ns()
        update_times = [self.last_update_ns[key] for key in source_keys if self.last_update_ns[key] is not None]
        newest_update_ns = max(update_times) if update_times else None
        oldest_update_ns = min(update_times) if update_times else None
        source_age_ms = (
            (sample_monotonic_ns - newest_update_ns) / 1_000_000 if newest_update_ns is not None else None
        )
        source_skew_ms = (
            (newest_update_ns - oldest_update_ns) / 1_000_000
            if newest_update_ns is not None and oldest_update_ns is not None
            else None
        )

        return {
            "version": PROBE_VERSION,
            "type": "jz_record_timing_probe",
            "stream": stream,
            "seq": seq,
            "sample_monotonic_ns": sample_monotonic_ns,
            "wall_time_ns": time.time_ns(),
            "source_age_ms": source_age_ms,
            "source_skew_ms": source_skew_ms,
            "counts": {key: self.counts[key] for key in source_keys},
            "last_update_monotonic_ns": {key: self.last_update_ns[key] for key in source_keys},
            "last_ros_stamp_ns": {key: self.last_ros_stamp_ns[key] for key in source_keys},
            "values": {
                LEFT: {name: joints[LEFT].get(name) for name in self.robot_cfg.left_joint_names},
                RIGHT: {name: joints[RIGHT].get(name) for name in self.robot_cfg.right_joint_names},
                "grippers": {
                    LEFT: grippers[LEFT].copy(),
                    RIGHT: grippers[RIGHT].copy(),
                },
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Readonly Orin-side timing probe. It subscribes to the same ROS state/action topics as the "
            "record bridges and sends diagnostic UDP packets to separate ports."
        )
    )
    parser.add_argument("--robot-config", default=str(DEFAULT_ROBOT_CONFIG))
    parser.add_argument("--target-ip", required=True, help="x86 IP address that runs x86_record_timing_probe.py")
    parser.add_argument("--bind-ip", default="192.168.1.81")
    parser.add_argument("--state-port", type=int, default=39110)
    parser.add_argument("--action-port", type=int, default=39130)
    parser.add_argument("--state-hz", type=float, default=20.0)
    parser.add_argument("--action-hz", type=float, default=30.0)
    parser.add_argument("--wait-timeout-s", type=float, default=10.0)
    parser.add_argument("--print-every", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    args = parse_args()
    try:
        rclpy, SingleThreadedExecutor, JointState, Float64MultiArray = import_ros_dependencies()
    except RuntimeError as exc:
        print(f"[orin_record_timing_probe] {exc}", file=sys.stderr, flush=True)
        return 2

    robot_cfg = load_robot_config(args.robot_config)
    collector = TimingCollector(robot_cfg)

    rclpy.init()
    node = rclpy.create_node("jz_record_timing_probe")
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    node.create_subscription(
        JointState,
        robot_cfg.left_joint_state_topic,
        lambda msg: collector.update_state_joints(LEFT, msg),
        robot_cfg.qos_depth,
    )
    node.create_subscription(
        JointState,
        robot_cfg.right_joint_state_topic,
        lambda msg: collector.update_state_joints(RIGHT, msg),
        robot_cfg.qos_depth,
    )
    node.create_subscription(
        JointState,
        robot_cfg.left_position_command_topic,
        lambda msg: collector.update_action_joints(LEFT, msg),
        robot_cfg.qos_depth,
    )
    node.create_subscription(
        JointState,
        robot_cfg.right_position_command_topic,
        lambda msg: collector.update_action_joints(RIGHT, msg),
        robot_cfg.qos_depth,
    )
    if robot_cfg.use_gripper:
        node.create_subscription(
            Float64MultiArray,
            robot_cfg.left_gripper_state_topic,
            lambda msg: collector.update_state_gripper(LEFT, msg),
            robot_cfg.qos_depth,
        )
        node.create_subscription(
            Float64MultiArray,
            robot_cfg.right_gripper_state_topic,
            lambda msg: collector.update_state_gripper(RIGHT, msg),
            robot_cfg.qos_depth,
        )
        node.create_subscription(
            Float64MultiArray,
            robot_cfg.left_gripper_command_topic,
            lambda msg: collector.update_action_gripper(LEFT, msg),
            robot_cfg.qos_depth,
        )
        node.create_subscription(
            Float64MultiArray,
            robot_cfg.right_gripper_command_topic,
            lambda msg: collector.update_action_gripper(RIGHT, msg),
            robot_cfg.qos_depth,
        )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind_ip, 0))
    state_target = (args.target_ip, args.state_port)
    action_target = (args.target_ip, args.action_port)

    print(
        "[orin_record_timing_probe] READONLY subscribe-only probe. "
        "It does not publish ROS commands and does not move the robot.",
        flush=True,
    )
    print(
        f"[orin_record_timing_probe] local={sock.getsockname()[0]}:{sock.getsockname()[1]} "
        f"state_target={state_target[0]}:{state_target[1]} action_target={action_target[0]}:{action_target[1]}",
        flush=True,
    )

    try:
        deadline = time.monotonic() + args.wait_timeout_s
        while (
            not STOP_REQUESTED
            and (not collector.state_ready() or not collector.action_ready())
            and time.monotonic() < deadline
        ):
            executor.spin_once(timeout_sec=0.05)
        if STOP_REQUESTED:
            return 0
        if not collector.state_ready() or not collector.action_ready():
            print(
                "[orin_record_timing_probe] initial topic timeout "
                f"state_ready={collector.state_ready()} action_ready={collector.action_ready()} "
                f"counts={collector.counts}",
                flush=True,
            )
            return 1

        state_period_s = 1.0 / args.state_hz
        action_period_s = 1.0 / args.action_hz
        next_state_send = time.monotonic()
        next_action_send = time.monotonic()
        state_seq = 0
        action_seq = 0

        while not STOP_REQUESTED:
            executor.spin_once(timeout_sec=0.0)
            now = time.monotonic()
            sent = False
            if now >= next_state_send:
                state_seq += 1
                payload = encode_packet(collector.packet("state", state_seq))
                sock.sendto(payload, state_target)
                next_state_send += state_period_s
                sent = True
            if now >= next_action_send:
                action_seq += 1
                payload = encode_packet(collector.packet("target_action", action_seq))
                sock.sendto(payload, action_target)
                next_action_send += action_period_s
                sent = True
            if sent and args.print_every > 0 and max(state_seq, action_seq) % args.print_every == 0:
                print(
                    "[orin_record_timing_probe] "
                    f"state_seq={state_seq} action_seq={action_seq} counts={collector.counts}",
                    flush=True,
                )
            time.sleep(0.001)
        return 0
    finally:
        sock.close()
        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
