#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
MY_DEVS_ROOT = REPO_ROOT / "my_devs"
for path in (str(SRC_ROOT), str(MY_DEVS_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import rclpy
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from lerobot.robots.jz_robot_udp.protocol import (
    encode_target_action_packet,
    make_jz_robot_udp_target_action_packet,
)
from my_devs.jz_robot.common import DEFAULT_ROBOT_CONFIG, load_robot_config


LEFT = "left"
RIGHT = "right"
STOP_REQUESTED = False


def _canonicalize_joint_name(name: str) -> str:
    return re.sub(r"_(?=\d+$)", "", name)


def request_stop(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


class ReadonlyTargetActionCollector:
    def __init__(self, robot_cfg: Any):
        self.robot_cfg = robot_cfg
        self.joints = {LEFT: {}, RIGHT: {}}
        self.grippers = {LEFT: {}, RIGHT: {}}
        self.counts = {
            "left_joints": 0,
            "right_joints": 0,
            "left_gripper": 0,
            "right_gripper": 0,
        }

    def update_joints(self, side: str, msg: JointState) -> None:
        required = (
            self.robot_cfg.left_joint_names
            if side == LEFT
            else self.robot_cfg.right_joint_names
        )
        aliases = {_canonicalize_joint_name(joint): joint for joint in required}
        joints = {}
        for idx, name in enumerate(msg.name):
            if idx >= len(msg.position):
                continue
            canonical_name = aliases.get(_canonicalize_joint_name(name))
            if canonical_name is None:
                continue
            joints[canonical_name] = float(msg.position[idx])
        self.joints[side].update(joints)
        self.counts[f"{side}_joints"] += 1

    def update_gripper(self, side: str, msg: Float64MultiArray) -> None:
        state = {}
        if len(msg.data) > 0:
            state["width"] = float(msg.data[0])
        if len(msg.data) > 1:
            state["force"] = float(msg.data[1])
        self.grippers[side].update(state)
        self.counts[f"{side}_gripper"] += 1

    def ready(self) -> bool:
        left_ready = all(name in self.joints[LEFT] for name in self.robot_cfg.left_joint_names)
        right_ready = all(name in self.joints[RIGHT] for name in self.robot_cfg.right_joint_names)
        return left_ready and right_ready

    def _gripper_action(self, side: str) -> dict[str, float]:
        return {
            "width": float(self.grippers[side].get("width", 0.0)),
            "force": float(self.grippers[side].get("force", 0.0)),
        }

    def packet(self, seq: int, robot_name: str) -> dict[str, Any]:
        grippers = {
            LEFT: self._gripper_action(LEFT),
            RIGHT: self._gripper_action(RIGHT),
        }
        if not self.robot_cfg.use_gripper:
            grippers = {
                LEFT: {"width": 0.0, "force": 0.0},
                RIGHT: {"width": 0.0, "force": 0.0},
            }

        return make_jz_robot_udp_target_action_packet(
            robot=robot_name,
            seq=seq,
            stamp_ns=time.time_ns(),
            actions={
                LEFT: {name: self.joints[LEFT][name] for name in self.robot_cfg.left_joint_names},
                RIGHT: {name: self.joints[RIGHT][name] for name in self.robot_cfg.right_joint_names},
                "grippers": grippers,
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Readonly ROS2 target-action to UDP bridge for JZRobot.")
    parser.add_argument("--robot-config", default=str(DEFAULT_ROBOT_CONFIG))
    parser.add_argument("--robot-name", default="robot1")
    parser.add_argument("--target-ip", required=True)
    parser.add_argument("--target-port", type=int, default=39030)
    parser.add_argument("--bind-ip", default="192.168.1.81")
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--count", type=int, default=0, help="0 means run forever.")
    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument("--wait-timeout-s", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    args = parse_args()
    robot_cfg = load_robot_config(args.robot_config)
    collector = ReadonlyTargetActionCollector(robot_cfg)

    rclpy.init()
    node = rclpy.create_node("jz_readonly_ros_target_action_udp_bridge")
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    node.create_subscription(
        JointState,
        robot_cfg.left_position_command_topic,
        lambda msg: collector.update_joints(LEFT, msg),
        robot_cfg.qos_depth,
    )
    node.create_subscription(
        JointState,
        robot_cfg.right_position_command_topic,
        lambda msg: collector.update_joints(RIGHT, msg),
        robot_cfg.qos_depth,
    )
    if robot_cfg.use_gripper:
        node.create_subscription(
            Float64MultiArray,
            robot_cfg.left_gripper_command_topic,
            lambda msg: collector.update_gripper(LEFT, msg),
            robot_cfg.qos_depth,
        )
        node.create_subscription(
            Float64MultiArray,
            robot_cfg.right_gripper_command_topic,
            lambda msg: collector.update_gripper(RIGHT, msg),
            robot_cfg.qos_depth,
        )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind_ip, 0))
    target = (args.target_ip, args.target_port)
    print(
        "[orin ros target action udp bridge] READONLY subscribe-only bridge. "
        "It forwards ROS target command topics as LeRobot actions and does not publish commands.",
        flush=True,
    )
    print(
        f"[orin ros target action udp bridge] local={sock.getsockname()[0]}:{sock.getsockname()[1]} "
        f"target={target[0]}:{target[1]} hz={args.hz}",
        flush=True,
    )

    try:
        deadline = time.monotonic() + args.wait_timeout_s
        while not STOP_REQUESTED and not collector.ready() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
        if STOP_REQUESTED:
            print(
                "[orin ros target action udp bridge] stop requested before initial target action ready",
                flush=True,
            )
            return 0
        if not collector.ready():
            print(
                "[orin ros target action udp bridge] initial target action timeout "
                f"counts={collector.counts}",
                flush=True,
            )
            return 1

        seq = 0
        period_s = 1.0 / args.hz
        next_send = time.monotonic()
        while not STOP_REQUESTED and (args.count <= 0 or seq < args.count):
            executor.spin_once(timeout_sec=0.0)
            now = time.monotonic()
            if now < next_send:
                time.sleep(min(next_send - now, 0.01))
                continue
            seq += 1
            payload = encode_target_action_packet(collector.packet(seq=seq, robot_name=args.robot_name))
            sock.sendto(payload, target)
            if seq == 1 or (args.print_every > 0 and seq % args.print_every == 0):
                print(
                    "[orin ros target action udp bridge] "
                    f"sent seq={seq} bytes={len(payload)} counts={collector.counts}",
                    flush=True,
                )
            next_send += period_s
        return 0
    finally:
        sock.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
