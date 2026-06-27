#!/usr/bin/env python3

from __future__ import annotations

import argparse
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

from lerobot.robots.jz_robot_udp.protocol import PROTOCOL_VERSION, STATE_MESSAGE_TYPE, encode_state_packet
from my_devs.jz_robot.common import DEFAULT_ROBOT_CONFIG, load_robot_config


LEFT = "left"
RIGHT = "right"


class ReadonlyStateCollector:
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
        self.joints[side].update(
            {name: float(msg.position[idx]) for idx, name in enumerate(msg.name) if idx < len(msg.position)}
        )
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
        gripper_ready = True
        if self.robot_cfg.use_gripper:
            gripper_ready = all(field in self.grippers[side] for side in (LEFT, RIGHT) for field in ("width", "force"))
        return left_ready and right_ready and gripper_ready

    def packet(self, seq: int, robot_name: str) -> dict[str, Any]:
        return {
            "version": PROTOCOL_VERSION,
            "type": STATE_MESSAGE_TYPE,
            "robot": robot_name,
            "seq": seq,
            "stamp_ns": time.time_ns(),
            "joints": {
                LEFT: {name: self.joints[LEFT][name] for name in self.robot_cfg.left_joint_names},
                RIGHT: {name: self.joints[RIGHT][name] for name in self.robot_cfg.right_joint_names},
            },
            "grippers": {
                LEFT: self.grippers[LEFT].copy(),
                RIGHT: self.grippers[RIGHT].copy(),
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Readonly ROS2 state to UDP bridge for JZRobot.")
    parser.add_argument("--robot-config", default=str(DEFAULT_ROBOT_CONFIG))
    parser.add_argument("--robot-name", default="robot1")
    parser.add_argument("--target-ip", required=True)
    parser.add_argument("--target-port", type=int, default=39010)
    parser.add_argument("--bind-ip", default="192.168.1.81")
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--count", type=int, default=0, help="0 means run forever.")
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--wait-timeout-s", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    robot_cfg = load_robot_config(args.robot_config)
    collector = ReadonlyStateCollector(robot_cfg)

    rclpy.init()
    node = rclpy.create_node("jz_readonly_ros_state_udp_bridge")
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    node.create_subscription(
        JointState,
        robot_cfg.left_joint_state_topic,
        lambda msg: collector.update_joints(LEFT, msg),
        robot_cfg.qos_depth,
    )
    node.create_subscription(
        JointState,
        robot_cfg.right_joint_state_topic,
        lambda msg: collector.update_joints(RIGHT, msg),
        robot_cfg.qos_depth,
    )
    if robot_cfg.use_gripper:
        node.create_subscription(
            Float64MultiArray,
            robot_cfg.left_gripper_state_topic,
            lambda msg: collector.update_gripper(LEFT, msg),
            robot_cfg.qos_depth,
        )
        node.create_subscription(
            Float64MultiArray,
            robot_cfg.right_gripper_state_topic,
            lambda msg: collector.update_gripper(RIGHT, msg),
            robot_cfg.qos_depth,
        )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind_ip, 0))
    target = (args.target_ip, args.target_port)
    print(
        "[orin ros state udp bridge] READONLY subscribe-only bridge. "
        "It does not publish command topics and does not call send_action.",
        flush=True,
    )
    print(
        f"[orin ros state udp bridge] local={sock.getsockname()[0]}:{sock.getsockname()[1]} "
        f"target={target[0]}:{target[1]} hz={args.hz}",
        flush=True,
    )

    try:
        deadline = time.monotonic() + args.wait_timeout_s
        while not collector.ready() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
        if not collector.ready():
            print(f"[orin ros state udp bridge] initial state timeout counts={collector.counts}", flush=True)
            return 1

        seq = 0
        period_s = 1.0 / args.hz
        next_send = time.monotonic()
        while args.count <= 0 or seq < args.count:
            executor.spin_once(timeout_sec=0.0)
            now = time.monotonic()
            if now < next_send:
                time.sleep(min(next_send - now, 0.01))
                continue
            seq += 1
            payload = encode_state_packet(collector.packet(seq=seq, robot_name=args.robot_name))
            sock.sendto(payload, target)
            if seq == 1 or (args.print_every > 0 and seq % args.print_every == 0):
                print(
                    f"[orin ros state udp bridge] sent seq={seq} bytes={len(payload)} counts={collector.counts}",
                    flush=True,
                )
            next_send += period_s
        return 0
    finally:
        sock.close()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
