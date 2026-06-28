#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(SRC_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import draccus

from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.jz_robot_udp import JZRobotUDPConfig  # noqa: F401 - register draccus choice

DEFAULT_CONFIG = REPO_ROOT / "src/lerobot/configs/robot/jz_robot_udp_three_rtsp.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 x86 send_action dry-run check for JZRobotUDP.")
    parser.add_argument("--robot-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--transport", choices=("local", "udp"), default="local")
    parser.add_argument("--execution", choices=("dry_run",), default="dry_run")
    parser.add_argument("--command-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-cameras", action="store_true")
    parser.add_argument("--command-target-ip", default=None)
    parser.add_argument("--command-target-port", type=int, default=None)
    parser.add_argument("--allowed-sender-ip", default=None)
    parser.add_argument("--print-every", type=int, default=1)
    return parser.parse_args()


def load_robot_config(config_path: str | Path) -> RobotConfig:
    return draccus.parse(config_class=RobotConfig, config_path=Path(config_path), args=[])


def set_if_present(obj: object, name: str, value: object) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


def configure_phase2(robot_cfg: RobotConfig, args: argparse.Namespace) -> None:
    set_if_present(robot_cfg, "send_action_transport", args.transport)
    set_if_present(robot_cfg, "send_action_execution", args.execution)
    if args.command_target_ip is not None:
        set_if_present(robot_cfg, "command_target_ip", args.command_target_ip)
    if args.command_target_port is not None:
        set_if_present(robot_cfg, "command_target_port", args.command_target_port)
    if args.allowed_sender_ip is not None:
        set_if_present(robot_cfg, "allowed_sender_ip", args.allowed_sender_ip)
    if args.skip_cameras and hasattr(robot_cfg, "rtsp_cameras"):
        robot_cfg.rtsp_cameras = {}


def make_zero_action(action_features: dict[str, object]) -> dict[str, float]:
    if not action_features:
        raise RuntimeError(
            "JZRobotUDP.action_features is empty. The core Phase 2 send_action worker likely has not landed yet."
        )
    return {key: 0.0 for key in action_features}


def mark_connected_for_command_only(robot: object) -> None:
    if hasattr(robot, "_is_connected"):
        # Command-only mode intentionally bypasses state/RTSP connection so the check exercises send_action only.
        setattr(robot, "_is_connected", True)


def clear_command_only_connection_flag(robot: object) -> None:
    if hasattr(robot, "_is_connected"):
        setattr(robot, "_is_connected", False)


def print_startup(args: argparse.Namespace) -> None:
    print("PHASE2 COMMAND DRY-RUN ONLY", flush=True)
    print("This script does not enable robot execution.", flush=True)
    print(
        "[x86 send_action check] "
        f"config={Path(args.robot_config).resolve()} transport={args.transport} "
        f"execution={args.execution} command_only={args.command_only}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    if args.count < 0:
        raise ValueError("--count must be non-negative")
    if args.hz <= 0:
        raise ValueError("--hz must be positive")
    if args.command_target_port is not None and not 0 < args.command_target_port <= 65535:
        raise ValueError("--command-target-port must be in 1..65535")

    robot_cfg = load_robot_config(args.robot_config)
    configure_phase2(robot_cfg, args)
    robot = make_robot_from_config(robot_cfg)

    print_startup(args)
    period_s = 1.0 / args.hz
    commands = 0

    try:
        if args.command_only:
            mark_connected_for_command_only(robot)
        else:
            robot.connect()

        action = make_zero_action(robot.action_features)
        for idx in range(1, args.count + 1):
            started = time.monotonic()
            returned = robot.send_action(dict(action))
            commands += 1
            if idx == 1 or (args.print_every > 0 and idx % args.print_every == 0):
                print(
                    "[x86 send_action check] DRY_RUN sent "
                    f"idx={idx} action_keys={len(action)} returned_keys={len(returned)}",
                    flush=True,
                )
            elapsed_s = time.monotonic() - started
            time.sleep(max(period_s - elapsed_s, 0.0))
    finally:
        if getattr(robot, "is_connected", False):
            if args.command_only:
                command_sender = getattr(robot, "_command_sender", None)
                if command_sender is not None:
                    command_sender.close()
                    setattr(robot, "_command_sender", None)
                clear_command_only_connection_flag(robot)
            else:
                robot.disconnect()

    print(f"SUMMARY: PASS commands={commands}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
