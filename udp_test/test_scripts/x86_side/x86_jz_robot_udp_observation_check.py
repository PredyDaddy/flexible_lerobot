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
    parser = argparse.ArgumentParser(description="Readonly x86 observation check for JZRobotUDP.")
    parser.add_argument("--robot-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--skip-cameras", action="store_true", help="Only check UDP state observation.")
    parser.add_argument("--allowed-sender-ip", default=None, help="Override allowed Orin UDP sender IP.")
    return parser.parse_args()


def load_robot_config(config_path: str | Path) -> RobotConfig:
    return draccus.parse(config_class=RobotConfig, config_path=Path(config_path), args=[])


def main() -> int:
    args = parse_args()
    robot_cfg = load_robot_config(args.robot_config)
    if args.skip_cameras and hasattr(robot_cfg, "rtsp_cameras"):
        robot_cfg.rtsp_cameras = {}
    if args.allowed_sender_ip is not None and hasattr(robot_cfg, "allowed_sender_ip"):
        robot_cfg.allowed_sender_ip = args.allowed_sender_ip
    robot = make_robot_from_config(robot_cfg)

    print("[x86 jz_robot_udp observation check] READONLY ONLY", flush=True)
    print("[x86 jz_robot_udp observation check] This script only calls get_observation().", flush=True)
    print(f"[x86 jz_robot_udp observation check] config={Path(args.robot_config).resolve()}", flush=True)

    period_s = 1.0 / args.hz
    try:
        robot.connect()
        for idx in range(1, args.count + 1):
            started = time.monotonic()
            obs = robot.get_observation()
            if idx == 1 or (args.print_every > 0 and idx % args.print_every == 0):
                image_shapes = {key: tuple(value.shape) for key, value in obs.items() if hasattr(value, "shape")}
                numeric_count = sum(1 for value in obs.values() if isinstance(value, int | float))
                print(
                    f"[x86 jz_robot_udp observation check] obs={idx} keys={len(obs)} "
                    f"numeric={numeric_count} image_shapes={image_shapes}",
                    flush=True,
                )
            elapsed_s = time.monotonic() - started
            time.sleep(max(period_s - elapsed_s, 0.0))
        print(f"SUMMARY: PASS observations={args.count}", flush=True)
        return 0
    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
