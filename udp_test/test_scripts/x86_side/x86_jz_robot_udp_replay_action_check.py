#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(SRC_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import make_default_robot_action_processor
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.jz_robot_udp import JZRobotUDPConfig  # noqa: F401 - register draccus choice
from lerobot.utils.constants import ACTION

import draccus

DEFAULT_CONFIG = REPO_ROOT / "src/lerobot/configs/robot/jz_robot_udp_three_rtsp.yaml"
DEFAULT_DATASET_ROOT = REPO_ROOT / "tests/outputs/jz_robot_udp_hold_phase3_active_001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a bounded number of dataset actions through JZRobotUDP.")
    parser.add_argument("--robot-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset-repo-id", default="local/jz_robot_udp_hold_phase3_active_001")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--transport", choices=("local", "udp"), default="udp")
    parser.add_argument("--execution", choices=("dry_run", "armed"), default="dry_run")
    parser.add_argument("--command-target-ip", default=None)
    parser.add_argument("--command-target-port", type=int, default=None)
    parser.add_argument("--allowed-sender-ip", default=None)
    parser.add_argument("--skip-cameras", action="store_true")
    parser.add_argument("--print-every", type=int, default=1)
    return parser.parse_args()


def load_robot_config(config_path: str | Path) -> RobotConfig:
    return draccus.parse(config_class=RobotConfig, config_path=Path(config_path), args=[])


def configure_robot(robot_cfg: RobotConfig, args: argparse.Namespace) -> None:
    if hasattr(robot_cfg, "send_action_transport"):
        robot_cfg.send_action_transport = args.transport
    if hasattr(robot_cfg, "send_action_execution"):
        robot_cfg.send_action_execution = args.execution
    if args.command_target_ip is not None and hasattr(robot_cfg, "command_target_ip"):
        robot_cfg.command_target_ip = args.command_target_ip
    if args.command_target_port is not None and hasattr(robot_cfg, "command_target_port"):
        robot_cfg.command_target_port = args.command_target_port
    if args.allowed_sender_ip is not None and hasattr(robot_cfg, "allowed_sender_ip"):
        robot_cfg.allowed_sender_ip = args.allowed_sender_ip
    if args.skip_cameras and hasattr(robot_cfg, "rtsp_cameras"):
        robot_cfg.rtsp_cameras = {}


def scalar_to_float(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError("boolean values are not valid action scalars")
    if hasattr(value, "numel") and hasattr(value, "item"):
        if value.numel() != 1:
            raise TypeError(f"action tensor must contain exactly one value, got {value.numel()}")
        return float(value.item())
    if hasattr(value, "shape") and hasattr(value, "item"):
        if value.shape not in ((), (1,)):
            raise TypeError(f"action array must contain exactly one value, got shape {value.shape}")
        return float(value.item())
    return float(value)


def action_vector_to_dict(action_vector: Sequence[Any], action_names: Sequence[str]) -> dict[str, float]:
    if len(action_vector) != len(action_names):
        raise ValueError(f"action length {len(action_vector)} does not match action names {len(action_names)}")
    return {name: scalar_to_float(value) for name, value in zip(action_names, action_vector, strict=True)}


def validate_args(args: argparse.Namespace) -> None:
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.command_target_port is not None and not 0 < args.command_target_port <= 65535:
        raise ValueError("--command-target-port must be in 1..65535")


def main() -> int:
    args = parse_args()
    validate_args(args)

    robot_cfg = load_robot_config(args.robot_config)
    configure_robot(robot_cfg, args)

    print("JZRobotUDP bounded dataset replay action check", flush=True)
    print(
        "This script replays only the requested frame count. "
        "Robot movement is possible only with --execution armed and an armed Orin executor.",
        flush=True,
    )
    print(
        f"[x86 replay action check] dataset_root={Path(args.dataset_root).resolve()} "
        f"episode={args.episode} start_index={args.start_index} count={args.count} "
        f"execution={args.execution} transport={args.transport}",
        flush=True,
    )

    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root, episodes=[args.episode])
    episode_frames = dataset.hf_dataset.filter(lambda row: row["episode_index"] == args.episode)
    action_names = dataset.features[ACTION]["names"]
    end_index = args.start_index + args.count
    if end_index > len(episode_frames):
        raise ValueError(f"requested frames [{args.start_index}, {end_index}) exceed episode length {len(episode_frames)}")

    robot = make_robot_from_config(robot_cfg)
    robot_action_processor = make_default_robot_action_processor()
    period_s = 1.0 / args.fps
    sent = 0

    try:
        robot.connect()
        for idx in range(args.start_index, end_index):
            started = time.perf_counter()
            action = action_vector_to_dict(episode_frames[idx][ACTION], action_names)
            observation = robot.get_observation()
            processed_action = robot_action_processor((action, observation))
            returned = robot.send_action(processed_action)
            sent += 1
            if sent == 1 or (args.print_every > 0 and sent % args.print_every == 0):
                print(
                    "[x86 replay action check] sent "
                    f"frame_index={idx} seq_count={sent} execution={args.execution} "
                    f"action_keys={len(action)} returned_keys={len(returned)}",
                    flush=True,
                )
            elapsed_s = time.perf_counter() - started
            time.sleep(max(period_s - elapsed_s, 0.0))
    finally:
        if robot.is_connected:
            robot.disconnect()

    print(f"SUMMARY: PASS replayed_actions={sent}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
