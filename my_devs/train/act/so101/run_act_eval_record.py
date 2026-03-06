#!/usr/bin/env python

"""Run SO101 ACT policy-driven evaluation recording via LeRobot Python APIs.

This script is the Python/API equivalent of `run_act_eval_record.sh`.
It builds a `RecordConfig` in code and calls `lerobot.scripts.lerobot_record.record`.

Compared with the shell wrapper, this version exposes ACT-specific deployment
overrides directly in Python so you can safely adjust `n_action_steps` or
enable temporal ensembling during real-robot evaluation.
"""

from __future__ import annotations

import argparse
import os
import shutil
from glob import glob
from pathlib import Path

from lerobot import policies  # noqa: F401  # Ensures policy configs are registered
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig, record
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import auto_select_torch_device

DEFAULT_POLICY_PATH = (
    "/data/cqy_workspace/flexible_lerobot/outputs/train/"
    "20260305_190147_act_grasp_block_in_bin1_e15/"
    "checkpoints/last/pretrained_model"
)
DEFAULT_CACHE_GLOB = "/home/cqy/.cache/huggingface/lerobot/admin123/eval_act_*"


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else parse_bool(raw)


def maybe_path(path_str: str | None) -> Path | None:
    return None if not path_str else Path(path_str).expanduser()


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"", "none", "null"}:
        return None
    return float(value)


def parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"", "none", "null", "0"}:
        return None
    return int(value)


def resolve_policy_device(requested: str | None, config_device: str | None) -> str:
    if requested is not None and requested.strip().lower() in {"", "none", "null"}:
        requested = None
    if config_device is None or str(config_device).lower() in {"", "none", "null"}:
        config_device = auto_select_torch_device().type
    if requested is None:
        return str(config_device)
    if requested == "auto":
        return auto_select_torch_device().type
    return requested


def cleanup_eval_cache(pattern: str) -> list[Path]:
    removed: list[Path] = []
    for item in sorted(glob(pattern)):
        path = Path(item)
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(path)
        elif path.exists():
            path.unlink()
            removed.append(path)
    return removed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record ACT policy-driven SO101 evaluation dataset using LeRobot Python APIs."
    )

    parser.add_argument("--robot-id", default=os.getenv("ROBOT_ID", "my_so101"))
    parser.add_argument("--robot-type", default=os.getenv("ROBOT_TYPE", "so101_follower"))
    parser.add_argument(
        "--calib-dir",
        default=os.getenv(
            "CALIB_DIR", "/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower"
        ),
    )
    parser.add_argument("--robot-port", default=os.getenv("ROBOT_PORT", "/dev/ttyACM0"))

    parser.add_argument("--top-cam-index", type=int, default=int(os.getenv("TOP_CAM_INDEX", "4")))
    parser.add_argument("--wrist-cam-index", type=int, default=int(os.getenv("WRIST_CAM_INDEX", "6")))
    parser.add_argument("--img-width", type=int, default=int(os.getenv("IMG_WIDTH", "640")))
    parser.add_argument("--img-height", type=int, default=int(os.getenv("IMG_HEIGHT", "480")))
    parser.add_argument("--fps", type=int, default=int(os.getenv("FPS", "30")))

    parser.add_argument("--policy-path", default=os.getenv("POLICY_PATH", DEFAULT_POLICY_PATH))
    parser.add_argument(
        "--policy-device",
        default=os.getenv("POLICY_DEVICE_OVERRIDE"),
        help="Override checkpoint device with one of cpu/cuda/mps/xpu/auto.",
    )
    parser.add_argument(
        "--policy-n-action-steps",
        type=int,
        default=parse_optional_int(os.getenv("POLICY_N_ACTION_STEPS")),
        help="Optional ACT deployment override. Must be in [1, chunk_size].",
    )
    parser.add_argument(
        "--policy-temporal-ensemble-coeff",
        type=parse_optional_float,
        default=parse_optional_float(os.getenv("POLICY_TEMPORAL_ENSEMBLE_COEFF")),
        help="Optional ACT deployment override. If set, requires policy-n-action-steps=1.",
    )

    parser.add_argument(
        "--dataset-repo-id",
        default=os.getenv("DATASET_REPO_ID", "admin123/eval_act_grasp_block_in_bin1_01"),
    )
    parser.add_argument("--dataset-task", default=os.getenv("DATASET_TASK", "Put the block in the bin"))
    parser.add_argument("--dataset-root", default=os.getenv("DATASET_ROOT"))
    parser.add_argument("--num-episodes", type=int, default=int(os.getenv("NUM_EPISODES", "5")))
    parser.add_argument("--episode-time-s", type=float, default=float(os.getenv("EPISODE_TIME_S", "40")))
    parser.add_argument("--reset-time-s", type=float, default=float(os.getenv("RESET_TIME_S", "10")))
    parser.add_argument(
        "--display-data",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("DISPLAY_DATA", False),
    )
    parser.add_argument(
        "--push-to-hub",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("PUSH_TO_HUB", False),
    )
    parser.add_argument(
        "--play-sounds",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("PLAY_SOUNDS", True),
    )

    parser.add_argument(
        "--cleanup-eval-cache",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("CLEANUP_EVAL_CACHE", False),
        help="If true, remove old cached eval datasets before recording.",
    )
    parser.add_argument(
        "--eval-cache-pattern",
        default=os.getenv("EVAL_CACHE_PATTERN", DEFAULT_CACHE_GLOB),
        help="Glob pattern used when cleanup-eval-cache is true.",
    )

    parser.add_argument(
        "--dry-run",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("DRY_RUN", False),
        help="If true, print resolved config and exit without recording.",
    )
    return parser


def apply_act_runtime_overrides(
    policy_cfg: PreTrainedConfig,
    policy_n_action_steps: int | None,
    policy_temporal_ensemble_coeff: float | None,
) -> None:
    if policy_cfg.type != "act":
        raise ValueError(f"Expected ACT policy, got {policy_cfg.type!r}")

    chunk_size = int(policy_cfg.chunk_size)

    if policy_n_action_steps is not None:
        if not 1 <= policy_n_action_steps <= chunk_size:
            raise ValueError(
                f"--policy-n-action-steps must be within [1, {chunk_size}], got {policy_n_action_steps}"
            )
        policy_cfg.n_action_steps = policy_n_action_steps

    if policy_temporal_ensemble_coeff is not None:
        if policy_cfg.n_action_steps != 1:
            raise ValueError(
                "ACT temporal ensembling requires n_action_steps == 1. "
                f"Current value: {policy_cfg.n_action_steps}"
            )
        policy_cfg.temporal_ensemble_coeff = policy_temporal_ensemble_coeff


def main() -> None:
    register_third_party_plugins()
    args = build_parser().parse_args()

    policy_path = Path(args.policy_path).expanduser()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"Policy path does not exist: {policy_path}")

    if args.cleanup_eval_cache:
        removed = cleanup_eval_cache(args.eval_cache_pattern)
        print(f"[INFO] Cleanup enabled. Removed {len(removed)} paths matching {args.eval_cache_pattern}")

    if args.robot_type not in {"so100_follower", "so101_follower"}:
        raise ValueError(
            f"Unsupported robot_type={args.robot_type!r}. "
            "This API script currently supports so100_follower/so101_follower."
        )

    cameras = {
        "top": OpenCVCameraConfig(
            index_or_path=args.top_cam_index,
            width=args.img_width,
            height=args.img_height,
            fps=args.fps,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=args.wrist_cam_index,
            width=args.img_width,
            height=args.img_height,
            fps=args.fps,
        ),
    }

    robot_cfg = SOFollowerRobotConfig(
        id=args.robot_id,
        calibration_dir=maybe_path(args.calib_dir),
        port=args.robot_port,
        cameras=cameras,
    )

    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = resolve_policy_device(args.policy_device, policy_cfg.device)
    apply_act_runtime_overrides(
        policy_cfg=policy_cfg,
        policy_n_action_steps=args.policy_n_action_steps,
        policy_temporal_ensemble_coeff=args.policy_temporal_ensemble_coeff,
    )

    dataset_cfg = DatasetRecordConfig(
        repo_id=args.dataset_repo_id,
        single_task=args.dataset_task,
        root=maybe_path(args.dataset_root),
        fps=args.fps,
        episode_time_s=args.episode_time_s,
        reset_time_s=args.reset_time_s,
        num_episodes=args.num_episodes,
        push_to_hub=args.push_to_hub,
    )

    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        policy=policy_cfg,
        display_data=args.display_data,
        play_sounds=args.play_sounds,
    )

    print(f"[INFO] Robot id: {args.robot_id}")
    print(f"[INFO] Robot type (requested): {args.robot_type}")
    print(f"[INFO] Robot port: {args.robot_port}")
    print(f"[INFO] Policy path: {policy_path}")
    print(f"[INFO] Policy type: {policy_cfg.type}")
    print(f"[INFO] Policy device: {policy_cfg.device}")
    print(
        "[INFO] ACT runtime config: "
        f"chunk_size={policy_cfg.chunk_size}, "
        f"n_action_steps={policy_cfg.n_action_steps}, "
        f"temporal_ensemble_coeff={policy_cfg.temporal_ensemble_coeff}"
    )
    print(f"[INFO] Dataset repo_id: {args.dataset_repo_id}")
    print(
        f"[INFO] Episodes: {args.num_episodes}, "
        f"episode_time_s: {args.episode_time_s}, reset_time_s: {args.reset_time_s}"
    )
    print(
        "[INFO] Cameras: "
        f"top(index={args.top_cam_index}, {args.img_width}x{args.img_height}@{args.fps}), "
        f"wrist(index={args.wrist_cam_index}, {args.img_width}x{args.img_height}@{args.fps})"
    )

    if args.dry_run:
        print("[INFO] DRY_RUN=true, exit without execution.")
        return

    record(cfg)


if __name__ == "__main__":
    main()
