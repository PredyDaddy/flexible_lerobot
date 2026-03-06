#!/usr/bin/env python

"""Run pure ACT policy inference on SO follower robot (no dataset recording).

This script performs:
1) read robot observation
2) policy inference
3) send action to robot
in a real-time loop.

Compared with the GROOT/PI inference helpers, this ACT version keeps the
implementation intentionally close to the repository's standard inference path:
- load the exact saved pre/post processors from checkpoint
- reuse `predict_action` from `lerobot.utils.control_utils`
- reuse robot observation/action processors for feature alignment

Example:
python my_devs/train/act/so101/run_act_infer.py \
    --robot-port /dev/ttyACM0 \
    --top-cam-index 4 \
    --wrist-cam-index 6 \
    --task "Put the block in the bin" \
    --run-time-s 120

ACT deployment note:
- The current checkpoint was trained with `chunk_size=100` and
  `n_action_steps=100`. That means the policy can infer one 100-step action
  chunk and then consume it from an internal queue. If you want more frequent
  replanning on the robot, set `--policy-n-action-steps` to a smaller value
  (for example 8/16/32) during deployment.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from lerobot import policies  # noqa: F401  # Register policy config classes.
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.policies.factory import get_policy_class
from lerobot.policies.utils import make_robot_action
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, make_default_processors
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import auto_select_torch_device, get_safe_torch_device

DEFAULT_POLICY_PATH = (
    "/data/cqy_workspace/flexible_lerobot/outputs/train/"
    "20260305_190147_act_grasp_block_in_bin1_e15/"
    "checkpoints/last/pretrained_model"
)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pure ACT inference loop for SO101/SO100 follower robot via LeRobot APIs."
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
        "--task",
        default=os.getenv("DATASET_TASK", "Put the block in the bin"),
        help="Language instruction passed to policy inference.",
    )
    parser.add_argument(
        "--run-time-s",
        type=float,
        default=float(os.getenv("RUN_TIME_S", "0")),
        help="Total inference duration in seconds. <=0 means run until Ctrl+C.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=int(os.getenv("LOG_INTERVAL", "30")),
        help="Print status every N steps.",
    )
    parser.add_argument(
        "--dry-run",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("DRY_RUN", False),
        help="Print resolved config and exit without connecting robot or loading model weights.",
    )
    return parser


def load_pre_post_processors(
    policy_path: Path,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Load saved policy processors directly from checkpoint directory."""
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(policy_path),
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(policy_path),
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


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

    if args.robot_type not in {"so100_follower", "so101_follower"}:
        raise ValueError(
            f"Unsupported robot_type={args.robot_type!r}. "
            "This script currently supports so100_follower/so101_follower."
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

    print(f"[INFO] Robot id: {args.robot_id}")
    print(f"[INFO] Robot type (requested): {args.robot_type}")
    print(f"[INFO] Robot port: {args.robot_port}")
    print(f"[INFO] Policy path: {policy_path}")
    print(f"[INFO] Policy type: {policy_cfg.type}")
    print(f"[INFO] Policy device: {policy_cfg.device}")
    print(f"[INFO] Task: {args.task}")
    print(f"[INFO] FPS: {args.fps}")
    print(f"[INFO] run_time_s: {args.run_time_s} (<=0 means until Ctrl+C)")
    print(
        "[INFO] ACT runtime config: "
        f"chunk_size={policy_cfg.chunk_size}, "
        f"n_action_steps={policy_cfg.n_action_steps}, "
        f"temporal_ensemble_coeff={policy_cfg.temporal_ensemble_coeff}"
    )

    if args.dry_run:
        print("[INFO] DRY_RUN=true, exit without execution.")
        return

    from lerobot.robots import make_robot_from_config

    robot = make_robot_from_config(robot_cfg)

    policy_class = get_policy_class(policy_cfg.type)
    policy = policy_class.from_pretrained(str(policy_path), config=policy_cfg, strict=False)
    policy.config.device = policy_cfg.device
    policy.to(policy_cfg.device)

    preprocessor, postprocessor = load_pre_post_processors(policy_path)

    _, robot_action_processor, robot_observation_processor = make_default_processors()
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=robot_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    step = 0
    start_t = time.perf_counter()
    end_t = start_t + args.run_time_s if args.run_time_s > 0 else None

    try:
        robot.connect()
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

        while True:
            if end_t is not None and time.perf_counter() >= end_t:
                print("[INFO] Reached requested run_time_s. Exiting inference loop.")
                break

            loop_t = time.perf_counter()

            obs = robot.get_observation()
            obs_processed = robot_observation_processor(obs)
            observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)

            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=args.task,
                robot_type=robot.robot_type,
            )
            action_dict = make_robot_action(action_values, dataset_features)
            robot_action_to_send = robot_action_processor((action_dict, obs))
            robot.send_action(robot_action_to_send)

            step += 1
            if args.log_interval > 0 and step % args.log_interval == 0:
                elapsed = time.perf_counter() - start_t
                print(f"[INFO] Step {step} | elapsed={elapsed:.2f}s")

            dt_s = time.perf_counter() - loop_t
            precise_sleep(max(1 / args.fps - dt_s, 0.0))
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received. Stopping inference.")
    finally:
        if robot.is_connected:
            robot.disconnect()
        print("[INFO] Inference finished.")


if __name__ == "__main__":
    main()
