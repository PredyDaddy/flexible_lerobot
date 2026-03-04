#!/usr/bin/env python

"""Run pure policy inference on SO follower robot (no dataset recording).

This script performs:
1) read robot observation
2) policy inference
3) send action to robot
in a real-time loop.

  python my_devs/train/groot/run_groot_infer.py \
    --robot-port /dev/ttyACM0 \
    --top-cam-index 4 \
    --wrist-cam-index 6 \
    --task "Put the block in the bin" \
    --run-time-s 120
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
from lerobot.utils.utils import get_safe_torch_device

DEFAULT_POLICY_PATH = (
    "/data/cqy_workspace/flexible_lerobot/outputs/train/"
    "groot_grasp_block_in_bin1_repro_20260302_223413/bs32_20260302_223447/"
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pure model inference loop for SO101/SO100 follower robot via LeRobot APIs."
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

    parser.add_argument(
        "--backend",
        default=os.getenv("INFER_BACKEND", "pytorch"),
        choices=["pytorch", "tensorrt"],
        help="Inference backend. 'tensorrt' requires pre-built .engine files.",
    )
    parser.add_argument(
        "--trt-engine-path",
        default=os.getenv("TRT_ENGINE_PATH"),
        help="Directory containing GROOT TensorRT engine files (gr00t_engine/*).",
    )
    parser.add_argument(
        "--vit-dtype",
        default=os.getenv("TRT_VIT_DTYPE", "fp16"),
        choices=["fp16", "fp8"],
        help="TensorRT engine precision suffix for ViT engine (vit_<dtype>.engine).",
    )
    parser.add_argument(
        "--llm-dtype",
        default=os.getenv("TRT_LLM_DTYPE", "fp16"),
        choices=["fp16", "nvfp4", "fp8", "nvfp4_full"],
        help="TensorRT engine precision suffix for LLM engine (llm_<dtype>.engine).",
    )
    parser.add_argument(
        "--dit-dtype",
        default=os.getenv("TRT_DIT_DTYPE", "fp16"),
        choices=["fp16", "fp8"],
        help="TensorRT engine precision suffix for DiT engine (DiT_<dtype>.engine).",
    )
    parser.add_argument(
        "--trt-action-head-only",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("TRT_ACTION_HEAD_ONLY", False),
        help="If true, keep backbone in PyTorch and only run action head with TensorRT engines.",
    )
    return parser


def load_pre_post_processors(
    policy_path: Path,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Load saved policy processors directly from checkpoint directory.

    We intentionally load these from files to preserve the exact normalization and
    transform pipeline used by the trained checkpoint.
    """
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

    print(f"[INFO] Robot id: {args.robot_id}")
    print(f"[INFO] Robot type (requested): {args.robot_type}")
    print(f"[INFO] Robot port: {args.robot_port}")
    print(f"[INFO] Policy path: {policy_path}")
    print(f"[INFO] Policy type: {policy_cfg.type}")
    print(f"[INFO] Task: {args.task}")
    print(f"[INFO] FPS: {args.fps}")
    print(f"[INFO] run_time_s: {args.run_time_s} (<=0 means until Ctrl+C)")
    print(f"[INFO] backend: {args.backend}")
    if args.backend == "tensorrt":
        print(f"[INFO] TRT engine path: {args.trt_engine_path}")
        print(
            f"[INFO] TRT dtypes: vit={args.vit_dtype}, llm={args.llm_dtype}, dit={args.dit_dtype}, "
            f"action_head_only={args.trt_action_head_only}"
        )

    if args.dry_run:
        print("[INFO] DRY_RUN=true, exit without execution.")
        return

    # Build robot.
    from lerobot.robots import make_robot_from_config

    robot = make_robot_from_config(robot_cfg)

    # Build policy model directly from checkpoint.
    policy_class = get_policy_class(policy_cfg.type)
    policy = policy_class.from_pretrained(str(policy_path), strict=False)
    policy.to(policy_cfg.device)

    if args.backend == "tensorrt":
        if not args.trt_engine_path:
            raise ValueError("--trt-engine-path is required when --backend=tensorrt")
        from lerobot.policies.groot.trt_runtime.patch import setup_tensorrt_engines

        setup_tensorrt_engines(
            policy._groot_model,
            args.trt_engine_path,
            vit_dtype=args.vit_dtype,
            llm_dtype=args.llm_dtype,
            dit_dtype=args.dit_dtype,
            action_head_only=args.trt_action_head_only,
        )

    # Load exact checkpoint processors.
    preprocessor, postprocessor = load_pre_post_processors(policy_path)

    # Mirror the feature building in `lerobot_record` for robust key mapping.
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
