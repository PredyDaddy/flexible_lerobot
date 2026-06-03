#!/usr/bin/env python

"""Run pure PI0.5 policy inference on SO follower robot (no dataset recording).

This script performs:
1) read robot observation
2) policy inference
3) send action to robot
in a real-time loop.

Notes for PI0.5:
- The saved preprocessor uses the PaliGemma tokenizer name "google/paligemma-3b-pt-224".
  If your machine cannot access huggingface.co, make sure this tokenizer is available locally.
  In our setup we keep a local directory at:
    <repo_root>/google/paligemma-3b-pt-224
  (it can be a symlink to a ModelScope download).
 
Example:
python my_devs/train/pi/so101/run_pi05_infer.py \
    --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
    --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
    --top-cam /dev/video4 \
    --wrist-cam /dev/video6 \
    --task "Put the eraser into the small box" \
    --run-time-s 120
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from safetensors import safe_open


def resolve_repo_root(script_path: Path) -> Path:
    resolved_path = script_path.resolve()
    for candidate in resolved_path.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from script path: {script_path}")


REPO_ROOT = resolve_repo_root(Path(__file__))
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

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
    "/data/cqy_workspace/flexible_lerobot/outputs/pi05_eraser_cup_multi_task_runs/"
    "20260602_200955/checkpoints/last/pretrained_model"
)
DEFAULT_TASK = "First put the eraser into the small box, then move the cup back to the upper-right corner"
DEFAULT_ROBOT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00"
DEFAULT_CALIB_DIR = "/home/cqy/.cache/huggingface/lerobot/calibration/robots/so_follower"


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


def parse_camera(value: str) -> int | Path:
    if value.isdecimal():
        return int(value)
    return Path(value).expanduser()


def optional_float(value: str | None) -> float | None:
    if value is None or value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pure PI0.5 inference loop for SO101/SO100 follower robot via LeRobot APIs."
    )
    parser.add_argument("--robot-id", default=os.getenv("ROBOT_ID", "hfy_follower"))
    parser.add_argument("--robot-type", default=os.getenv("ROBOT_TYPE", "so101_follower"))
    parser.add_argument(
        "--calib-dir",
        default=os.getenv("CALIB_DIR", DEFAULT_CALIB_DIR),
    )
    parser.add_argument("--robot-port", default=os.getenv("ROBOT_PORT", DEFAULT_ROBOT_PORT))
    parser.add_argument(
        "--max-relative-target",
        type=optional_float,
        default=optional_float(os.getenv("MAX_RELATIVE_TARGET")),
        help="Optional SO follower safety clip for each motor target. Example: 10. Use none/omit to disable.",
    )

    parser.add_argument(
        "--top-cam",
        type=parse_camera,
        default=parse_camera(os.getenv("TOP_CAM", os.getenv("TOP_CAM_INDEX", "/dev/video4"))),
        help="Top camera device path or numeric OpenCV index.",
    )
    parser.add_argument(
        "--wrist-cam",
        type=parse_camera,
        default=parse_camera(os.getenv("WRIST_CAM", os.getenv("WRIST_CAM_INDEX", "/dev/video6"))),
        help="Wrist camera device path or numeric OpenCV index.",
    )
    parser.add_argument("--top-cam-fourcc", default=os.getenv("TOP_CAM_FOURCC", "YUYV"))
    parser.add_argument("--wrist-cam-fourcc", default=os.getenv("WRIST_CAM_FOURCC", "MJPG"))
    parser.add_argument("--img-width", type=int, default=int(os.getenv("IMG_WIDTH", "640")))
    parser.add_argument("--img-height", type=int, default=int(os.getenv("IMG_HEIGHT", "480")))
    parser.add_argument("--fps", type=int, default=int(os.getenv("FPS", "30")))

    parser.add_argument("--policy-path", default=os.getenv("POLICY_PATH", DEFAULT_POLICY_PATH))
    parser.add_argument(
        "--task",
        default=os.getenv("DATASET_TASK", DEFAULT_TASK),
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
        "--check-policy-load",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("CHECK_POLICY_LOAD", False),
        help="Load checkpoint weights and saved processors, then exit without connecting robot.",
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


def ensure_local_tokenizer_dir(repo_root: Path) -> None:
    """Fail fast if the common offline tokenizer directory is missing.

    The PI0.5 preprocessor uses tokenizer_name='google/paligemma-3b-pt-224'.
    If there is a local directory with the same name under repo root, transformers
    will load it as a local tokenizer path (no network required).
    """
    local_tok = repo_root / "google" / "paligemma-3b-pt-224"
    if not local_tok.is_dir():
        raise FileNotFoundError(
            "Missing local tokenizer directory for offline PI0.5 inference.\n"
            f"Expected: {local_tok}\n"
            "Fix:\n"
            "  - Download tokenizer files to that directory (or create a symlink), then retry.\n"
            "  - In our setup this is a symlink to ModelScope cache:\n"
            "      assets/modelscope/google/paligemma-3b-pt-224\n"
        )


def summarize_safetensors(path: Path, limit: int = 8) -> None:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        print(f"[INFO] {path.name}: {path.stat().st_size} bytes, tensors={len(keys)}")
        for key in keys[:limit]:
            tensor_slice = handle.get_slice(key)
            print(
                f"[INFO]   {key}: "
                f"shape={tuple(tensor_slice.get_shape())} dtype={tensor_slice.get_dtype()}"
            )
        if len(keys) > limit:
            print("[INFO]   ...")


def validate_policy_artifacts(policy_path: Path) -> None:
    required_files = [
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_preprocessor_step_2_normalizer_processor.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "train_config.json",
    ]
    missing = [name for name in required_files if not (policy_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Policy checkpoint is missing required files: {missing}")

    with (policy_path / "config.json").open() as f:
        config = json.load(f)

    expected_inputs = {
        "observation.state": [6],
        "observation.images.top": [3, 480, 640],
        "observation.images.wrist": [3, 480, 640],
    }
    for key, expected_shape in expected_inputs.items():
        feature = config.get("input_features", {}).get(key)
        if feature is None or feature.get("shape") != expected_shape:
            raise ValueError(
                f"Unexpected input feature for {key}: {feature}. Expected shape {expected_shape}."
            )

    action_feature = config.get("output_features", {}).get("action")
    if action_feature is None or action_feature.get("shape") != [6]:
        raise ValueError(f"Unexpected action feature: {action_feature}. Expected shape [6].")

    print("[INFO] Policy artifacts found and feature shapes match SO101 top/wrist setup.")
    print(
        "[INFO] Policy config: "
        f"type={config.get('type')} dtype={config.get('dtype')} "
        f"chunk_size={config.get('chunk_size')} n_action_steps={config.get('n_action_steps')}"
    )
    summarize_safetensors(policy_path / "model.safetensors")
    summarize_safetensors(policy_path / "policy_preprocessor_step_2_normalizer_processor.safetensors")
    summarize_safetensors(policy_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors")


def main() -> None:
    register_third_party_plugins()
    args = build_parser().parse_args()

    # Make script location-independent: switch to repo root so relative tokenizer paths work.
    repo_root = REPO_ROOT
    os.chdir(repo_root)

    # Reduce noise / avoid fork warnings.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    policy_path = Path(args.policy_path).expanduser()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"Policy path does not exist: {policy_path}")
    validate_policy_artifacts(policy_path)

    if args.robot_type not in {"so100_follower", "so101_follower"}:
        raise ValueError(
            f"Unsupported robot_type={args.robot_type!r}. "
            "This script currently supports so100_follower/so101_follower."
        )

    cameras = {
        "top": OpenCVCameraConfig(
            index_or_path=args.top_cam,
            width=args.img_width,
            height=args.img_height,
            fps=args.fps,
            fourcc=args.top_cam_fourcc,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=args.wrist_cam,
            width=args.img_width,
            height=args.img_height,
            fps=args.fps,
            fourcc=args.wrist_cam_fourcc,
        ),
    }
    robot_cfg = SOFollowerRobotConfig(
        id=args.robot_id,
        calibration_dir=maybe_path(args.calib_dir),
        port=args.robot_port,
        max_relative_target=args.max_relative_target,
        cameras=cameras,
    )

    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    policy_cfg.pretrained_path = policy_path

    print(f"[INFO] Repo root: {repo_root}")
    print(f"[INFO] Robot id: {args.robot_id}")
    print(f"[INFO] Robot type (requested): {args.robot_type}")
    print(f"[INFO] Robot port: {args.robot_port}")
    calibration_file = Path(args.calib_dir).expanduser() / f"{args.robot_id}.json"
    print(f"[INFO] Robot calibration file: {calibration_file}")
    print(f"[INFO] Top camera: {args.top_cam} fourcc={args.top_cam_fourcc}")
    print(f"[INFO] Wrist camera: {args.wrist_cam} fourcc={args.wrist_cam_fourcc}")
    print(f"[INFO] max_relative_target: {args.max_relative_target}")
    print(f"[INFO] Policy path: {policy_path}")
    print(f"[INFO] Policy type: {policy_cfg.type}")
    print(f"[INFO] Task: {args.task}")
    print(f"[INFO] FPS: {args.fps}")
    print(f"[INFO] run_time_s: {args.run_time_s} (<=0 means until Ctrl+C)")

    if args.dry_run:
        print("[INFO] DRY_RUN=true, exit without execution.")
        return

    ensure_local_tokenizer_dir(repo_root)

    # Build policy model directly from checkpoint.
    policy_class = get_policy_class(policy_cfg.type)
    policy = policy_class.from_pretrained(str(policy_path), strict=False)
    policy.to(policy_cfg.device)

    # Load exact checkpoint processors.
    preprocessor, postprocessor = load_pre_post_processors(policy_path)

    if args.check_policy_load:
        print("[INFO] CHECK_POLICY_LOAD=true, loaded policy weights and processors successfully.")
        return

    # Build robot only after policy loading checks pass, so offline validation does not touch hardware.
    from lerobot.robots import make_robot_from_config

    robot = make_robot_from_config(robot_cfg)

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
