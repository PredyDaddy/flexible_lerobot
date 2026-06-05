#!/usr/bin/env python

"""Run SO101 PI0.5 inference with split TensorRT prefix-cache and denoise engines.

Example:
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/run_pi05_split_trt_infer_so101.py \
    --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
    --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
    --top-cam /dev/video4 \
    --wrist-cam /dev/video6 \
    --task "Put the eraser into the small box" \
    --run-time-s 120 \
    --confirm-control
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
OPENPI_TRT_DIR = SCRIPT_DIR.parent
REPO_ROOT = OPENPI_TRT_DIR.parents[1]
if OPENPI_TRT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, OPENPI_TRT_DIR.as_posix())
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from runtime.pi05_trt_split import patch_sample_actions_with_split_trt  # noqa: E402
from scripts.pi05_onnx_common import (  # noqa: E402
    DEFAULT_POLICY_PATH,
    DEFAULT_TASK,
    configure_runtime,
    ensure_local_tokenizer_dir,
    load_policy,
)

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: E402
from lerobot.datasets.pipeline_features import (  # noqa: E402
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts  # noqa: E402
from lerobot.policies.utils import make_robot_action  # noqa: E402
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, make_default_processors  # noqa: E402
from lerobot.processor.converters import (  # noqa: E402
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig  # noqa: E402
from lerobot.utils.constants import OBS_STR  # noqa: E402
from lerobot.utils.control_utils import predict_action  # noqa: E402
from lerobot.utils.robot_utils import precise_sleep  # noqa: E402
from lerobot.utils.utils import get_safe_torch_device  # noqa: E402


DEFAULT_PREFIX_ENGINE_PATH = Path("my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.engine")
DEFAULT_DENOISE_ENGINE_PATH = Path("my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.engine")
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
        description=(
            "SO101/SO100 PI0.5 real inference with split TensorRT prefix-cache and denoise engines. "
            "Requires --confirm-control before robot connection/action sending."
        )
    )
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--prefix-engine-path", type=Path, default=DEFAULT_PREFIX_ENGINE_PATH)
    parser.add_argument("--denoise-engine-path", type=Path, default=DEFAULT_DENOISE_ENGINE_PATH)
    parser.add_argument("--robot-id", default=os.getenv("ROBOT_ID", "hfy_follower"))
    parser.add_argument("--robot-type", default=os.getenv("ROBOT_TYPE", "so101_follower"))
    parser.add_argument("--calib-dir", default=os.getenv("CALIB_DIR", DEFAULT_CALIB_DIR))
    parser.add_argument("--robot-port", default=os.getenv("ROBOT_PORT", DEFAULT_ROBOT_PORT))
    parser.add_argument(
        "--max-relative-target",
        type=optional_float,
        default=optional_float(os.getenv("MAX_RELATIVE_TARGET")),
    )
    parser.add_argument("--top-cam", type=parse_camera, default=parse_camera(os.getenv("TOP_CAM", "/dev/video4")))
    parser.add_argument(
        "--wrist-cam",
        type=parse_camera,
        default=parse_camera(os.getenv("WRIST_CAM", "/dev/video6")),
    )
    parser.add_argument("--top-cam-fourcc", default=os.getenv("TOP_CAM_FOURCC", "YUYV"))
    parser.add_argument("--wrist-cam-fourcc", default=os.getenv("WRIST_CAM_FOURCC", "MJPG"))
    parser.add_argument("--img-width", type=int, default=int(os.getenv("IMG_WIDTH", "640")))
    parser.add_argument("--img-height", type=int, default=int(os.getenv("IMG_HEIGHT", "480")))
    parser.add_argument("--fps", type=int, default=int(os.getenv("FPS", "30")))
    parser.add_argument("--task", default=os.getenv("DATASET_TASK", DEFAULT_TASK))
    parser.add_argument("--run-time-s", type=float, default=float(os.getenv("RUN_TIME_S", "0")))
    parser.add_argument("--log-interval", type=int, default=int(os.getenv("LOG_INTERVAL", "30")))
    parser.add_argument(
        "--dry-run",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("DRY_RUN", False),
        help="Print config and exit before loading policy or touching hardware.",
    )
    parser.add_argument(
        "--check-policy-load",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("CHECK_POLICY_LOAD", False),
        help="Load policy/processors/TensorRT engines, then exit before robot connection.",
    )
    parser.add_argument(
        "--confirm-control",
        action="store_true",
        help="Required to connect robot and send actions. Without this flag the script exits safely.",
    )
    return parser


def load_pre_post_processors(
    policy_path: Path,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
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
    configure_runtime()
    args = build_parser().parse_args()

    policy_path = args.policy_path.expanduser().resolve()
    prefix_engine_path = args.prefix_engine_path.expanduser()
    denoise_engine_path = args.denoise_engine_path.expanduser()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"Policy path does not exist: {policy_path}")
    if not prefix_engine_path.is_file():
        raise FileNotFoundError(f"Prefix TensorRT engine does not exist: {prefix_engine_path}")
    if not denoise_engine_path.is_file():
        raise FileNotFoundError(f"Denoise TensorRT engine does not exist: {denoise_engine_path}")

    print("[SAFETY] This script can control the robot only with explicit --confirm-control.")
    print("[SAFETY] No robot connection or action sending happens before that gate.")
    print(f"[INFO] Policy path: {policy_path}")
    print(f"[INFO] TensorRT prefix engine: {prefix_engine_path}")
    print(f"[INFO] TensorRT denoise engine: {denoise_engine_path}")
    print(f"[INFO] Robot port: {args.robot_port}")
    print(f"[INFO] Top camera: {args.top_cam} fourcc={args.top_cam_fourcc}")
    print(f"[INFO] Wrist camera: {args.wrist_cam} fourcc={args.wrist_cam_fourcc}")
    print(f"[INFO] Task: {args.task}")
    print(f"[INFO] run_time_s: {args.run_time_s} (<=0 means until Ctrl+C)")

    if args.dry_run:
        print("[INFO] DRY_RUN=true, exit without loading model or touching hardware.")
        return

    ensure_local_tokenizer_dir()

    policy = load_policy(policy_path, device="cuda", model_dtype="float32")
    runtime = patch_sample_actions_with_split_trt(policy, prefix_engine_path, denoise_engine_path)
    preprocessor, postprocessor = load_pre_post_processors(policy_path)

    if args.check_policy_load:
        print("[INFO] CHECK_POLICY_LOAD=true, loaded policy/processors/split TRT engines.")
        print("[INFO] Exiting before robot connection.")
        return

    if not args.confirm_control:
        print("[SAFETY] Missing --confirm-control. Exiting before robot connection/action sending.")
        return

    if args.robot_type not in {"so100_follower", "so101_follower"}:
        raise ValueError(f"Unsupported robot_type={args.robot_type!r}")

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

    from lerobot.robots import make_robot_from_config

    robot = make_robot_from_config(robot_cfg)
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
                print("[INFO] Reached requested run_time_s. Exiting split TRT inference loop.")
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
                print(f"[INFO] Step {step} | elapsed={elapsed:.2f}s | backend=split_trt")

            dt_s = time.perf_counter() - loop_t
            precise_sleep(max(1 / args.fps - dt_s, 0.0))
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received. Stopping split TRT inference.")
    finally:
        if robot.is_connected:
            robot.disconnect()
        # Keep runtime referenced until after loop teardown so engines stay alive.
        del runtime
        print("[INFO] Split TRT inference finished.")


if __name__ == "__main__":
    main()
