#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
if PACKAGE_PARENT.as_posix() not in sys.path:
    sys.path.insert(0, PACKAGE_PARENT.as_posix())

from vlash_iner.common import (
    DEFAULT_CALIB_DIR,
    KNOWN_TASKS,
    DEFAULT_POLICY_PATH,
    DEFAULT_ROBOT_PORT,
    ensure_repo_on_path,
    env_bool,
    optional_json_dict,
    optional_float,
    parse_bool,
    parse_camera,
    resolve_repo_root,
)


REPO_ROOT = resolve_repo_root(Path(__file__))
ensure_repo_on_path(REPO_ROOT)

from vlash_iner.policy_loader import log_info
from vlash_iner.policy_loader import load_pi05_bundle
from vlash_iner.robot_runtime import (
    RobotRuntimeConfig,
    build_so_follower_config,
    make_dataset_features_and_processors,
    print_runtime_summary,
    validate_runtime_config,
)
from vlash_iner.safety import ActionSafetyChecker, ActionSafetyConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isolated synchronous PI0.5 inference for SO101/SO100.")
    parser.add_argument("--robot-id", default=os.getenv("ROBOT_ID", "hfy_follower"))
    parser.add_argument("--robot-type", default=os.getenv("ROBOT_TYPE", "so101_follower"))
    parser.add_argument("--calib-dir", default=os.getenv("CALIB_DIR", DEFAULT_CALIB_DIR))
    parser.add_argument("--robot-port", default=os.getenv("ROBOT_PORT", DEFAULT_ROBOT_PORT))
    parser.add_argument(
        "--max-relative-target",
        type=optional_float,
        default=optional_float(os.getenv("MAX_RELATIVE_TARGET")),
    )
    parser.add_argument(
        "--max-relative-target-json",
        type=optional_json_dict,
        default=optional_json_dict(os.getenv("MAX_RELATIVE_TARGET_JSON")),
        help="Optional per-motor safety cap JSON. Overrides --max-relative-target when provided.",
    )
    parser.add_argument("--top-cam", type=parse_camera, default=parse_camera(os.getenv("TOP_CAM", "/dev/video4")))
    parser.add_argument("--wrist-cam", type=parse_camera, default=parse_camera(os.getenv("WRIST_CAM", "/dev/video6")))
    parser.add_argument("--top-cam-fourcc", default=os.getenv("TOP_CAM_FOURCC", "YUYV"))
    parser.add_argument("--wrist-cam-fourcc", default=os.getenv("WRIST_CAM_FOURCC", "MJPG"))
    parser.add_argument("--img-width", type=int, default=int(os.getenv("IMG_WIDTH", "640")))
    parser.add_argument("--img-height", type=int, default=int(os.getenv("IMG_HEIGHT", "480")))
    parser.add_argument("--fps", type=int, default=int(os.getenv("FPS", "30")))
    parser.add_argument("--policy-path", default=os.getenv("POLICY_PATH", DEFAULT_POLICY_PATH))
    parser.add_argument(
        "--task",
        default=os.getenv("DATASET_TASK"),
        help="Exact language instruction passed to PI0.5. Required unless --task-id is used.",
    )
    parser.add_argument(
        "--task-id",
        choices=sorted(KNOWN_TASKS),
        default=os.getenv("DATASET_TASK_ID"),
        help="Known task alias from the multi-task dataset.",
    )
    parser.add_argument("--list-tasks", action="store_true", help="Print known task aliases and exit.")
    parser.add_argument("--run-time-s", type=float, default=float(os.getenv("RUN_TIME_S", "0")))
    parser.add_argument("--log-interval", type=int, default=int(os.getenv("LOG_INTERVAL", "30")))
    parser.add_argument("--print-action", type=parse_bool, nargs="?", const=True, default=env_bool("PRINT_ACTION", False))
    parser.add_argument("--dry-run", type=parse_bool, nargs="?", const=True, default=env_bool("DRY_RUN", False))
    parser.add_argument(
        "--check-policy-load",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("CHECK_POLICY_LOAD", False),
    )
    parser.add_argument("--max-action-abs", type=optional_float, default=optional_float(os.getenv("MAX_ACTION_ABS")))
    parser.add_argument("--max-action-delta", type=optional_float, default=optional_float(os.getenv("MAX_ACTION_DELTA")))
    return parser


def main() -> None:
    from lerobot.datasets.utils import build_dataset_frame
    from lerobot.policies.utils import make_robot_action
    from lerobot.utils.constants import OBS_STR
    from lerobot.utils.control_utils import predict_action
    from lerobot.utils.import_utils import register_third_party_plugins
    from lerobot.utils.robot_utils import precise_sleep

    register_third_party_plugins()
    parser = build_parser()
    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.list_tasks:
        print("[INFO] Known task aliases:")
        for task_id, task in KNOWN_TASKS.items():
            print(f"[INFO]   {task_id}: {task}")
        return

    task = args.task if args.task is not None else (KNOWN_TASKS.get(args.task_id) if args.task_id else None)
    if task is None:
        parser.error(
            "A task must be specified for multi-task PI0.5 inference. "
            "Use --task-id eraser_to_box, --task-id cup_to_upper_right, "
            "--task-id eraser_then_cup, or pass --task \"...\"."
        )

    policy_path = Path(args.policy_path).expanduser()
    robot_runtime_cfg = RobotRuntimeConfig(
        robot_id=args.robot_id,
        robot_type=args.robot_type,
        calib_dir=args.calib_dir,
        robot_port=args.robot_port,
        top_cam=args.top_cam,
        wrist_cam=args.wrist_cam,
        top_cam_fourcc=args.top_cam_fourcc,
        wrist_cam_fourcc=args.wrist_cam_fourcc,
        img_width=args.img_width,
        img_height=args.img_height,
        fps=args.fps,
        max_relative_target=args.max_relative_target_json
        if args.max_relative_target_json is not None
        else args.max_relative_target,
    )
    print_runtime_summary(repo_root=REPO_ROOT, robot_cfg=robot_runtime_cfg, policy_path=policy_path, task=task)
    print(f"[INFO] run_time_s: {args.run_time_s} (<=0 means until Ctrl+C)")
    if args.dry_run:
        print("[INFO] DRY_RUN=true, exit without connecting robot or loading model weights.")
        return

    log_info("Validating runtime config: robot type, calibration file, serial port, and camera paths...")
    validate_runtime_config(robot_runtime_cfg)
    bundle = load_pi05_bundle(policy_path, repo_root=REPO_ROOT, strict=False)
    if args.check_policy_load:
        print("[INFO] CHECK_POLICY_LOAD=true, loaded policy weights and processors successfully.")
        return

    from lerobot.robots import make_robot_from_config

    log_info("Building SO follower robot config...")
    robot_cfg = build_so_follower_config(robot_runtime_cfg)
    log_info("Creating robot instance...")
    robot = make_robot_from_config(robot_cfg)
    log_info("Building dataset features and robot processors...")
    dataset_features, robot_action_processor, robot_observation_processor = make_dataset_features_and_processors(robot)
    safety = ActionSafetyChecker(
        ActionSafetyConfig(
            action_dim=bundle.policy.config.output_features["action"].shape[0],
            max_abs=args.max_action_abs,
            max_delta=args.max_action_delta,
        )
    )

    step = 0
    start_t = time.perf_counter()
    end_t = start_t + args.run_time_s if args.run_time_s > 0 else None
    try:
        log_info("Connecting robot and cameras...")
        robot.connect()
        log_info("Robot connected. Resetting policy/processors...")
        bundle.reset()
        log_info("Entering inference loop...")
        while True:
            if end_t is not None and time.perf_counter() >= end_t:
                print("[INFO] Reached requested run_time_s. Exiting inference loop.")
                break
            loop_t = time.perf_counter()
            if step == 0:
                log_info("Reading first robot observation...")
            obs = robot.get_observation()
            obs_processed = robot_observation_processor(obs)
            observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)
            if step == 0:
                log_info("Running first policy inference...")
            action_values = predict_action(
                observation=observation_frame,
                policy=bundle.policy,
                device=bundle.device,
                preprocessor=bundle.preprocessor,
                postprocessor=bundle.postprocessor,
                use_amp=bundle.policy.config.use_amp,
                task=task,
                robot_type=robot.robot_type,
            )
            action_dict = make_robot_action(action_values, dataset_features)
            action_vector = [action_dict[name] for name in dataset_features["action"]["names"]]
            safety.validate(action_vector)
            robot_action_to_send = robot_action_processor((action_dict, obs))
            if step == 0:
                log_info("Sending first robot action...")
            robot.send_action(robot_action_to_send)
            step += 1
            if args.log_interval > 0 and step % args.log_interval == 0:
                elapsed = time.perf_counter() - start_t
                print(f"[INFO] Step {step} | elapsed={elapsed:.2f}s")
                if args.print_action:
                    print(f"[INFO] Raw policy action: {action_dict}", flush=True)
            precise_sleep(max(1 / args.fps - (time.perf_counter() - loop_t), 0.0))
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received. Stopping inference.")
    except ValueError as exc:
        print(f"[ERROR] Safety check stopped inference: {exc}", flush=True)
    finally:
        if robot.is_connected:
            robot.disconnect()
        print("[INFO] Inference finished.")


if __name__ == "__main__":
    main()
