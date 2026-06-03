#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

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

from vlash_iner.async_manager import AsyncChunkManager
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
    parser = argparse.ArgumentParser(description="Isolated VLASH-style async PI0.5 inference for SO101/SO100.")
    parser.add_argument("--robot-id", default=os.getenv("ROBOT_ID", "hfy_follower"))
    parser.add_argument("--robot-type", default=os.getenv("ROBOT_TYPE", "so101_follower"))
    parser.add_argument("--calib-dir", default=os.getenv("CALIB_DIR", DEFAULT_CALIB_DIR))
    parser.add_argument("--robot-port", default=os.getenv("ROBOT_PORT", DEFAULT_ROBOT_PORT))
    parser.add_argument("--max-relative-target", type=optional_float, default=optional_float(os.getenv("MAX_RELATIVE_TARGET")))
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
    parser.add_argument(
        "--control-fps",
        type=float,
        default=float(os.getenv("CONTROL_FPS", "0")),
        help="Action loop frequency. <=0 reuses --fps.",
    )
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
    parser.add_argument("--connect-retries", type=int, default=int(os.getenv("CONNECT_RETRIES", "3")))
    parser.add_argument("--connect-retry-s", type=float, default=float(os.getenv("CONNECT_RETRY_S", "1.0")))
    parser.add_argument("--dry-run", type=parse_bool, nargs="?", const=True, default=env_bool("DRY_RUN", False))
    parser.add_argument("--check-policy-load", type=parse_bool, nargs="?", const=True, default=env_bool("CHECK_POLICY_LOAD", False))
    parser.add_argument("--n-action-steps", type=int, default=int(os.getenv("N_ACTION_STEPS", "0")))
    parser.add_argument("--inference-overlap-steps", type=int, default=int(os.getenv("INFERENCE_OVERLAP_STEPS", "0")))
    parser.add_argument(
        "--background-inference",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("BACKGROUND_INFERENCE", False),
        help="Prefetch the next action chunk in a background thread at the overlap point.",
    )
    parser.add_argument(
        "--chunk-blend-steps",
        type=int,
        default=int(os.getenv("CHUNK_BLEND_STEPS", "0")),
        help="Smooth the first N actions of a new chunk from the previous action.",
    )
    parser.add_argument("--action-quant-ratio", type=int, default=int(os.getenv("ACTION_QUANT_RATIO", "1")))
    parser.add_argument(
        "--reuse-observation-within-chunk",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("REUSE_OBSERVATION_WITHIN_CHUNK", False),
        help="Read camera/state only when a new chunk inference is needed, then execute cached chunk actions faster.",
    )
    parser.add_argument("--future-state-aware", type=parse_bool, nargs="?", const=True, default=env_bool("FUTURE_STATE_AWARE", False))
    parser.add_argument("--max-action-abs", type=optional_float, default=optional_float(os.getenv("MAX_ACTION_ABS")))
    parser.add_argument("--max-action-delta", type=optional_float, default=optional_float(os.getenv("MAX_ACTION_DELTA")))
    return parser


def main() -> None:
    from lerobot.datasets.utils import build_dataset_frame
    from lerobot.policies.utils import make_robot_action
    from lerobot.policies.utils import prepare_observation_for_inference
    from lerobot.utils.constants import OBS_STR
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

    if args.action_quant_ratio != 1:
        raise NotImplementedError(
            "action_quant_ratio > 1 is intentionally disabled in this first async implementation. "
            "Keep --action-quant-ratio 1 until skipped-action execution semantics are implemented."
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
    print(f"[INFO] inference_overlap_steps: {args.inference_overlap_steps}")
    print(f"[INFO] background_inference: {args.background_inference}")
    print(f"[INFO] chunk_blend_steps: {args.chunk_blend_steps}")
    print(f"[INFO] future_state_aware: {args.future_state_aware}")
    print(f"[INFO] action_quant_ratio: {args.action_quant_ratio}")
    control_fps = args.control_fps if args.control_fps > 0 else float(args.fps)
    print(f"[INFO] control_fps: {control_fps}")
    print(f"[INFO] reuse_observation_within_chunk: {args.reuse_observation_within_chunk}")
    if args.dry_run:
        print("[INFO] DRY_RUN=true, exit without connecting robot or loading model weights.")
        return

    validate_runtime_config(robot_runtime_cfg)
    bundle = load_pi05_bundle(policy_path, repo_root=REPO_ROOT, strict=False)
    if args.check_policy_load:
        print("[INFO] CHECK_POLICY_LOAD=true, loaded policy weights and processors successfully.")
        return

    from lerobot.robots import make_robot_from_config

    robot_cfg = build_so_follower_config(robot_runtime_cfg)
    robot = make_robot_from_config(robot_cfg)
    dataset_features, robot_action_processor, robot_observation_processor = make_dataset_features_and_processors(robot)
    action_names = dataset_features["action"]["names"]
    action_dim = len(action_names)
    n_action_steps = args.n_action_steps or int(bundle.policy.config.n_action_steps)

    safety = ActionSafetyChecker(
        ActionSafetyConfig(
            action_dim=action_dim,
            max_abs=args.max_action_abs,
            max_delta=args.max_action_delta,
        )
    )

    def predict_chunk(observation_frame: dict, future_state: np.ndarray | None) -> np.ndarray:
        observation = dict(observation_frame)
        if future_state is not None:
            observation["observation.state"] = future_state.astype(np.float32, copy=False)
        observation = prepare_observation_for_inference(
            observation,
            bundle.device,
            task,
            robot.robot_type,
        )
        observation = bundle.preprocessor(observation)
        with torch.inference_mode():
            raw_chunk = bundle.policy.predict_action_chunk(observation)
            processed = bundle.postprocessor(raw_chunk)
        chunk = processed.squeeze(0).detach().cpu().numpy()
        return np.asarray(chunk, dtype=np.float32)

    manager = AsyncChunkManager(
        predict_chunk,
        n_action_steps=n_action_steps,
        overlap_steps=args.inference_overlap_steps,
        action_dim=action_dim,
        background_inference=args.background_inference,
        blend_steps=args.chunk_blend_steps,
        future_state_aware=args.future_state_aware,
    )

    step = 0
    start_t = time.perf_counter()
    end_t = start_t + args.run_time_s if args.run_time_s > 0 else None
    latest_observation_frame: dict | None = None
    latest_obs: dict | None = None
    try:
        connect_attempts = max(args.connect_retries, 1)
        for attempt in range(1, connect_attempts + 1):
            try:
                print(f"[INFO] Connecting robot and cameras... attempt {attempt}/{connect_attempts}", flush=True)
                robot.connect()
                break
            except ConnectionError as exc:
                if robot.is_connected:
                    robot.disconnect()
                if attempt >= connect_attempts:
                    raise
                print(f"[WARN] Robot connect failed: {exc}", flush=True)
                print(f"[INFO] Retrying robot connect in {args.connect_retry_s:.1f}s...", flush=True)
                time.sleep(args.connect_retry_s)
        bundle.reset()
        while True:
            if end_t is not None and time.perf_counter() >= end_t:
                print("[INFO] Reached requested run_time_s. Exiting inference loop.")
                break
            loop_t = time.perf_counter()
            if manager.should_fetch_observation() or latest_observation_frame is None:
                obs = robot.get_observation()
                latest_obs = obs
                obs_processed = robot_observation_processor(obs)
                latest_observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)
            else:
                obs = latest_obs
                if not args.reuse_observation_within_chunk:
                    obs = robot.get_observation()
                    latest_obs = obs
                if obs is None:
                    raise RuntimeError("No cached robot observation is available for action processing.")
            action_vector = manager.get_action(latest_observation_frame)
            safety.validate(action_vector)
            if (step + 1) % args.action_quant_ratio == 0:
                action_tensor = torch.as_tensor(action_vector, dtype=torch.float32).unsqueeze(0)
                action_dict = make_robot_action(action_tensor, dataset_features)
                robot_action_to_send = robot_action_processor((action_dict, obs))
                robot.send_action(robot_action_to_send)
            step += 1
            if args.log_interval > 0 and step % args.log_interval == 0:
                elapsed = time.perf_counter() - start_t
                print(
                    f"[INFO] Step {step} | elapsed={elapsed:.2f}s "
                    f"last_infer={manager.stats.last_inference_s:.3f}s "
                    f"infer_count={manager.stats.inference_count} "
                    f"pending={manager.stats.pending_inference} "
                    f"wait_count={manager.stats.wait_count}"
                )
            precise_sleep(max(1 / control_fps - (time.perf_counter() - loop_t), 0.0))
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received. Stopping inference.")
    except ConnectionError as exc:
        print(f"[ERROR] Robot connection failed: {exc}", flush=True)
    finally:
        manager.close()
        if robot.is_connected:
            robot.disconnect()
        print("[INFO] Async inference finished.")


if __name__ == "__main__":
    main()
