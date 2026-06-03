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
    DEFAULT_POLICY_PATH,
    DEFAULT_ROBOT_PORT,
    KNOWN_TASKS,
    ensure_repo_on_path,
    env_bool,
    optional_float,
    optional_json_dict,
    parse_bool,
    parse_camera,
    resolve_repo_root,
    validate_policy_artifacts,
)


REPO_ROOT = resolve_repo_root(Path(__file__))
ensure_repo_on_path(REPO_ROOT)

from vlash_iner.policy_loader import load_pre_post_processors, log_info
from vlash_iner.async_manager import AsyncChunkManager
from vlash_iner.policy_loader import Pi05PolicyBundle
from vlash_iner.robot_runtime import (
    RobotRuntimeConfig,
    build_so_follower_config,
    make_dataset_features_and_processors,
    print_runtime_summary,
    validate_runtime_config,
)
from vlash_iner.safety import ActionSafetyChecker, ActionSafetyConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load PI0.5 with torch.compile and run dummy warmup without connecting robot hardware."
    )
    parser.add_argument("--policy-path", default=os.getenv("POLICY_PATH", DEFAULT_POLICY_PATH))
    parser.add_argument(
        "--task",
        default=os.getenv("DATASET_TASK"),
        help="Exact language instruction used during dummy warmup. Required unless --task-id is used.",
    )
    parser.add_argument(
        "--task-id",
        choices=sorted(KNOWN_TASKS),
        default=os.getenv("DATASET_TASK_ID"),
        help="Known task alias from the multi-task dataset.",
    )
    parser.add_argument("--list-tasks", action="store_true", help="Print known task aliases and exit.")
    parser.add_argument("--warmup-steps", type=int, default=int(os.getenv("WARMUP_STEPS", "3")))
    parser.add_argument("--compile-mode", default=os.getenv("COMPILE_MODE", "reduce-overhead"))
    parser.add_argument("--device", default=os.getenv("DEVICE"), help="Override policy device, e.g. cuda or cpu.")
    parser.add_argument("--img-width", type=int, default=int(os.getenv("IMG_WIDTH", "640")))
    parser.add_argument("--img-height", type=int, default=int(os.getenv("IMG_HEIGHT", "480")))
    parser.add_argument("--state-dim", type=int, default=int(os.getenv("STATE_DIM", "6")))
    parser.add_argument("--run-robot", type=parse_bool, nargs="?", const=True, default=env_bool("RUN_ROBOT", False))
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
    parser.add_argument("--fps", type=int, default=int(os.getenv("FPS", "30")))
    parser.add_argument(
        "--control-fps",
        type=float,
        default=float(os.getenv("CONTROL_FPS", "0")),
        help="Action loop frequency. <=0 reuses --fps.",
    )
    parser.add_argument("--run-time-s", type=float, default=float(os.getenv("RUN_TIME_S", "0")))
    parser.add_argument("--log-interval", type=int, default=int(os.getenv("LOG_INTERVAL", "30")))
    parser.add_argument("--connect-retries", type=int, default=int(os.getenv("CONNECT_RETRIES", "3")))
    parser.add_argument("--connect-retry-s", type=float, default=float(os.getenv("CONNECT_RETRY_S", "1.0")))
    parser.add_argument("--n-action-steps", type=int, default=int(os.getenv("N_ACTION_STEPS", "0")))
    parser.add_argument("--inference-overlap-steps", type=int, default=int(os.getenv("INFERENCE_OVERLAP_STEPS", "8")))
    parser.add_argument(
        "--background-inference",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("BACKGROUND_INFERENCE", True),
    )
    parser.add_argument("--chunk-blend-steps", type=int, default=int(os.getenv("CHUNK_BLEND_STEPS", "2")))
    parser.add_argument(
        "--reuse-observation-within-chunk",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("REUSE_OBSERVATION_WITHIN_CHUNK", True),
    )
    parser.add_argument(
        "--future-state-aware",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("FUTURE_STATE_AWARE", True),
    )
    parser.add_argument("--action-quant-ratio", type=int, default=int(os.getenv("ACTION_QUANT_RATIO", "1")))
    parser.add_argument("--max-action-abs", type=optional_float, default=optional_float(os.getenv("MAX_ACTION_ABS")))
    parser.add_argument("--max-action-delta", type=optional_float, default=optional_float(os.getenv("MAX_ACTION_DELTA")))
    parser.add_argument(
        "--summarize-artifacts",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("SUMMARIZE_ARTIFACTS", False),
    )
    return parser


def resolve_task(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    if args.list_tasks:
        print("[INFO] Known task aliases:")
        for task_id, task in KNOWN_TASKS.items():
            print(f"[INFO]   {task_id}: {task}")
        raise SystemExit(0)

    task = args.task if args.task is not None else (KNOWN_TASKS.get(args.task_id) if args.task_id else None)
    if task is None:
        parser.error(
            "A task must be specified for multi-task PI0.5 compile warmup. "
            "Use --task-id eraser_to_box, --task-id cup_to_upper_right, "
            "--task-id eraser_then_cup, or pass --task \"...\"."
        )
    return task


def load_compiled_policy(policy_path: Path, *, device: str | None, compile_mode: str):
    from lerobot import policies  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class
    from lerobot.utils.utils import get_safe_torch_device

    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    policy_cfg.pretrained_path = policy_path
    policy_cfg.compile_model = True
    policy_cfg.compile_mode = compile_mode
    if device is not None:
        policy_cfg.device = device

    policy_class = get_policy_class(policy_cfg.type)
    log_info(
        "Loading compiled policy: "
        f"type={policy_cfg.type} device={policy_cfg.device} compile_mode={policy_cfg.compile_mode}"
    )
    policy = policy_class.from_pretrained(str(policy_path), config=policy_cfg, strict=False)
    safe_device = get_safe_torch_device(policy.config.device)
    log_info(f"Moving compiled policy to device={safe_device}...")
    policy.to(safe_device)
    policy.eval()
    return policy, safe_device


def make_dummy_observation(*, img_height: int, img_width: int, state_dim: int) -> dict:
    return {
        "observation.state": np.zeros((state_dim,), dtype=np.float32),
        "observation.images.top": np.zeros((img_height, img_width, 3), dtype=np.uint8),
        "observation.images.wrist": np.zeros((img_height, img_width, 3), dtype=np.uint8),
    }


def warmup_compiled_bundle(
    bundle: Pi05PolicyBundle,
    *,
    task: str,
    robot_type: str,
    warmup_steps: int,
    img_height: int,
    img_width: int,
    state_dim: int,
) -> None:
    from lerobot.policies.utils import prepare_observation_for_inference

    dummy_observation = make_dummy_observation(
        img_height=img_height,
        img_width=img_width,
        state_dim=state_dim,
    )

    log_info("Starting torch.compile warmup with dummy top/wrist images and zero state...")
    bundle.preprocessor.reset()
    bundle.postprocessor.reset()
    warmup_start = time.perf_counter()
    for step in range(1, warmup_steps + 1):
        step_start = time.perf_counter()
        observation = dict(dummy_observation)
        observation = prepare_observation_for_inference(observation, bundle.device, task, robot_type)
        observation = bundle.preprocessor(observation)
        with torch.inference_mode():
            raw_chunk = bundle.policy.predict_action_chunk(observation)
            action_chunk = bundle.postprocessor(raw_chunk)
        if bundle.device.type == "cuda":
            torch.cuda.synchronize(bundle.device)
        elapsed = time.perf_counter() - step_start
        log_info(
            f"Warmup step {step}/{warmup_steps}: "
            f"elapsed={elapsed:.3f}s action_shape={tuple(action_chunk.shape)}"
        )

    total_elapsed = time.perf_counter() - warmup_start
    log_info(f"Compile warmup complete in {total_elapsed:.2f}s.")


def run_robot_loop(
    bundle: Pi05PolicyBundle,
    *,
    robot_runtime_cfg: RobotRuntimeConfig,
    task: str,
    run_time_s: float,
    control_fps: float,
    log_interval: int,
    connect_retries: int,
    connect_retry_s: float,
    n_action_steps_override: int,
    inference_overlap_steps: int,
    background_inference: bool,
    chunk_blend_steps: int,
    reuse_observation_within_chunk: bool,
    future_state_aware: bool,
    action_quant_ratio: int,
    max_action_abs: float | None,
    max_action_delta: float | None,
) -> None:
    from lerobot.datasets.utils import build_dataset_frame
    from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
    from lerobot.robots import make_robot_from_config
    from lerobot.utils.constants import OBS_STR
    from lerobot.utils.import_utils import register_third_party_plugins
    from lerobot.utils.robot_utils import precise_sleep

    register_third_party_plugins()
    if action_quant_ratio != 1:
        raise NotImplementedError(
            "action_quant_ratio > 1 is not enabled in this compile runtime. Keep --action-quant-ratio 1."
        )

    validate_runtime_config(robot_runtime_cfg)
    robot_cfg = build_so_follower_config(robot_runtime_cfg)
    robot = make_robot_from_config(robot_cfg)
    dataset_features, robot_action_processor, robot_observation_processor = make_dataset_features_and_processors(robot)
    action_names = dataset_features["action"]["names"]
    action_dim = len(action_names)
    n_action_steps = n_action_steps_override or int(bundle.policy.config.n_action_steps)
    safety = ActionSafetyChecker(
        ActionSafetyConfig(
            action_dim=action_dim,
            max_abs=max_action_abs,
            max_delta=max_action_delta,
        )
    )

    def predict_chunk(observation_frame: dict, future_state: np.ndarray | None) -> np.ndarray:
        observation = dict(observation_frame)
        if future_state is not None:
            observation["observation.state"] = future_state.astype(np.float32, copy=False)
        observation = prepare_observation_for_inference(observation, bundle.device, task, robot.robot_type)
        observation = bundle.preprocessor(observation)
        with torch.inference_mode():
            raw_chunk = bundle.policy.predict_action_chunk(observation)
            processed = bundle.postprocessor(raw_chunk)
        chunk = processed.squeeze(0).detach().cpu().numpy()
        return np.asarray(chunk, dtype=np.float32)

    manager = AsyncChunkManager(
        predict_chunk,
        n_action_steps=n_action_steps,
        overlap_steps=inference_overlap_steps,
        action_dim=action_dim,
        background_inference=background_inference,
        blend_steps=chunk_blend_steps,
        future_state_aware=future_state_aware,
    )

    step = 0
    start_t = time.perf_counter()
    end_t = start_t + run_time_s if run_time_s > 0 else None
    latest_observation_frame: dict | None = None
    latest_obs: dict | None = None
    try:
        connect_attempts = max(connect_retries, 1)
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
                print(f"[INFO] Retrying robot connect in {connect_retry_s:.1f}s...", flush=True)
                time.sleep(connect_retry_s)

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
                if not reuse_observation_within_chunk:
                    obs = robot.get_observation()
                    latest_obs = obs
                if obs is None:
                    raise RuntimeError("No cached robot observation is available for action processing.")

            action_vector = manager.get_action(latest_observation_frame)
            safety.validate(action_vector)
            if (step + 1) % action_quant_ratio == 0:
                action_tensor = torch.as_tensor(action_vector, dtype=torch.float32).unsqueeze(0)
                action_dict = make_robot_action(action_tensor, dataset_features)
                robot_action_to_send = robot_action_processor((action_dict, obs))
                robot.send_action(robot_action_to_send)

            step += 1
            if log_interval > 0 and step % log_interval == 0:
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
        print("[INFO] Compiled async robot inference finished.")


def main() -> None:

    parser = build_parser()
    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    task = resolve_task(parser, args)
    policy_path = Path(args.policy_path).expanduser().resolve()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"Policy path does not exist: {policy_path}")
    if args.warmup_steps <= 0:
        parser.error("--warmup-steps must be positive")

    if args.action_quant_ratio != 1:
        raise NotImplementedError(
            "action_quant_ratio > 1 is not enabled in this compile runtime. Keep --action-quant-ratio 1."
        )

    mode = "compile warmup + robot async inference" if args.run_robot else "compile warmup only"
    log_info(f"Mode: {mode}")
    log_info(f"Repo root: {REPO_ROOT}")
    log_info(f"Policy path: {policy_path}")
    log_info(f"Task: {task}")
    log_info(f"Warmup steps: {args.warmup_steps}")

    validate_policy_artifacts(policy_path, summarize=args.summarize_artifacts)

    policy, device = load_compiled_policy(policy_path, device=args.device, compile_mode=args.compile_mode)
    preprocessor, postprocessor = load_pre_post_processors(policy_path)
    bundle = Pi05PolicyBundle(
        policy=policy,
        policy_cfg=policy.config,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=device,
        policy_path=policy_path,
    )

    warmup_compiled_bundle(
        bundle,
        task=task,
        robot_type=args.robot_type,
        warmup_steps=args.warmup_steps,
        img_height=args.img_height,
        img_width=args.img_width,
        state_dim=args.state_dim,
    )

    if not args.run_robot:
        log_info("No robot hardware was connected or commanded.")
        return

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
    control_fps = args.control_fps if args.control_fps > 0 else float(args.fps)
    print_runtime_summary(repo_root=REPO_ROOT, robot_cfg=robot_runtime_cfg, policy_path=policy_path, task=task)
    log_info(f"run_time_s: {args.run_time_s} (<=0 means until Ctrl+C)")
    log_info(f"control_fps: {control_fps}")
    log_info(f"inference_overlap_steps: {args.inference_overlap_steps}")
    log_info(f"background_inference: {args.background_inference}")
    log_info(f"future_state_aware: {args.future_state_aware}")
    log_info(f"chunk_blend_steps: {args.chunk_blend_steps}")
    log_info(f"reuse_observation_within_chunk: {args.reuse_observation_within_chunk}")
    run_robot_loop(
        bundle,
        robot_runtime_cfg=robot_runtime_cfg,
        task=task,
        run_time_s=args.run_time_s,
        control_fps=control_fps,
        log_interval=args.log_interval,
        connect_retries=args.connect_retries,
        connect_retry_s=args.connect_retry_s,
        n_action_steps_override=args.n_action_steps,
        inference_overlap_steps=args.inference_overlap_steps,
        background_inference=args.background_inference,
        chunk_blend_steps=args.chunk_blend_steps,
        reuse_observation_within_chunk=args.reuse_observation_within_chunk,
        future_state_aware=args.future_state_aware,
        action_quant_ratio=args.action_quant_ratio,
        max_action_abs=args.max_action_abs,
        max_action_delta=args.max_action_delta,
    )


if __name__ == "__main__":
    main()
