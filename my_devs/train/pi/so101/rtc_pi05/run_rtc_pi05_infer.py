#!/usr/bin/env python

from __future__ import annotations

import argparse
import builtins
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any


def resolve_repo_root(script_path: Path) -> Path:
    resolved = script_path.resolve()
    for candidate in resolved.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from script path: {script_path}")


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = resolve_repo_root(Path(__file__))
for path in (REPO_ROOT, SCRIPT_DIR):
    if path.as_posix() not in sys.path:
        sys.path.insert(0, path.as_posix())

from lerobot.configs.types import RTCAttentionSchedule  # noqa: E402
from lerobot.utils.utils import get_safe_torch_device  # noqa: E402

from rtc_pi05.action_chunk_queue import ActionChunkQueue  # noqa: E402
from rtc_pi05.actor_loop import run_actor_loop  # noqa: E402
from rtc_pi05.checkpoint_loader import (  # noqa: E402
    disable_policy_rtc,
    enable_policy_rtc,
    load_policy_bundle,
    validate_policy_artifacts,
)
from rtc_pi05.config import RuntimeConfig  # noqa: E402
from rtc_pi05.inference_worker import run_inference_worker  # noqa: E402
from rtc_pi05.metrics import RuntimeMetrics  # noqa: E402
from rtc_pi05.observation_buffer import ObservationBuffer  # noqa: E402
from rtc_pi05.robot_builder import build_dataset_artifacts, build_robot_config  # noqa: E402
from rtc_pi05.robot_io import SerializedRobotIO  # noqa: E402
from rtc_pi05.runtime_state import RuntimeState  # noqa: E402
from rtc_pi05.safety import ActionSafety  # noqa: E402

DEFAULT_POLICY_PATH = (
    "/data/cqy_workspace/flexible_lerobot/outputs/pi05_eraser_cup_multi_task_runs/"
    "20260602_200955/checkpoints/last/pretrained_model"
)
DEFAULT_TASK = "Put the eraser into the small box"
DEFAULT_ROBOT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00"
DEFAULT_CALIB_DIR = "/home/cqy/.cache/huggingface/lerobot/calibration/robots/so_follower"
LOG_PREFIX = "[RTC-PI05]"
KNOWN_NON_FOLLOWER_PORTS = {
    "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123582-if00": "known leader/main-arm port in this setup",
}


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
        description="Local async PI0.5 SO101 inference runtime with optional Real-Time Chunking."
    )
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
        "--top-cam",
        type=parse_camera,
        default=parse_camera(os.getenv("TOP_CAM", os.getenv("TOP_CAM_INDEX", "/dev/video4"))),
    )
    parser.add_argument(
        "--wrist-cam",
        type=parse_camera,
        default=parse_camera(os.getenv("WRIST_CAM", os.getenv("WRIST_CAM_INDEX", "/dev/video6"))),
    )
    parser.add_argument("--top-cam-fourcc", default=os.getenv("TOP_CAM_FOURCC", "YUYV"))
    parser.add_argument("--wrist-cam-fourcc", default=os.getenv("WRIST_CAM_FOURCC", "MJPG"))
    parser.add_argument("--img-width", type=int, default=int(os.getenv("IMG_WIDTH", "640")))
    parser.add_argument("--img-height", type=int, default=int(os.getenv("IMG_HEIGHT", "480")))
    parser.add_argument("--fps", type=int, default=int(os.getenv("FPS", "30")))

    parser.add_argument("--policy-path", default=os.getenv("POLICY_PATH", DEFAULT_POLICY_PATH))
    parser.add_argument("--task", default=os.getenv("DATASET_TASK", DEFAULT_TASK))
    parser.add_argument("--run-time-s", type=float, default=float(os.getenv("RUN_TIME_S", "0")))
    parser.add_argument("--dry-run", type=parse_bool, nargs="?", const=True, default=env_bool("DRY_RUN", False))
    parser.add_argument(
        "--check-policy-load",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("CHECK_POLICY_LOAD", False),
    )
    parser.add_argument(
        "--connect-smoke",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("CONNECT_SMOKE", False),
        help="Connect robot, read one observation, disconnect. Does not load policy unless --check-policy-load is also true.",
    )
    parser.add_argument(
        "--assume-calibrated",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("ASSUME_CALIBRATED", False),
        help="Auto-accept an existing SOFollower calibration file if connect() prompts for it.",
    )
    parser.add_argument(
        "--allow-known-non-follower-port",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("ALLOW_KNOWN_NON_FOLLOWER_PORT", False),
        help="Safety override for ports known not to be the follower arm.",
    )
    parser.add_argument("--device", default=os.getenv("DEVICE"))
    parser.add_argument(
        "--strict-so101-features",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("STRICT_SO101_FEATURES", True),
    )

    parser.add_argument("--queue-low-watermark", type=int, default=int(os.getenv("QUEUE_LOW_WATERMARK", "8")))
    parser.add_argument("--queue-target-size", type=int, default=int(os.getenv("QUEUE_TARGET_SIZE", "24")))
    parser.add_argument("--max-queue-size", type=int, default=int(os.getenv("MAX_QUEUE_SIZE", "50")))
    parser.add_argument(
        "--first-chunk-timeout-s",
        type=float,
        default=float(os.getenv("FIRST_CHUNK_TIMEOUT_S", "30.0")),
    )
    parser.add_argument("--enable-rtc", type=parse_bool, nargs="?", const=True, default=env_bool("ENABLE_RTC", True))
    parser.add_argument("--rtc-execution-horizon", type=int, default=int(os.getenv("RTC_EXECUTION_HORIZON", "10")))
    parser.add_argument(
        "--rtc-max-guidance-weight",
        type=float,
        default=float(os.getenv("RTC_MAX_GUIDANCE_WEIGHT", "10.0")),
    )
    parser.add_argument("--rtc-prefix-attention-schedule", default=os.getenv("RTC_PREFIX_ATTENTION_SCHEDULE", "LINEAR"))
    parser.add_argument("--rtc-debug", type=parse_bool, nargs="?", const=True, default=env_bool("RTC_DEBUG", False))
    parser.add_argument("--rtc-debug-maxlen", type=int, default=int(os.getenv("RTC_DEBUG_MAXLEN", "100")))
    parser.add_argument("--empty-queue-strategy", default=os.getenv("EMPTY_QUEUE_STRATEGY", "hold-last-action"))
    parser.add_argument("--max-action-delta", type=optional_float, default=optional_float(os.getenv("MAX_ACTION_DELTA")))
    parser.add_argument(
        "--metrics-log-interval-s",
        type=float,
        default=float(os.getenv("METRICS_LOG_INTERVAL_S", "5.0")),
    )
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "my_devs/train/pi/so101/rtc_pi05/outputs"))
    return parser


def build_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    schedule_name = str(args.rtc_prefix_attention_schedule).upper()
    try:
        schedule = RTCAttentionSchedule[schedule_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported RTC prefix attention schedule: {args.rtc_prefix_attention_schedule}") from exc
    return RuntimeConfig(
        policy_path=Path(args.policy_path).expanduser(),
        task=args.task,
        fps=args.fps,
        run_time_s=args.run_time_s,
        queue_low_watermark=args.queue_low_watermark,
        queue_target_size=args.queue_target_size,
        max_queue_size=args.max_queue_size,
        first_chunk_timeout_s=args.first_chunk_timeout_s,
        enable_rtc=args.enable_rtc,
        rtc_execution_horizon=args.rtc_execution_horizon,
        rtc_max_guidance_weight=args.rtc_max_guidance_weight,
        rtc_prefix_attention_schedule=schedule,
        rtc_debug=args.rtc_debug,
        rtc_debug_maxlen=args.rtc_debug_maxlen,
        empty_queue_strategy=args.empty_queue_strategy,
        max_action_delta=args.max_action_delta,
        metrics_log_interval_s=args.metrics_log_interval_s,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.chdir(REPO_ROOT)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = build_runtime_config(args)
    validate_robot_port(args.robot_port, allow_known_non_follower=args.allow_known_non_follower_port)
    print_resolved_config(args, config)
    validate_policy_artifacts(config.policy_path, strict_so101_features=args.strict_so101_features)

    robot_cfg = build_robot_config(
        robot_id=args.robot_id,
        robot_type=args.robot_type,
        calib_dir=args.calib_dir,
        robot_port=args.robot_port,
        max_relative_target=args.max_relative_target,
        top_cam=args.top_cam,
        wrist_cam=args.wrist_cam,
        top_cam_fourcc=args.top_cam_fourcc,
        wrist_cam_fourcc=args.wrist_cam_fourcc,
        img_width=args.img_width,
        img_height=args.img_height,
        fps=args.fps,
    )

    if args.dry_run:
        print(f"{LOG_PREFIX} DRY_RUN passed: config and checkpoint artifacts are valid.")
        return 0

    bundle = None
    if args.check_policy_load or not args.connect_smoke:
        bundle = load_policy_bundle(
            config.policy_path,
            repo_root=REPO_ROOT,
            device_override=args.device,
            strict_so101_features=args.strict_so101_features,
        )
        if config.enable_rtc:
            enable_policy_rtc(bundle.policy, config.build_rtc_config())
        else:
            disable_policy_rtc(bundle.policy)
        print(
            f"{LOG_PREFIX} Policy load passed: type={bundle.policy_config.type} "
            f"device={bundle.policy_config.device} rtc={config.enable_rtc}"
        )
        if args.check_policy_load and not args.connect_smoke:
            return 0

    from lerobot.robots import make_robot_from_config

    robot = make_robot_from_config(robot_cfg)
    try:
        with maybe_auto_accept_calibration(args.assume_calibrated):
            robot.connect()
        print(f"{LOG_PREFIX} Robot connected: type={robot.robot_type}")
        if args.connect_smoke:
            obs = robot.get_observation()
            print(f"{LOG_PREFIX} CONNECT_SMOKE observation keys: {sorted(obs.keys())}")
            print(f"{LOG_PREFIX} CONNECT_SMOKE passed.")
            return 0

        assert bundle is not None
        dataset_features, robot_action_processor, robot_observation_processor = build_dataset_artifacts(robot)
        run_runtime(
            config=config,
            robot=robot,
            bundle=bundle,
            dataset_features=dataset_features,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            output_dir=Path(args.output_dir).expanduser(),
        )
        return 0
    finally:
        if getattr(robot, "is_connected", False):
            robot.disconnect()
            print(f"{LOG_PREFIX} Robot disconnected.")


def run_runtime(
    *,
    config: RuntimeConfig,
    robot: Any,
    bundle: Any,
    dataset_features: dict[str, dict[str, Any]],
    robot_action_processor: Any,
    robot_observation_processor: Any,
    output_dir: Path,
) -> None:
    policy = bundle.policy
    reset_if_available(policy)
    reset_if_available(bundle.preprocessor)
    reset_if_available(bundle.postprocessor)

    state = RuntimeState()
    metrics = RuntimeMetrics()
    action_queue = ActionChunkQueue(
        empty_queue_strategy=config.empty_queue_strategy,
        max_queue_size=config.max_queue_size,
    )
    observation_buffer = ObservationBuffer()
    robot_io = SerializedRobotIO(robot)
    safety = ActionSafety(max_action_delta=config.max_action_delta)
    device = get_safe_torch_device(policy.config.device)

    actor_thread = threading.Thread(
        target=run_actor_loop,
        kwargs={
            "config": config,
            "state": state,
            "metrics": metrics,
            "action_queue": action_queue,
            "observation_buffer": observation_buffer,
            "robot_io": robot_io,
            "dataset_features": dataset_features,
            "robot_action_processor": robot_action_processor,
            "safety": safety,
        },
        name="RTCPI05Actor",
        daemon=True,
    )
    inference_thread = threading.Thread(
        target=run_inference_worker,
        kwargs={
            "config": config,
            "state": state,
            "metrics": metrics,
            "action_queue": action_queue,
            "observation_buffer": observation_buffer,
            "dataset_features": dataset_features,
            "policy": policy,
            "preprocessor": bundle.preprocessor,
            "postprocessor": bundle.postprocessor,
            "robot_observation_processor": robot_observation_processor,
            "robot_type": robot.robot_type,
            "device": device,
        },
        name="RTCPI05Inference",
        daemon=True,
    )

    actor_thread.start()
    inference_thread.start()
    last_log_s = time.perf_counter()
    try:
        while state.running:
            time.sleep(0.1)
            now_s = time.perf_counter()
            if now_s - last_log_s >= config.metrics_log_interval_s:
                print_metrics(metrics, action_queue)
                last_log_s = now_s
    except KeyboardInterrupt:
        state.request_stop("KeyboardInterrupt")
        print(f"{LOG_PREFIX} KeyboardInterrupt received.")
    finally:
        state.request_stop(state.stop_reason or "runtime exiting")
        actor_thread.join(timeout=5.0)
        inference_thread.join(timeout=5.0)
        print_metrics(metrics, action_queue)
        write_summary(output_dir, config, state, metrics, action_queue)
        if actor_thread.is_alive() or inference_thread.is_alive():
            raise RuntimeError(
                "Runtime threads did not exit cleanly: "
                f"actor_alive={actor_thread.is_alive()} inference_alive={inference_thread.is_alive()}"
            )
        if state.last_error is not None:
            raise RuntimeError(state.last_error.traceback_text)


def reset_if_available(obj: Any) -> None:
    reset = getattr(obj, "reset", None)
    if callable(reset):
        reset()


def validate_robot_port(robot_port: str, *, allow_known_non_follower: bool = False) -> None:
    reason = KNOWN_NON_FOLLOWER_PORTS.get(robot_port)
    if reason is None or allow_known_non_follower:
        return
    raise ValueError(
        f"Refusing to use robot-port={robot_port!r}: {reason}. "
        f"For SO101 follower inference use {DEFAULT_ROBOT_PORT!r}. "
        "If you are intentionally debugging this port, pass --allow-known-non-follower-port true."
    )


@contextmanager
def maybe_auto_accept_calibration(enabled: bool):
    if not enabled:
        yield
        return
    original_input = builtins.input

    def _auto_enter(prompt: str = "") -> str:
        if prompt:
            print(prompt)
        print(f"{LOG_PREFIX} --assume-calibrated=true, using existing calibration file.")
        return ""

    builtins.input = _auto_enter
    try:
        yield
    finally:
        builtins.input = original_input


def print_resolved_config(args: argparse.Namespace, config: RuntimeConfig) -> None:
    print(f"{LOG_PREFIX} Repo root: {REPO_ROOT}")
    print(f"{LOG_PREFIX} Policy path: {config.policy_path}")
    print(f"{LOG_PREFIX} Robot id: {args.robot_id}")
    print(f"{LOG_PREFIX} Robot type: {args.robot_type}")
    print(f"{LOG_PREFIX} Robot port: {args.robot_port}")
    print(f"{LOG_PREFIX} Top camera: {args.top_cam} fourcc={args.top_cam_fourcc}")
    print(f"{LOG_PREFIX} Wrist camera: {args.wrist_cam} fourcc={args.wrist_cam_fourcc}")
    print(f"{LOG_PREFIX} Task: {config.task}")
    print(f"{LOG_PREFIX} fps={config.fps} run_time_s={config.run_time_s} enable_rtc={config.enable_rtc}")
    print(
        f"{LOG_PREFIX} queue low/target/max="
        f"{config.queue_low_watermark}/{config.queue_target_size}/{config.max_queue_size}"
    )


def print_metrics(metrics: RuntimeMetrics, action_queue: ActionChunkQueue) -> None:
    snapshot = metrics.snapshot()
    queue_snapshot = action_queue.snapshot()
    print(
        f"{LOG_PREFIX} metrics actor_ticks={snapshot.actor_ticks} "
        f"inferences={snapshot.inference_count} queue_depth={queue_snapshot.depth} "
        f"latest_infer_ms={snapshot.latest_inference_s * 1000:.1f} "
        f"p95_infer_ms={snapshot.p95_inference_s * 1000:.1f} "
        f"drop_steps={snapshot.latest_drop_steps} "
        f"pred_delay={snapshot.latest_predicted_delay_steps} "
        f"empty={queue_snapshot.empty_events} hold={queue_snapshot.hold_last_events} "
        f"dropped_all={queue_snapshot.dropped_all_chunks}"
    )


def write_summary(
    output_dir: Path,
    config: RuntimeConfig,
    state: RuntimeState,
    metrics: RuntimeMetrics,
    action_queue: ActionChunkQueue,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"runtime_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    metrics_snapshot = metrics.snapshot()
    queue_snapshot = action_queue.snapshot()
    summary = {
        "config": {
            "policy_path": str(config.policy_path),
            "task": config.task,
            "fps": config.fps,
            "run_time_s": config.run_time_s,
            "enable_rtc": config.enable_rtc,
            "rtc_execution_horizon": config.rtc_execution_horizon,
            "rtc_prefix_attention_schedule": config.rtc_prefix_attention_schedule.value,
        },
        "state": {
            "running": state.running,
            "stop_reason": state.stop_reason,
            "actor_iterations": state.actor_iterations,
            "inference_iterations": state.inference_iterations,
            "sent_actions": state.sent_actions,
            "last_error": None if state.last_error is None else state.last_error.message,
        },
        "metrics": asdict(metrics_snapshot),
        "queue": asdict(queue_snapshot),
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{LOG_PREFIX} Summary written: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
