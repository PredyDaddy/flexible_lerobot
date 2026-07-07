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


def resolve_repo_root(script_path: Path) -> Path:
    resolved = script_path.resolve()
    for candidate in resolved.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {script_path}")


REPO_ROOT = resolve_repo_root(Path(__file__))
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from my_devs.train.pi.so101.rtc_pi05.rtc_pi05.action_chunk_queue import ActionChunkQueue  # noqa: E402
from my_devs.train.pi.so101.rtc_pi05.rtc_pi05.robot_builder import (  # noqa: E402
    build_dataset_artifacts,
    build_robot_config,
)
from my_devs.train.pi.so101.rtc_pi05.rtc_pi05.robot_io import SerializedRobotIO  # noqa: E402
from my_devs.train.pi.so101.rtc_pi05.rtc_pi05.safety import ActionSafety  # noqa: E402
from my_devs.train.pi.so101.rtc_pi05.server.client_runtime import (  # noqa: E402
    ClientMetrics,
    ClientRuntimeConfig,
    ClientRuntimeState,
    FrameBuffer,
    run_actor_loop,
    run_producer_loop,
    run_sensor_loop,
)
from my_devs.train.pi.so101.rtc_pi05.server.remote_policy_client import RemotePolicyClient  # noqa: E402

DEFAULT_TASK = "Put the eraser into the small box"
DEFAULT_ROBOT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00"
DEFAULT_CALIB_DIR = "/home/cqy/.cache/huggingface/lerobot/calibration/robots/so_follower"
LOG_PREFIX = "[RTC-PI05-CLIENT]"
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
    parser = argparse.ArgumentParser(description="SO101 robot client for PI0.5 RTC policy server.")
    parser.add_argument("--server-url", default=os.getenv("PI05_SERVER_URL", "http://127.0.0.1:8088"))
    parser.add_argument("--robot-id", default=os.getenv("ROBOT_ID", "hfy_follower"))
    parser.add_argument("--robot-type", default=os.getenv("ROBOT_TYPE", "so101_follower"))
    parser.add_argument("--calib-dir", default=os.getenv("CALIB_DIR", DEFAULT_CALIB_DIR))
    parser.add_argument("--robot-port", default=os.getenv("ROBOT_PORT", DEFAULT_ROBOT_PORT))
    parser.add_argument("--max-relative-target", type=optional_float, default=optional_float(os.getenv("MAX_RELATIVE_TARGET")))
    parser.add_argument("--top-cam", type=parse_camera, default=parse_camera(os.getenv("TOP_CAM", "/dev/video4")))
    parser.add_argument("--wrist-cam", type=parse_camera, default=parse_camera(os.getenv("WRIST_CAM", "/dev/video6")))
    parser.add_argument("--top-cam-fourcc", default=os.getenv("TOP_CAM_FOURCC", "YUYV"))
    parser.add_argument("--wrist-cam-fourcc", default=os.getenv("WRIST_CAM_FOURCC", "MJPG"))
    parser.add_argument("--img-width", type=int, default=int(os.getenv("IMG_WIDTH", "640")))
    parser.add_argument("--img-height", type=int, default=int(os.getenv("IMG_HEIGHT", "480")))
    parser.add_argument("--camera-fps", type=int, default=int(os.getenv("CAMERA_FPS", "30")))
    parser.add_argument("--control-fps", type=int, default=int(os.getenv("CONTROL_FPS", "30")))
    parser.add_argument("--task", default=os.getenv("DATASET_TASK", DEFAULT_TASK))
    parser.add_argument("--run-time-s", type=float, default=float(os.getenv("RUN_TIME_S", "0")))
    parser.add_argument("--enable-rtc", type=parse_bool, nargs="?", const=True, default=env_bool("ENABLE_RTC", True))
    parser.add_argument("--queue-low-watermark", type=int, default=int(os.getenv("QUEUE_LOW_WATERMARK", "4")))
    parser.add_argument("--queue-target-size", type=int, default=int(os.getenv("QUEUE_TARGET_SIZE", "12")))
    parser.add_argument("--max-queue-size", type=int, default=int(os.getenv("MAX_QUEUE_SIZE", "50")))
    parser.add_argument("--first-chunk-timeout-s", type=float, default=float(os.getenv("FIRST_CHUNK_TIMEOUT_S", "60")))
    parser.add_argument("--rtc-execution-horizon", type=int, default=int(os.getenv("RTC_EXECUTION_HORIZON", "10")))
    parser.add_argument("--empty-queue-strategy", default=os.getenv("EMPTY_QUEUE_STRATEGY", "hold-last-action"))
    parser.add_argument("--max-action-delta", type=optional_float, default=optional_float(os.getenv("MAX_ACTION_DELTA")))
    parser.add_argument("--request-timeout-s", type=float, default=float(os.getenv("REQUEST_TIMEOUT_S", "120")))
    parser.add_argument("--metrics-log-interval-s", type=float, default=float(os.getenv("METRICS_LOG_INTERVAL_S", "2")))
    parser.add_argument("--assume-calibrated", type=parse_bool, nargs="?", const=True, default=env_bool("ASSUME_CALIBRATED", False))
    parser.add_argument(
        "--allow-known-non-follower-port",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("ALLOW_KNOWN_NON_FOLLOWER_PORT", False),
    )
    parser.add_argument("--connect-smoke", type=parse_bool, nargs="?", const=True, default=env_bool("CONNECT_SMOKE", False))
    parser.add_argument("--dry-run", type=parse_bool, nargs="?", const=True, default=env_bool("DRY_RUN", False))
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "my_devs/train/pi/so101/rtc_pi05/server/outputs"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.chdir(REPO_ROOT)
    validate_robot_port(args.robot_port, allow_known_non_follower=args.allow_known_non_follower_port)
    config = ClientRuntimeConfig(
        task=args.task,
        server_url=args.server_url,
        camera_fps=args.camera_fps,
        control_fps=args.control_fps,
        run_time_s=args.run_time_s,
        queue_low_watermark=args.queue_low_watermark,
        queue_target_size=args.queue_target_size,
        max_queue_size=args.max_queue_size,
        first_chunk_timeout_s=args.first_chunk_timeout_s,
        enable_rtc=args.enable_rtc,
        rtc_execution_horizon=args.rtc_execution_horizon,
        empty_queue_strategy=args.empty_queue_strategy,
        max_action_delta=args.max_action_delta,
        metrics_log_interval_s=args.metrics_log_interval_s,
        request_timeout_s=args.request_timeout_s,
    )
    print_resolved_config(args, config)
    if args.dry_run:
        print(f"{LOG_PREFIX} DRY_RUN passed.")
        return 0

    remote_policy = RemotePolicyClient(config.server_url, timeout_s=config.request_timeout_s)
    print(f"{LOG_PREFIX} Server health: {remote_policy.health()}")

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
        fps=args.camera_fps,
    )
    from lerobot.robots import make_robot_from_config

    robot = make_robot_from_config(robot_cfg)
    try:
        with maybe_auto_accept_calibration(args.assume_calibrated):
            robot.connect()
        print(f"{LOG_PREFIX} Robot connected: type={robot.robot_type}")
        if args.connect_smoke:
            obs = robot.get_observation()
            print(f"{LOG_PREFIX} CONNECT_SMOKE observation keys: {sorted(obs.keys())}")
            return 0

        dataset_features, robot_action_processor, robot_observation_processor = build_dataset_artifacts(robot)
        run_client_runtime(
            config=config,
            remote_policy=remote_policy,
            robot=robot,
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


def run_client_runtime(
    *,
    config: ClientRuntimeConfig,
    remote_policy: RemotePolicyClient,
    robot: object,
    dataset_features: dict[str, dict],
    robot_action_processor: object,
    robot_observation_processor: object,
    output_dir: Path,
) -> None:
    state = ClientRuntimeState()
    metrics = ClientMetrics()
    frame_buffer = FrameBuffer()
    action_queue = ActionChunkQueue(
        empty_queue_strategy=config.empty_queue_strategy,
        max_queue_size=config.max_queue_size,
    )
    robot_io = SerializedRobotIO(robot)
    safety = ActionSafety(max_action_delta=config.max_action_delta)
    threads = [
        threading.Thread(
            target=run_sensor_loop,
            kwargs={
                "config": config,
                "state": state,
                "robot_io": robot_io,
                "frame_buffer": frame_buffer,
                "dataset_features": dataset_features,
                "robot_observation_processor": robot_observation_processor,
            },
            name="RTCPI05SensorClient",
            daemon=True,
        ),
        threading.Thread(
            target=run_actor_loop,
            kwargs={
                "config": config,
                "state": state,
                "robot_io": robot_io,
                "frame_buffer": frame_buffer,
                "action_queue": action_queue,
                "dataset_features": dataset_features,
                "robot_action_processor": robot_action_processor,
                "safety": safety,
            },
            name="RTCPI05ActorClient",
            daemon=True,
        ),
        threading.Thread(
            target=run_producer_loop,
            kwargs={
                "config": config,
                "state": state,
                "metrics": metrics,
                "remote_policy": remote_policy,
                "frame_buffer": frame_buffer,
                "action_queue": action_queue,
                "robot_type": robot.robot_type,
            },
            name="RTCPI05ProducerClient",
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()

    last_log_s = time.perf_counter()
    try:
        while state.running:
            time.sleep(0.1)
            now_s = time.perf_counter()
            if now_s - last_log_s >= config.metrics_log_interval_s:
                print_metrics(state, metrics, action_queue)
                last_log_s = now_s
    except KeyboardInterrupt:
        state.request_stop("KeyboardInterrupt")
    finally:
        state.request_stop(state.stop_reason or "client runtime exiting")
        for thread in threads:
            thread.join(timeout=5.0)
        print_metrics(state, metrics, action_queue)
        write_summary(output_dir, config, state, metrics, action_queue)
        alive = [thread.name for thread in threads if thread.is_alive()]
        if alive:
            raise RuntimeError(f"Client runtime threads did not stop cleanly: {alive}")
        if state.last_error is not None:
            raise RuntimeError(state.last_error)


def validate_robot_port(robot_port: str, *, allow_known_non_follower: bool = False) -> None:
    reason = KNOWN_NON_FOLLOWER_PORTS.get(robot_port)
    if reason is None or allow_known_non_follower:
        return
    raise ValueError(
        f"Refusing to use robot-port={robot_port!r}: {reason}. "
        f"For SO101 follower inference use {DEFAULT_ROBOT_PORT!r}."
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


def print_resolved_config(args: argparse.Namespace, config: ClientRuntimeConfig) -> None:
    print(f"{LOG_PREFIX} Repo root: {REPO_ROOT}")
    print(f"{LOG_PREFIX} Server URL: {config.server_url}")
    print(f"{LOG_PREFIX} Robot port: {args.robot_port}")
    print(f"{LOG_PREFIX} Top camera: {args.top_cam} fourcc={args.top_cam_fourcc}")
    print(f"{LOG_PREFIX} Wrist camera: {args.wrist_cam} fourcc={args.wrist_cam_fourcc}")
    print(f"{LOG_PREFIX} camera_fps={config.camera_fps} control_fps={config.control_fps}")
    print(f"{LOG_PREFIX} run_time_s={config.run_time_s} enable_rtc={config.enable_rtc}")


def print_metrics(state: ClientRuntimeState, metrics: ClientMetrics, action_queue: ActionChunkQueue) -> None:
    queue = action_queue.snapshot()
    print(
        f"{LOG_PREFIX} metrics sensor_ticks={state.sensor_ticks} actor_ticks={state.actor_ticks} "
        f"sent={state.sent_actions} requests={state.inference_requests} queue_depth={queue.depth} "
        f"latest_request_ms={metrics.latest_request_s() * 1000:.1f} "
        f"server_ms={metrics.latest_server_latency_s * 1000:.1f} "
        f"drop_steps={metrics.latest_drop_steps} pred_delay={metrics.latest_predicted_delay_steps} "
        f"empty={queue.empty_events} hold={queue.hold_last_events}"
    )


def write_summary(
    output_dir: Path,
    config: ClientRuntimeConfig,
    state: ClientRuntimeState,
    metrics: ClientMetrics,
    action_queue: ActionChunkQueue,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"client_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        "config": asdict(config),
        "state": {
            "stop_reason": state.stop_reason,
            "last_error": state.last_error,
            "sensor_ticks": state.sensor_ticks,
            "actor_ticks": state.actor_ticks,
            "sent_actions": state.sent_actions,
            "inference_requests": state.inference_requests,
        },
        "metrics": {
            "latest_request_s": metrics.latest_request_s(),
            "latest_server_latency_s": metrics.latest_server_latency_s,
            "latest_drop_steps": metrics.latest_drop_steps,
            "latest_predicted_delay_steps": metrics.latest_predicted_delay_steps,
        },
        "queue": asdict(action_queue.snapshot()),
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{LOG_PREFIX} Summary written: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
