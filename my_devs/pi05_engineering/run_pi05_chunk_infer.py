#!/usr/bin/env python

from __future__ import annotations

import argparse
import importlib
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from my_devs.pi05_engineering.runtime import common as runtime_common
from my_devs.pi05_engineering.runtime.metrics import PI05RuntimeMetrics
from my_devs.pi05_engineering.runtime.queue_controller import QueueController
from my_devs.pi05_engineering.runtime.robot_io import SerializedRobotIO
from my_devs.pi05_engineering.runtime.runtime_config import PI05RuntimeConfig
from my_devs.pi05_engineering.runtime.runtime_state import PI05RuntimeState


DEFAULT_POLICY_PATH = (
    "/data/cqy_workspace/flexible_lerobot/outputs/train/"
    "pi05_grasp_block_in_bin1_repro_20260303_170727/bs8_20260303_170727/"
    "checkpoints/last/pretrained_model"
)
DEFAULT_CALIB_DIR = "/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower"
ENTRY_LOG_PREFIX = "[PI05-CHUNK]"
WORKER_JOIN_TIMEOUT_S = 2.0


@dataclass(slots=True)
class RuntimeEntrypointContext:
    repo_root: Path
    args: argparse.Namespace
    runtime_config: PI05RuntimeConfig
    robot_cfg: Any
    policy_path: Path
    policy_config: Any | None = None
    robot: Any | None = None
    policy_artifacts: Any | None = None
    dataset_artifacts: Any | None = None
    queue_controller: QueueController | None = None
    metrics: PI05RuntimeMetrics | None = None
    state: PI05RuntimeState | None = None
    threads: list[Any] = field(default_factory=list)
    worker_shutdown_clean: bool | None = None
    alive_worker_names: list[str] = field(default_factory=list)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 1 local PI05 chunk runtime entrypoint for SO101/SO100 follower robots."
    )

    parser.add_argument("--robot-id", default=os.getenv("ROBOT_ID", "my_so101"))
    parser.add_argument("--robot-type", default=os.getenv("ROBOT_TYPE", "so101_follower"))
    parser.add_argument("--calib-dir", default=os.getenv("CALIB_DIR", DEFAULT_CALIB_DIR))
    parser.add_argument("--robot-port", default=os.getenv("ROBOT_PORT", "/dev/ttyACM0"))

    parser.add_argument("--top-cam-index", type=int, default=int(os.getenv("TOP_CAM_INDEX", "4")))
    parser.add_argument("--wrist-cam-index", type=int, default=int(os.getenv("WRIST_CAM_INDEX", "6")))
    parser.add_argument("--img-width", type=int, default=int(os.getenv("IMG_WIDTH", "640")))
    parser.add_argument("--img-height", type=int, default=int(os.getenv("IMG_HEIGHT", "480")))
    parser.add_argument("--fps", type=int, default=int(os.getenv("FPS", "30")))
    parser.add_argument("--camera-fps", type=int, default=int(os.getenv("CAMERA_FPS", "0")) or None)

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
        help="Runtime duration in seconds. <=0 means run until stop/interrupt.",
    )
    parser.add_argument(
        "--dry-run",
        type=runtime_common.parse_bool,
        nargs="?",
        const=True,
        default=runtime_common.env_bool("DRY_RUN", False),
        help="Validate CLI/config/path wiring and exit without connecting hardware.",
    )
    parser.add_argument(
        "--offline-load-smoke",
        type=runtime_common.parse_bool,
        nargs="?",
        const=True,
        default=runtime_common.env_bool("OFFLINE_LOAD_SMOKE", False),
        help="Load policy/processors offline without building or connecting the robot.",
    )
    parser.add_argument(
        "--connect-smoke",
        type=runtime_common.parse_bool,
        nargs="?",
        const=True,
        default=runtime_common.env_bool("CONNECT_SMOKE", False),
        help="Build the robot and run connect/disconnect only without loading policy or starting runtime threads.",
    )
    parser.add_argument(
        "--load-connect-smoke",
        type=runtime_common.parse_bool,
        nargs="?",
        const=True,
        default=runtime_common.env_bool("LOAD_CONNECT_SMOKE", False),
        help="Load policy/processors, then run robot connect/disconnect only without starting runtime threads.",
    )

    parser.add_argument(
        "--queue-low-watermark",
        type=int,
        default=int(os.getenv("QUEUE_LOW_WATERMARK", "0")),
    )
    parser.add_argument(
        "--actions-per-chunk",
        type=int,
        default=int(os.getenv("ACTIONS_PER_CHUNK", "8")),
    )
    parser.add_argument(
        "--enable-rtc",
        type=runtime_common.parse_bool,
        nargs="?",
        const=True,
        default=runtime_common.env_bool("ENABLE_RTC", False),
    )
    parser.add_argument(
        "--metrics-log-interval",
        type=float,
        default=float(os.getenv("METRICS_LOG_INTERVAL", "5.0")),
    )
    parser.add_argument(
        "--startup-wait-for-first-chunk",
        type=runtime_common.parse_bool,
        nargs="?",
        const=True,
        default=runtime_common.env_bool("STARTUP_WAIT_FOR_FIRST_CHUNK", True),
    )
    parser.add_argument(
        "--empty-queue-strategy",
        default=os.getenv("EMPTY_QUEUE_STRATEGY", "hold-last-action"),
    )
    parser.add_argument(
        "--max-action-delta",
        type=float,
        default=float(os.getenv("MAX_ACTION_DELTA", "0")) or None,
    )

    parser.add_argument(
        "--queue-max-size",
        type=int,
        default=int(os.getenv("QUEUE_MAX_SIZE", "0")) or None,
    )
    parser.add_argument(
        "--startup-timeout-s",
        type=float,
        default=float(os.getenv("STARTUP_TIMEOUT_S", "15.0")),
    )
    parser.add_argument(
        "--latency-window-size",
        type=int,
        default=int(os.getenv("LATENCY_WINDOW_SIZE", "100")),
    )

    parser.add_argument(
        "--rtc-execution-horizon",
        type=int,
        default=int(os.getenv("RTC_EXECUTION_HORIZON", "10")),
    )
    parser.add_argument(
        "--rtc-max-guidance-weight",
        type=float,
        default=float(os.getenv("RTC_MAX_GUIDANCE_WEIGHT", "10.0")),
    )
    parser.add_argument(
        "--rtc-prefix-attention-schedule",
        default=os.getenv("RTC_PREFIX_ATTENTION_SCHEDULE", "LINEAR"),
        help="RTCAttentionSchedule enum name, e.g. LINEAR or EXP.",
    )
    parser.add_argument(
        "--rtc-debug",
        type=runtime_common.parse_bool,
        nargs="?",
        const=True,
        default=runtime_common.env_bool("RTC_DEBUG", False),
    )
    parser.add_argument(
        "--rtc-debug-maxlen",
        type=int,
        default=int(os.getenv("RTC_DEBUG_MAXLEN", "100")),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_repo_root_for_entry() -> Path:
    return runtime_common.resolve_repo_root(Path(__file__))


def build_runtime_config(args: argparse.Namespace) -> PI05RuntimeConfig:
    schedule_name = str(args.rtc_prefix_attention_schedule).upper()
    try:
        schedule = importlib.import_module("lerobot.configs.types").RTCAttentionSchedule[schedule_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported rtc_prefix_attention_schedule={args.rtc_prefix_attention_schedule!r}") from exc

    return PI05RuntimeConfig(
        policy_path=args.policy_path,
        fps=args.fps,
        run_time_s=args.run_time_s,
        dry_run=args.dry_run,
        actions_per_chunk=args.actions_per_chunk,
        queue_low_watermark=args.queue_low_watermark,
        queue_max_size=args.queue_max_size,
        startup_wait_for_first_chunk=args.startup_wait_for_first_chunk,
        startup_timeout_s=args.startup_timeout_s,
        empty_queue_strategy=args.empty_queue_strategy,
        max_action_delta=args.max_action_delta,
        enable_rtc=args.enable_rtc,
        rtc_execution_horizon=args.rtc_execution_horizon,
        rtc_max_guidance_weight=args.rtc_max_guidance_weight,
        rtc_prefix_attention_schedule=schedule,
        rtc_debug=args.rtc_debug,
        rtc_debug_maxlen=args.rtc_debug_maxlen,
        metrics_log_interval=args.metrics_log_interval,
        latency_window_size=args.latency_window_size,
    )


def build_runtime_context(args: argparse.Namespace) -> RuntimeEntrypointContext:
    repo_root = resolve_repo_root_for_entry()
    runtime_config = build_runtime_config(args)
    policy_path = runtime_common.resolve_policy_path(args.policy_path)
    args.camera_fps = args.fps if getattr(args, "camera_fps", None) is None else int(args.camera_fps)
    robot_cfg = runtime_common.build_so101_robot_config(
        robot_id=args.robot_id,
        robot_type=args.robot_type,
        calib_dir=args.calib_dir,
        robot_port=args.robot_port,
        top_cam_index=args.top_cam_index,
        wrist_cam_index=args.wrist_cam_index,
        img_width=args.img_width,
        img_height=args.img_height,
        camera_fps=args.camera_fps,
    )
    return RuntimeEntrypointContext(
        repo_root=repo_root,
        args=args,
        runtime_config=runtime_config,
        robot_cfg=robot_cfg,
        policy_path=policy_path,
    )


def run_dry_run(context: RuntimeEntrypointContext) -> RuntimeEntrypointContext:
    context.policy_config = runtime_common.load_policy_config(context.policy_path)
    print(f"{ENTRY_LOG_PREFIX} Repo root: {context.repo_root}")
    print(f"{ENTRY_LOG_PREFIX} Policy path: {context.policy_path}")
    print(f"{ENTRY_LOG_PREFIX} Policy type: {context.policy_config.type}")
    print(f"{ENTRY_LOG_PREFIX} Task: {context.args.task}")
    print(f"{ENTRY_LOG_PREFIX} control_fps: {context.runtime_config.fps}")
    print(f"{ENTRY_LOG_PREFIX} camera_fps: {context.args.camera_fps}")
    print(f"{ENTRY_LOG_PREFIX} run_time_s: {context.runtime_config.run_time_s}")
    print(f"{ENTRY_LOG_PREFIX} enable_rtc: {context.runtime_config.enable_rtc}")
    print(f"{ENTRY_LOG_PREFIX} queue_low_watermark: {context.runtime_config.queue_low_watermark}")
    print(f"{ENTRY_LOG_PREFIX} actions_per_chunk: {context.runtime_config.actions_per_chunk}")
    print(
        f"{ENTRY_LOG_PREFIX} DRY_RUN only validates CLI/config/path wiring and safe exit. "
        "It does not connect hardware or prove predict_action_chunk is wired."
    )
    return context


def run_offline_load_smoke(context: RuntimeEntrypointContext) -> RuntimeEntrypointContext:
    context.policy_artifacts = runtime_common.run_offline_load_smoke(
        context.policy_path,
        repo_root=context.repo_root,
        strict=False,
    )
    context.policy_config = context.policy_artifacts.policy_config
    configure_policy_for_runtime(context)
    rtc_enabled = getattr(getattr(context.policy_artifacts.policy, "config", None), "rtc_config", None)
    rtc_enabled = bool(getattr(rtc_enabled, "enabled", False))

    print(f"{ENTRY_LOG_PREFIX} Repo root: {context.repo_root}")
    print(f"{ENTRY_LOG_PREFIX} Policy path: {context.policy_path}")
    print(f"{ENTRY_LOG_PREFIX} Policy type: {context.policy_config.type}")
    print(f"{ENTRY_LOG_PREFIX} enable_rtc: {context.runtime_config.enable_rtc}")
    print(f"{ENTRY_LOG_PREFIX} control_fps: {context.runtime_config.fps}")
    print(f"{ENTRY_LOG_PREFIX} camera_fps: {context.args.camera_fps}")
    print(f"{ENTRY_LOG_PREFIX} policy.config.rtc_config.enabled: {rtc_enabled}")
    print(
        f"{ENTRY_LOG_PREFIX} OFFLINE_LOAD_SMOKE validates policy/processors loading only. "
        "It does not build the robot or call robot.connect()."
    )
    return context


def run_connect_smoke(context: RuntimeEntrypointContext) -> RuntimeEntrypointContext:
    os.chdir(context.repo_root)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    robot_factory = get_robot_factory()
    context.robot = SerializedRobotIO(robot_factory(context.robot_cfg))

    print(f"{ENTRY_LOG_PREFIX} Repo root: {context.repo_root}")
    print(f"{ENTRY_LOG_PREFIX} Robot port: {context.args.robot_port}")
    print(f"{ENTRY_LOG_PREFIX} control_fps: {context.runtime_config.fps}")
    print(f"{ENTRY_LOG_PREFIX} camera_fps: {context.args.camera_fps}")
    print(
        f"{ENTRY_LOG_PREFIX} CONNECT_SMOKE validates robot.connect()/disconnect() only. "
        "It does not load policy weights or start producer/actor threads."
    )

    connect = getattr(context.robot, "connect", None)
    if not callable(connect):
        raise RuntimeError("Robot object does not expose a connect() method.")

    try:
        connect()
    finally:
        _disconnect_robot_if_needed(context.robot)

    return context


def run_load_connect_smoke(context: RuntimeEntrypointContext) -> RuntimeEntrypointContext:
    os.chdir(context.repo_root)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    robot_factory = get_robot_factory()
    context.robot = SerializedRobotIO(robot_factory(context.robot_cfg))
    context.policy_artifacts = runtime_common.load_policy_and_processors(
        context.policy_path,
        repo_root=context.repo_root,
        strict=False,
    )
    context.policy_config = context.policy_artifacts.policy_config
    configure_policy_for_runtime(context)

    print(f"{ENTRY_LOG_PREFIX} Repo root: {context.repo_root}")
    print(f"{ENTRY_LOG_PREFIX} Policy path: {context.policy_path}")
    print(f"{ENTRY_LOG_PREFIX} Robot port: {context.args.robot_port}")
    print(f"{ENTRY_LOG_PREFIX} enable_rtc: {context.runtime_config.enable_rtc}")
    print(f"{ENTRY_LOG_PREFIX} control_fps: {context.runtime_config.fps}")
    print(f"{ENTRY_LOG_PREFIX} camera_fps: {context.args.camera_fps}")
    print(
        f"{ENTRY_LOG_PREFIX} LOAD_CONNECT_SMOKE validates policy/processors loading plus "
        "robot.connect()/disconnect() only. It does not start producer/actor threads."
    )

    connect = getattr(context.robot, "connect", None)
    if not callable(connect):
        raise RuntimeError("Robot object does not expose a connect() method.")

    try:
        connect()
    finally:
        _disconnect_robot_if_needed(context.robot)

    return context


def get_robot_factory() -> Callable[[Any], Any]:
    from lerobot.robots import make_robot_from_config

    return make_robot_from_config


def resolve_loop_targets() -> tuple[Callable[..., Any], Callable[..., Any]]:
    return (
        _resolve_loop_target(
            module_name="my_devs.pi05_engineering.runtime.producer_loop",
            candidate_names=("run_producer_loop", "producer_loop", "main"),
            loop_label="producer",
        ),
        _resolve_loop_target(
            module_name="my_devs.pi05_engineering.runtime.actor_loop",
            candidate_names=("run_actor_loop", "actor_loop", "main"),
            loop_label="actor",
        ),
    )


def _resolve_loop_target(
    *,
    module_name: str,
    candidate_names: tuple[str, ...],
    loop_label: str,
) -> Callable[..., Any]:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return _missing_loop_target(loop_label)

    for name in candidate_names:
        target = getattr(module, name, None)
        if callable(target):
            return target
    return _missing_loop_target(loop_label)


def _missing_loop_target(loop_label: str) -> Callable[..., Any]:
    def _runner(**_kwargs: Any) -> None:
        raise RuntimeError(f"{loop_label} loop module is not implemented yet.")

    return _runner


def build_queue_controller(runtime_config: PI05RuntimeConfig) -> QueueController:
    return QueueController(
        enable_rtc=runtime_config.enable_rtc,
        empty_queue_strategy=runtime_config.empty_queue_strategy,
        rtc_execution_horizon=runtime_config.rtc_execution_horizon,
        rtc_max_guidance_weight=runtime_config.rtc_max_guidance_weight,
    )


def build_metrics(runtime_config: PI05RuntimeConfig) -> PI05RuntimeMetrics:
    return PI05RuntimeMetrics(window_size=runtime_config.latency_window_size)


def build_state() -> PI05RuntimeState:
    return PI05RuntimeState()


def configure_policy_for_runtime(context: RuntimeEntrypointContext) -> None:
    if context.policy_artifacts is None:
        raise RuntimeError("policy_artifacts must be loaded before policy runtime configuration.")

    policy = context.policy_artifacts.policy
    if context.runtime_config.enable_rtc:
        policy.config.rtc_config = context.runtime_config.build_rtc_config()
        init_rtc_processor = getattr(policy, "init_rtc_processor", None)
        if callable(init_rtc_processor):
            init_rtc_processor()
    else:
        policy.config.rtc_config = None

    for artifact in (policy, context.policy_artifacts.preprocessor, context.policy_artifacts.postprocessor):
        reset = getattr(artifact, "reset", None)
        if callable(reset):
            reset()


def build_producer_target(
    context: RuntimeEntrypointContext,
    producer_runner: Callable[..., Any] | None = None,
) -> Callable[[RuntimeEntrypointContext], None]:
    resolved_runner = resolve_loop_targets()[0] if producer_runner is None else producer_runner

    if context.state is None or context.metrics is None or context.queue_controller is None:
        raise RuntimeError("Producer target requires state/metrics/queue_controller to be initialized.")
    if context.robot is None or context.dataset_artifacts is None or context.policy_artifacts is None:
        raise RuntimeError("Producer target requires robot, dataset artifacts, and policy artifacts.")

    get_observation = getattr(context.robot, "get_observation", None)
    if not callable(get_observation):
        raise RuntimeError("Robot object does not expose a get_observation() method.")

    dataset_features = context.dataset_artifacts.dataset_features
    robot_observation_processor = getattr(context.dataset_artifacts, "robot_observation_processor", None)
    policy = context.policy_artifacts.policy
    preprocessor = context.policy_artifacts.preprocessor
    postprocessor = context.policy_artifacts.postprocessor
    robot_type = getattr(context.robot, "robot_type", context.args.robot_type)

    def _producer_target(_: RuntimeEntrypointContext) -> None:
        resolved_runner(
            config=context.runtime_config,
            state=context.state,
            metrics=context.metrics,
            queue_controller=context.queue_controller,
            observation_provider=get_observation,
            dataset_features=dataset_features,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            task=context.args.task,
            robot_type=robot_type,
            robot_observation_processor=robot_observation_processor,
        )

    return _producer_target


def build_actor_target(
    context: RuntimeEntrypointContext,
    actor_runner: Callable[..., Any] | None = None,
) -> Callable[[RuntimeEntrypointContext], None]:
    resolved_runner = resolve_loop_targets()[1] if actor_runner is None else actor_runner

    if context.state is None or context.metrics is None or context.queue_controller is None:
        raise RuntimeError("Actor target requires state/metrics/queue_controller to be initialized.")
    if context.robot is None or context.dataset_artifacts is None:
        raise RuntimeError("Actor target requires robot and dataset artifacts.")

    send_action = getattr(context.robot, "send_action", None)
    if not callable(send_action):
        raise RuntimeError("Robot object does not expose a send_action() method.")

    dataset_features = context.dataset_artifacts.dataset_features
    robot_action_processor = getattr(context.dataset_artifacts, "robot_action_processor", None)

    def _actor_target(_: RuntimeEntrypointContext) -> None:
        resolved_runner(
            config=context.runtime_config,
            runtime_state=context.state,
            metrics=context.metrics,
            queue_controller=context.queue_controller,
            dataset_features=dataset_features,
            send_action=send_action,
            robot_action_processor=robot_action_processor,
            observation_for_processor=None,
        )

    return _actor_target


def _build_worker_thread(
    *,
    name: str,
    target: Callable[[RuntimeEntrypointContext], None],
    kwargs: dict[str, Any],
    daemon: bool = True,
) -> threading.Thread:
    return threading.Thread(
        target=_thread_entrypoint,
        kwargs={"name": name, "target": target, "target_kwargs": kwargs},
        name=name,
        daemon=daemon,
    )


def _thread_entrypoint(
    *,
    name: str,
    target: Callable[[RuntimeEntrypointContext], None],
    target_kwargs: dict[str, Any],
) -> None:
    context = target_kwargs["context"]
    try:
        target(context)
    except BaseException as exc:  # noqa: BLE001
        if context.state is not None:
            context.state.record_exception(name, exc)
        else:
            raise


def start_runtime_threads(
    context: RuntimeEntrypointContext,
    *,
    producer_target: Callable[[RuntimeEntrypointContext], None],
    actor_target: Callable[[RuntimeEntrypointContext], None],
) -> list[threading.Thread]:
    threads: list[threading.Thread] = []
    worker_specs = (
        ("PI05Producer", producer_target),
        ("PI05Actor", actor_target),
    )
    for name, target in worker_specs:
        threads.append(_build_worker_thread(name=name, target=target, kwargs={"context": context}, daemon=True))
    for thread in threads:
        thread.start()
    return threads


def _wait_for_first_chunk_if_needed(context: RuntimeEntrypointContext) -> None:
    if context.state is None or not context.runtime_config.startup_wait_for_first_chunk:
        return

    if context.state.first_chunk_ready_event.is_set():
        return

    deadline = time.perf_counter() + context.runtime_config.startup_timeout_s
    while context.state.running and not context.state.first_chunk_ready_event.is_set():
        remaining_s = deadline - time.perf_counter()
        if remaining_s <= 0:
            timeout_message = (
                "Timed out waiting for the first chunk. "
                f"startup_timeout_s={context.runtime_config.startup_timeout_s}"
            )
            context.state.request_stop(timeout_message)
            raise TimeoutError(timeout_message)
        context.state.first_chunk_ready_event.wait(timeout=min(remaining_s, 0.1))


def _run_until_stopped(context: RuntimeEntrypointContext) -> None:
    assert context.state is not None
    assert context.metrics is not None

    end_time = (
        time.perf_counter() + context.runtime_config.run_time_s
        if context.runtime_config.should_stop_by_duration
        else None
    )
    next_metrics_log_at = time.perf_counter() + context.runtime_config.metrics_log_interval

    while context.state.running:
        now = time.perf_counter()
        if end_time is not None and now >= end_time:
            context.state.request_stop("run_time_s reached")
            break

        if now >= next_metrics_log_at:
            print(f"{ENTRY_LOG_PREFIX} Metrics: {_build_observability_log_dict(context, now_s=now)}")
            next_metrics_log_at = now + context.runtime_config.metrics_log_interval

        time.sleep(min(context.runtime_config.metrics_log_interval, 0.1))


def _build_observability_log_dict(
    context: RuntimeEntrypointContext,
    *,
    now_s: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if context.metrics is not None:
        payload.update(context.metrics.as_log_dict(now_s=now_s))
    if context.state is not None:
        state_snapshot = context.state.snapshot()
        payload.update(
            {
                "state_running": state_snapshot.running,
                "state_stop_reason": state_snapshot.stop_reason,
                "state_first_chunk_ready": state_snapshot.first_chunk_ready,
                "state_producer_iterations": state_snapshot.producer_iterations,
                "state_actor_iterations": state_snapshot.actor_iterations,
                "state_last_action_index_delta": state_snapshot.last_action_index_delta,
                "state_last_inference_delay": state_snapshot.last_inference_delay,
                "state_last_real_delay": state_snapshot.last_real_delay,
                "state_last_merge_mode": state_snapshot.last_merge_mode,
                "state_last_prev_chunk_left_over_length": state_snapshot.last_prev_chunk_left_over_length,
                "state_last_original_actions_length": state_snapshot.last_original_actions_length,
                "state_last_processed_actions_length": state_snapshot.last_processed_actions_length,
                "state_last_trimmed_prefix_steps": state_snapshot.last_trimmed_prefix_steps,
                "state_last_enqueued_steps": state_snapshot.last_enqueued_steps,
            }
        )
    if context.queue_controller is not None:
        queue_counters = context.queue_controller.counters()
        payload.update(
            {
                "queue_empty_events": queue_counters.empty_queue_events,
                "queue_held_last_action_events": queue_counters.held_last_action_events,
                "queue_skip_send_events": queue_counters.skip_send_events,
                "queue_actions_popped": queue_counters.actions_popped,
            }
        )
    payload.update(
        {
            "worker_shutdown_clean": context.worker_shutdown_clean,
            "alive_worker_names": list(context.alive_worker_names),
        }
    )
    return payload


def _disconnect_robot_if_needed(robot: Any | None) -> None:
    if robot is None:
        return
    disconnect = getattr(robot, "disconnect", None)
    if not callable(disconnect):
        return

    is_connected = getattr(robot, "is_connected", None)
    if is_connected is False:
        return
    disconnect()


def _join_worker_threads(
    threads: list[Any],
    *,
    timeout_s: float = WORKER_JOIN_TIMEOUT_S,
) -> list[str]:
    for thread in threads:
        thread.join(timeout=timeout_s)

    alive_worker_names: list[str] = []
    for thread in threads:
        is_alive = getattr(thread, "is_alive", None)
        if not callable(is_alive):
            continue
        if is_alive():
            alive_worker_names.append(getattr(thread, "name", repr(thread)))
    return alive_worker_names


def _mark_runtime_started(context: RuntimeEntrypointContext, *, timestamp_s: float | None = None) -> float:
    started_at_s = time.perf_counter() if timestamp_s is None else float(timestamp_s)
    if context.state is not None:
        context.state.rebase_start_time(started_at_s)
    if context.metrics is not None:
        context.metrics.rebase_start_time(started_at_s)
    return started_at_s


def run_live(context: RuntimeEntrypointContext) -> RuntimeEntrypointContext:
    os.chdir(context.repo_root)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    pending_error: BaseException | None = None
    shutdown_error: RuntimeError | None = None
    producer_runner, actor_runner = resolve_loop_targets()
    context.state = build_state()
    context.metrics = build_metrics(context.runtime_config)
    context.queue_controller = build_queue_controller(context.runtime_config)

    robot_factory = get_robot_factory()
    context.robot = SerializedRobotIO(robot_factory(context.robot_cfg))
    context.dataset_artifacts = runtime_common.build_dataset_features(
        action_features=context.robot.action_features,
        observation_features=context.robot.observation_features,
    )
    context.policy_artifacts = runtime_common.load_policy_and_processors(
        context.policy_path,
        repo_root=context.repo_root,
        strict=False,
    )
    context.policy_config = context.policy_artifacts.policy_config
    configure_policy_for_runtime(context)
    producer_target = build_producer_target(context, producer_runner=producer_runner)
    actor_target = build_actor_target(context, actor_runner=actor_runner)

    connect = getattr(context.robot, "connect", None)
    if not callable(connect):
        raise RuntimeError("Robot object does not expose a connect() method.")

    try:
        connect()
        _mark_runtime_started(context)
        context.threads = start_runtime_threads(
            context,
            producer_target=producer_target,
            actor_target=actor_target,
        )
        _wait_for_first_chunk_if_needed(context)
        _run_until_stopped(context)
    except KeyboardInterrupt:
        context.state.request_stop("KeyboardInterrupt")
    except BaseException as exc:  # noqa: BLE001
        pending_error = exc
        context.state.record_exception("entrypoint", exc)
    finally:
        if context.state is not None and context.state.running:
            context.state.request_stop("runtime cleanup")

        context.alive_worker_names = _join_worker_threads(context.threads)
        context.worker_shutdown_clean = len(context.alive_worker_names) == 0
        if context.alive_worker_names:
            message = (
                "Timed out waiting for worker threads to exit before disconnect: "
                + ", ".join(context.alive_worker_names)
            )
            shutdown_error = RuntimeError(message)
            if context.state is not None and context.state.last_error is None:
                context.state.record_exception("entrypoint.shutdown", shutdown_error)
            print(f"{ENTRY_LOG_PREFIX} {message}")
        else:
            _disconnect_robot_if_needed(context.robot)

        if context.metrics is not None or context.state is not None:
            print(
                f"{ENTRY_LOG_PREFIX} Final metrics: "
                f"{_build_observability_log_dict(context, now_s=time.perf_counter())}"
            )

    if pending_error is not None:
        raise pending_error
    if shutdown_error is not None:
        raise shutdown_error
    return context


def run(argv: list[str] | None = None) -> RuntimeEntrypointContext:
    args = parse_args(argv)
    context = build_runtime_context(args)
    os.chdir(context.repo_root)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if getattr(args, "offline_load_smoke", False):
        return run_offline_load_smoke(context)
    if getattr(args, "connect_smoke", False):
        return run_connect_smoke(context)
    if getattr(args, "load_connect_smoke", False):
        return run_load_connect_smoke(context)
    if context.runtime_config.dry_run:
        return run_dry_run(context)
    return run_live(context)


def main(argv: list[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
