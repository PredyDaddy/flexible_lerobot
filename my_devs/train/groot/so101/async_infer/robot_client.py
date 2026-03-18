from __future__ import annotations

import argparse
import os
import pickle  # nosec
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if SCRIPT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, SCRIPT_DIR.as_posix())

try:
    import grpc
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional async deps
    raise ModuleNotFoundError(
        "grpc is required for async inference. Install it in lerobot_flex with "
        '`python -m pip install -e ".[async]"`.'
    ) from exc

from lerobot.policies.utils import make_robot_action
from lerobot.robots import make_robot_from_config
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep

from shared import (
    DEFAULT_POLICY_PATH,
    DEFAULT_SERVER_ADDRESS,
    FPSTracker,
    RemoteGrootPolicySetup,
    TimedAction,
    TimedObservation,
    build_observation_frame,
    build_robot_config,
    build_robot_runtime_helpers,
    clamp_actions_per_chunk,
    configure_logging,
    env_bool,
    get_aggregate_function,
    parse_bool,
)


@dataclass
class RobotClientConfig:
    robot_id: str
    robot_type: str
    calib_dir: str
    robot_port: str
    top_cam_index: int
    wrist_cam_index: int
    img_width: int
    img_height: int
    fps: int
    policy_path: str
    task: str
    run_time_s: float
    log_interval: int
    server_address: str
    actions_per_chunk: int
    chunk_size_threshold: float
    aggregate_fn_name: str
    backend: str
    trt_engine_path: str | None
    vit_dtype: str
    llm_dtype: str
    dit_dtype: str
    trt_action_head_only: bool
    client_device: str
    robot_connect_retries: int
    robot_connect_retry_delay_s: float
    log_level: str
    dry_run: bool = False


class GrootRobotClient:
    def __init__(self, config: RobotClientConfig, logger):
        self.config = config
        self.logger = logger

        robot_cfg = build_robot_config(argparse.Namespace(**config.__dict__))
        self.robot = make_robot_from_config(robot_cfg)
        self.robot_action_processor, self.robot_observation_processor, self.dataset_features = (
            build_robot_runtime_helpers(self.robot)
        )

        actions_per_chunk = clamp_actions_per_chunk(config.actions_per_chunk, logger)
        self.policy_setup = RemoteGrootPolicySetup(
            policy_path=config.policy_path,
            task=config.task,
            robot_type=config.robot_type,
            actions_per_chunk=actions_per_chunk,
            backend=config.backend,
            trt_engine_path=config.trt_engine_path,
            vit_dtype=config.vit_dtype,
            llm_dtype=config.llm_dtype,
            dit_dtype=config.dit_dtype,
            trt_action_head_only=config.trt_action_head_only,
        )

        initial_backoff = f"{1.0 / float(config.fps):.4f}s"
        self.channel = grpc.insecure_channel(
            config.server_address,
            grpc_channel_options(initial_backoff=initial_backoff),
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

        self.shutdown_event = threading.Event()
        self.start_barrier = threading.Barrier(2)
        self.must_go = threading.Event()
        self.must_go.set()

        self.action_queue: Queue[TimedAction] = Queue()
        self.action_queue_lock = threading.Lock()
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = actions_per_chunk
        self.aggregate_fn = get_aggregate_function(config.aggregate_fn_name)
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        self.executed_actions = 0
        self.last_raw_observation: dict[str, Any] | None = None

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    def start(self) -> bool:
        try:
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            self.stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(self.policy_setup)))
            handshake_time = time.perf_counter() - start_time

            self.logger.info(
                "Policy server is ready at %s | handshake=%.4fs | starting robot connect",
                self.config.server_address,
                handshake_time,
            )
            self._connect_robot_with_retry()

            self.shutdown_event.clear()
            self.logger.info(
                "Async robot client is ready | server=%s | handshake=%.4fs",
                self.config.server_address,
                handshake_time,
            )
            return True
        except Exception as exc:  # pragma: no cover - runtime path
            self.logger.exception("Failed to start robot client: %s", exc)
            try:
                if self.robot.is_connected:
                    self.robot.disconnect()
            except Exception:  # pragma: no cover - best effort shutdown
                self.logger.exception("Failed to disconnect robot after startup error")
            return False

    def _connect_robot_with_retry(self) -> None:
        attempts = max(1, self.config.robot_connect_retries)
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                self.logger.info("Connecting robot | attempt=%s/%s", attempt, attempts)
                self.robot.connect()
                self.logger.info("Robot connected")
                return
            except Exception as exc:
                last_error = exc
                self.logger.warning("Robot connect attempt %s/%s failed: %s", attempt, attempts, exc)

                try:
                    if self.robot.is_connected:
                        self.robot.disconnect()
                except Exception:
                    self.logger.exception("Robot cleanup after failed connect attempt also failed")

                if attempt < attempts:
                    time.sleep(max(0.0, self.config.robot_connect_retry_delay_s))

        assert last_error is not None
        raise last_error

    def stop(self) -> None:
        self.shutdown_event.set()
        try:
            if self.robot.is_connected:
                self.robot.disconnect()
        finally:
            self.channel.close()
        self.logger.info("Robot client stopped")

    def send_observation(self, observation_t: TimedObservation) -> bool:
        if not self.running:
            raise RuntimeError("Client not running. Call start() before sending observations.")

        observation_bytes = pickle.dumps(observation_t)  # nosec
        observation_iterator = send_bytes_in_chunks(
            observation_bytes,
            services_pb2.Observation,
            log_prefix="[CLIENT] Observation",
            silent=True,
        )
        try:
            _ = self.stub.SendObservations(observation_iterator)
            self.logger.debug("Sent observation #%s", observation_t.get_timestep())
            return True
        except grpc.RpcError as exc:
            self.logger.error("Failed to send observation #%s: %s", observation_t.get_timestep(), exc)
            return False

    def _inspect_action_queue(self) -> tuple[int, list[int]]:
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timesteps = sorted(action.get_timestep() for action in self.action_queue.queue)
        return queue_size, timesteps

    def _aggregate_action_queues(self, incoming_actions: list[TimedAction]) -> None:
        future_action_queue: Queue[TimedAction] = Queue()
        with self.action_queue_lock:
            current_queue = {action.get_timestep(): action.get_action() for action in self.action_queue.queue}

        with self.latest_action_lock:
            latest_action = self.latest_action

        for new_action in incoming_actions:
            if new_action.get_timestep() <= latest_action:
                continue

            if new_action.get_timestep() not in current_queue:
                future_action_queue.put(new_action)
                continue

            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=self.aggregate_fn(current_queue[new_action.get_timestep()], new_action.get_action()),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self) -> None:
        self.start_barrier.wait()
        self.logger.info("Action receiver thread started")

        while self.running:
            try:
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue

                receive_time = time.time()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                if not isinstance(timed_actions, list):
                    raise TypeError(f"Expected list[TimedAction], got {type(timed_actions)}")
                if len(timed_actions) == 0:
                    continue

                if self.config.client_device != "cpu":
                    for timed_action in timed_actions:
                        if timed_action.get_action().device.type != self.config.client_device:
                            timed_action.action = timed_action.get_action().to(self.config.client_device)

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))
                self._aggregate_action_queues(timed_actions)
                self.must_go.set()

                timesteps = [action.get_timestep() for action in timed_actions]
                self.logger.debug(
                    "Received action chunk %s:%s | size=%s | server_to_client_latency=%.2fms",
                    timesteps[0],
                    timesteps[-1],
                    len(timed_actions),
                    (receive_time - timed_actions[0].get_timestamp()) * 1000.0,
                )
            except grpc.RpcError as exc:
                if self.running:
                    self.logger.error("Error receiving actions: %s", exc)
            except Exception as exc:  # pragma: no cover - runtime path
                self.logger.exception("Unexpected error in receive_actions: %s", exc)

    def actions_available(self) -> bool:
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def control_loop_action(self) -> None:
        with self.action_queue_lock:
            if self.action_queue.empty():
                return
            timed_action = self.action_queue.get_nowait()

        action_dict = make_robot_action(timed_action.get_action(), self.dataset_features)
        observation_context = self.last_raw_observation if self.last_raw_observation is not None else {}
        robot_action = self.robot_action_processor((action_dict, observation_context))
        _ = self.robot.send_action(robot_action)

        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        self.executed_actions += 1
        if self.config.log_interval > 0 and self.executed_actions % self.config.log_interval == 0:
            self.logger.info("Executed action #%s", timed_action.get_timestep())

    def _ready_to_send_observation(self) -> bool:
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
        return queue_size / float(max(self.action_chunk_size, 1)) <= self.config.chunk_size_threshold

    def control_loop_observation(self) -> dict[str, Any]:
        raw_observation = self.robot.get_observation()
        self.last_raw_observation = raw_observation
        observation_frame = build_observation_frame(
            raw_observation,
            self.dataset_features,
            self.robot_observation_processor,
        )

        with self.latest_action_lock:
            latest_action = self.latest_action

        observation_t = TimedObservation(
            timestamp=time.time(),
            timestep=max(latest_action, 0),
            observation=observation_frame,
        )

        with self.action_queue_lock:
            observation_t.must_go = self.must_go.is_set() and self.action_queue.empty()
            queue_size = self.action_queue.qsize()

        _ = self.send_observation(observation_t)
        if observation_t.must_go:
            self.must_go.clear()

        fps_metrics = self.fps_tracker.calculate_fps_metrics(observation_t.get_timestamp())
        self.logger.debug(
            "Sent observation #%s | must_go=%s | queue_size=%s | avg_fps=%.2f | target_fps=%.2f",
            observation_t.get_timestep(),
            observation_t.must_go,
            queue_size,
            fps_metrics["avg_fps"],
            fps_metrics["target_fps"],
        )
        return raw_observation

    def control_loop(self) -> None:
        self.start_barrier.wait()
        self.logger.info("Control loop started")

        start_t = time.perf_counter()
        end_t = start_t + self.config.run_time_s if self.config.run_time_s > 0 else None

        while self.running:
            if end_t is not None and time.perf_counter() >= end_t:
                self.logger.info("Reached requested run_time_s. Stopping client loop.")
                break

            loop_t = time.perf_counter()

            if self.actions_available():
                self.control_loop_action()

            if self._ready_to_send_observation():
                _ = self.control_loop_observation()

            precise_sleep(max(1.0 / float(self.config.fps) - (time.perf_counter() - loop_t), 0.0))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GR00T async robot client for SO101 real-robot inference.")
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
        help="Language instruction passed to remote policy inference.",
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
        help="Print status every N executed actions.",
    )

    parser.add_argument("--server-address", default=os.getenv("ASYNC_SERVER_ADDRESS", DEFAULT_SERVER_ADDRESS))
    parser.add_argument(
        "--actions-per-chunk",
        type=int,
        default=int(os.getenv("ACTIONS_PER_CHUNK", "16")),
        help="Number of actions requested from the server per chunk. GR00T is capped at 16.",
    )
    parser.add_argument(
        "--chunk-size-threshold",
        type=float,
        default=float(os.getenv("CHUNK_SIZE_THRESHOLD", "0.5")),
        help="Send a new observation when queue_size/actions_per_chunk drops below this threshold.",
    )
    parser.add_argument(
        "--aggregate-fn",
        default=os.getenv("AGGREGATE_FN", "latest_only"),
        choices=["latest_only", "weighted_average", "average", "conservative"],
        help="How overlapping action timesteps from newer chunks replace old queued actions.",
    )
    parser.add_argument("--client-device", default=os.getenv("CLIENT_DEVICE", "cpu"))
    parser.add_argument(
        "--robot-connect-retries",
        type=int,
        default=int(os.getenv("ROBOT_CONNECT_RETRIES", "3")),
        help="Retry count for robot.connect() to handle transient motor packet failures during startup.",
    )
    parser.add_argument(
        "--robot-connect-retry-delay-s",
        type=float,
        default=float(os.getenv("ROBOT_CONNECT_RETRY_DELAY_S", "1.5")),
        help="Sleep between robot.connect() retries in seconds.",
    )

    parser.add_argument(
        "--backend",
        default=os.getenv("INFER_BACKEND", "pytorch"),
        choices=["pytorch", "tensorrt"],
        help="Inference backend used by the remote policy server.",
    )
    parser.add_argument(
        "--trt-engine-path",
        default=os.getenv("TRT_ENGINE_PATH"),
        help="Directory containing GROOT TensorRT engine files when backend=tensorrt.",
    )
    parser.add_argument(
        "--vit-dtype",
        default=os.getenv("TRT_VIT_DTYPE", "fp16"),
        choices=["fp16", "fp8"],
    )
    parser.add_argument(
        "--llm-dtype",
        default=os.getenv("TRT_LLM_DTYPE", "fp16"),
        choices=["fp16", "nvfp4", "fp8", "nvfp4_full"],
    )
    parser.add_argument(
        "--dit-dtype",
        default=os.getenv("TRT_DIT_DTYPE", "fp16"),
        choices=["fp16", "fp8"],
    )
    parser.add_argument(
        "--trt-action-head-only",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("TRT_ACTION_HEAD_ONLY", False),
    )

    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    parser.add_argument(
        "--dry-run",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("DRY_RUN", False),
        help="Print resolved config and exit without connecting robot or server.",
    )
    return parser


def run(config: RobotClientConfig) -> None:
    logger = configure_logging("groot_async_robot_client", config.log_level)
    logger.info(
        "Starting async GR00T robot client | robot_id=%s | robot_type=%s | server=%s | "
        "task=%s | actions_per_chunk=%s | backend=%s",
        config.robot_id,
        config.robot_type,
        config.server_address,
        config.task,
        config.actions_per_chunk,
        config.backend,
    )

    client = GrootRobotClient(config, logger)
    if not client.start():
        raise RuntimeError("Failed to start async robot client")

    action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
    action_receiver_thread.start()

    try:
        client.control_loop()
    finally:
        client.stop()
        action_receiver_thread.join(timeout=2.0)


def main() -> None:
    register_third_party_plugins()
    args = build_parser().parse_args()
    config = RobotClientConfig(
        robot_id=args.robot_id,
        robot_type=args.robot_type,
        calib_dir=args.calib_dir,
        robot_port=args.robot_port,
        top_cam_index=args.top_cam_index,
        wrist_cam_index=args.wrist_cam_index,
        img_width=args.img_width,
        img_height=args.img_height,
        fps=args.fps,
        policy_path=args.policy_path,
        task=args.task,
        run_time_s=args.run_time_s,
        log_interval=args.log_interval,
        server_address=args.server_address,
        actions_per_chunk=args.actions_per_chunk,
        chunk_size_threshold=args.chunk_size_threshold,
        aggregate_fn_name=args.aggregate_fn,
        backend=args.backend,
        trt_engine_path=args.trt_engine_path,
        vit_dtype=args.vit_dtype,
        llm_dtype=args.llm_dtype,
        dit_dtype=args.dit_dtype,
        trt_action_head_only=args.trt_action_head_only,
        client_device=args.client_device,
        robot_connect_retries=args.robot_connect_retries,
        robot_connect_retry_delay_s=args.robot_connect_retry_delay_s,
        log_level=args.log_level,
        dry_run=args.dry_run,
    )
    if config.dry_run:
        print(config)
        return
    run(config)


if __name__ == "__main__":
    main()
