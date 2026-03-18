from __future__ import annotations

import argparse
import os
import pickle  # nosec
import sys
import threading
import time
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
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

import torch

from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks
from lerobot.utils.utils import get_safe_torch_device

from shared import (
    FPSTracker,
    RemoteGrootPolicySetup,
    TimedAction,
    TimedObservation,
    clamp_actions_per_chunk,
    configure_logging,
    env_bool,
    load_groot_policy_bundle,
    observations_similar,
    parse_bool,
)


@dataclass
class PolicyServerConfig:
    host: str
    port: int
    fps: int
    inference_latency: float
    obs_queue_timeout: float
    log_level: str
    dry_run: bool = False


class GrootPolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    def __init__(self, config: PolicyServerConfig, logger):
        self.config = config
        self.logger = logger
        self.shutdown_event = threading.Event()
        self.fps_tracker = FPSTracker(target_fps=config.fps)
        self.observation_queue: Queue[TimedObservation] = Queue(maxsize=1)
        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps: set[int] = set()
        self.last_processed_obs: TimedObservation | None = None

        self.setup: RemoteGrootPolicySetup | None = None
        self.policy_cfg = None
        self.policy = None
        self.policy_device = None
        self.preprocessor = None
        self.postprocessor = None

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    def _reset_runtime_state(self) -> None:
        self.shutdown_event.set()
        self.fps_tracker.reset()
        self.observation_queue = Queue(maxsize=1)
        self.last_processed_obs = None
        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()

        if self.policy is not None:
            self.policy.reset()
        if self.preprocessor is not None:
            self.preprocessor.reset()
        if self.postprocessor is not None:
            self.postprocessor.reset()

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info("Client %s connected and ready", client_id)
        self._reset_runtime_state()
        self.shutdown_event.clear()
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()
        setup = pickle.loads(request.data)  # nosec
        if not isinstance(setup, RemoteGrootPolicySetup):
            raise TypeError(f"Policy setup must be RemoteGrootPolicySetup, got {type(setup)}")

        setup.actions_per_chunk = clamp_actions_per_chunk(setup.actions_per_chunk, self.logger)
        self.logger.info(
            "Loading policy for %s | policy_path=%s | task=%s | robot_type=%s | "
            "actions_per_chunk=%s | backend=%s",
            client_id,
            setup.policy_path,
            setup.task,
            setup.robot_type,
            setup.actions_per_chunk,
            setup.backend,
        )

        start = time.perf_counter()
        self.setup = setup
        self.policy_cfg, self.policy, self.preprocessor, self.postprocessor = load_groot_policy_bundle(setup)
        self.policy_device = get_safe_torch_device(self.policy_cfg.device)
        end = time.perf_counter()

        self.logger.info(
            "Policy ready | device=%s | checkpoint_type=%s | load_time=%.4fs",
            self.policy_cfg.device,
            self.policy_cfg.type,
            end - start,
        )
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        client_id = context.peer()
        receive_time = time.time()
        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(request_iterator, None, self.shutdown_event, self.logger.name)
        if received_bytes is None:
            return services_pb2.Empty()

        timed_observation = pickle.loads(received_bytes)  # nosec
        if not isinstance(timed_observation, TimedObservation):
            raise TypeError(f"Observation payload must be TimedObservation, got {type(timed_observation)}")
        deserialize_time = time.perf_counter() - start_deserialize

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)
        self.logger.debug(
            "Received observation #%s from %s | avg_fps=%.2f | target_fps=%.2f | "
            "one_way_latency=%.2fms | deserialize=%.2fms",
            obs_timestep,
            client_id,
            fps_metrics["avg_fps"],
            fps_metrics["target_fps"],
            (receive_time - obs_timestamp) * 1000.0,
            deserialize_time * 1000.0,
        )

        if not self._enqueue_observation(timed_observation):
            self.logger.debug("Observation #%s filtered out", obs_timestep)

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        if self.setup is None or self.policy is None or self.preprocessor is None or self.postprocessor is None:
            self.logger.debug("GetActions requested before policy setup completed")
            return services_pb2.Empty()

        client_id = context.peer()
        try:
            getactions_started = time.perf_counter()
            observation_t = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            self.logger.info(
                "Running inference for observation #%s from %s (must_go=%s)",
                observation_t.get_timestep(),
                client_id,
                observation_t.must_go,
            )

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(observation_t.get_timestep())

            action_chunk = self._predict_action_chunk(observation_t)
            serialize_start = time.perf_counter()
            actions_bytes = pickle.dumps(action_chunk)  # nosec
            serialize_time = time.perf_counter() - serialize_start

            total_elapsed = time.perf_counter() - getactions_started
            self.logger.info(
                "Action chunk ready for observation #%s | num_actions=%s | total=%.2fms | serialize=%.2fms",
                observation_t.get_timestep(),
                len(action_chunk),
                total_elapsed * 1000.0,
                serialize_time * 1000.0,
            )

            remaining = max(0.0, self.config.inference_latency - total_elapsed)
            if remaining > 0:
                time.sleep(remaining)

            return services_pb2.Actions(data=actions_bytes)

        except Empty:
            return services_pb2.Empty()
        except Exception as exc:  # pragma: no cover - runtime path
            self.logger.exception("Error while serving GetActions: %s", exc)
            return services_pb2.Empty()

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        if obs.get_timestep() in predicted_timesteps:
            self.logger.debug("Skipping observation #%s because timestep was already predicted", obs.get_timestep())
            return False

        if observations_similar(obs, previous_obs):
            self.logger.debug("Skipping observation #%s because state delta is too small", obs.get_timestep())
            return False

        return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        if obs.must_go or self.last_processed_obs is None or self._obs_sanity_checks(obs, self.last_processed_obs):
            if self.observation_queue.full():
                _ = self.observation_queue.get_nowait()
                self.logger.debug("Observation queue full, evicted oldest observation")
            self.observation_queue.put(obs)
            return True
        return False

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        environment_dt = 1.0 / float(self.config.fps)
        return [
            TimedAction(timestamp=t_0 + i * environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        assert self.setup is not None
        assert self.policy is not None
        assert self.preprocessor is not None
        assert self.postprocessor is not None
        assert self.policy_device is not None

        total_start = time.perf_counter()

        start_prepare = time.perf_counter()
        raw_observation = dict(observation_t.get_observation())
        observation = prepare_observation_for_inference(
            raw_observation,
            self.policy_device,
            task=self.setup.task,
            robot_type=self.setup.robot_type,
        )
        prepare_time = time.perf_counter() - start_prepare

        start_preprocess = time.perf_counter()
        observation = self.preprocessor(observation)
        self.last_processed_obs = observation_t
        preprocessing_time = time.perf_counter() - start_preprocess

        start_inference = time.perf_counter()
        with torch.inference_mode():
            action_tensor = self.policy.predict_action_chunk(observation)
        if action_tensor.ndim != 3:
            action_tensor = action_tensor.unsqueeze(0)
        action_tensor = action_tensor[:, : self.setup.actions_per_chunk, :]
        inference_time = time.perf_counter() - start_inference

        start_postprocess = time.perf_counter()
        _, chunk_size, _ = action_tensor.shape
        processed_actions: list[torch.Tensor] = []
        for index in range(chunk_size):
            single_action = action_tensor[:, index, :]
            processed_action = self.postprocessor(single_action)
            processed_actions.append(processed_action.detach().cpu().squeeze(0))
        postprocessing_time = time.perf_counter() - start_postprocess

        total_time = time.perf_counter() - total_start
        self.logger.info(
            "Observation #%s | prepare=%.2fms | preprocess=%.2fms | infer=%.2fms | "
            "postprocess=%.2fms | total=%.2fms",
            observation_t.get_timestep(),
            prepare_time * 1000.0,
            preprocessing_time * 1000.0,
            inference_time * 1000.0,
            postprocessing_time * 1000.0,
            total_time * 1000.0,
        )

        return self._time_action_chunk(
            observation_t.get_timestamp(),
            processed_actions,
            observation_t.get_timestep(),
        )

    def stop(self) -> None:
        self._reset_runtime_state()
        self.logger.info("Policy server stopping")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GR00T async policy server for SO101 real-robot inference.")
    parser.add_argument("--host", default=os.getenv("ASYNC_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("ASYNC_PORT", "8080")))
    parser.add_argument("--fps", type=int, default=int(os.getenv("FPS", "30")))
    parser.add_argument(
        "--inference-latency",
        type=float,
        default=float(os.getenv("ASYNC_INFERENCE_LATENCY", "0.0")),
        help="Optional server-side minimum interval per GetActions cycle in seconds.",
    )
    parser.add_argument(
        "--obs-queue-timeout",
        type=float,
        default=float(os.getenv("ASYNC_OBS_QUEUE_TIMEOUT", "1.0")),
        help="Timeout when waiting for the next observation from the client.",
    )
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    parser.add_argument(
        "--dry-run",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("DRY_RUN", False),
        help="Print resolved config and exit without starting gRPC server.",
    )
    return parser


def serve(config: PolicyServerConfig) -> None:
    logger = configure_logging("groot_async_policy_server", config.log_level)
    logger.info(
        "Starting async GR00T policy server | host=%s | port=%s | fps=%s | inference_latency=%.4fs",
        config.host,
        config.port,
        config.fps,
        config.inference_latency,
    )

    policy_server = GrootPolicyServer(config, logger)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{config.host}:{config.port}")
    server.start()

    try:
        server.wait_for_termination()
    finally:
        policy_server.stop()
        server.stop(grace=0)


def main() -> None:
    args = build_parser().parse_args()
    config = PolicyServerConfig(
        host=args.host,
        port=args.port,
        fps=args.fps,
        inference_latency=args.inference_latency,
        obs_queue_timeout=args.obs_queue_timeout,
        log_level=args.log_level,
        dry_run=args.dry_run,
    )
    if config.dry_run:
        print(config)
        return
    serve(config)


if __name__ == "__main__":
    main()
