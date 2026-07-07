from __future__ import annotations

import copy
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.utils import make_robot_action
from lerobot.utils.constants import OBS_STR
from lerobot.utils.robot_utils import precise_sleep

from my_devs.train.pi.so101.rtc_pi05.rtc_pi05.action_chunk_queue import (
    ActionChunk,
    ActionChunkQueue,
)
from my_devs.train.pi.so101.rtc_pi05.rtc_pi05.robot_io import SerializedRobotIO
from my_devs.train.pi.so101.rtc_pi05.rtc_pi05.safety import ActionSafety

from .protocol import InferenceRequest
from .remote_policy_client import RemotePolicyClient


@dataclass(slots=True)
class ClientRuntimeConfig:
    task: str
    server_url: str = "http://127.0.0.1:8088"
    camera_fps: int = 30
    control_fps: int = 30
    run_time_s: float = 0.0
    queue_low_watermark: int = 4
    queue_target_size: int = 12
    max_queue_size: int = 50
    first_chunk_timeout_s: float = 60.0
    enable_rtc: bool = True
    rtc_execution_horizon: int = 10
    empty_queue_strategy: str = "hold-last-action"
    max_action_delta: float | None = None
    metrics_log_interval_s: float = 2.0
    request_timeout_s: float = 120.0
    producer_idle_sleep_s: float = 0.005

    def __post_init__(self) -> None:
        if self.camera_fps <= 0:
            raise ValueError("camera_fps must be positive")
        if self.control_fps <= 0:
            raise ValueError("control_fps must be positive")
        if self.run_time_s < 0:
            raise ValueError("run_time_s must be non-negative")
        if self.queue_low_watermark < 0:
            raise ValueError("queue_low_watermark must be non-negative")
        if self.queue_target_size <= 0:
            raise ValueError("queue_target_size must be positive")
        if self.max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")

    @property
    def camera_dt_s(self) -> float:
        return 1.0 / self.camera_fps

    @property
    def control_dt_s(self) -> float:
        return 1.0 / self.control_fps


@dataclass(frozen=True, slots=True)
class FrameSnapshot:
    raw_observation: dict[str, Any]
    observation_frame: dict[str, Any]
    timestamp_s: float
    sequence_id: int


class FrameBuffer:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._snapshot: FrameSnapshot | None = None
        self._sequence_id = 0

    def update(
        self,
        *,
        raw_observation: Mapping[str, Any],
        observation_frame: Mapping[str, Any],
        timestamp_s: float,
    ) -> FrameSnapshot:
        with self._condition:
            self._sequence_id += 1
            snapshot = FrameSnapshot(
                raw_observation=copy.deepcopy(dict(raw_observation)),
                observation_frame=copy.deepcopy(dict(observation_frame)),
                timestamp_s=float(timestamp_s),
                sequence_id=self._sequence_id,
            )
            self._snapshot = snapshot
            self._condition.notify_all()
            return self._copy(snapshot)

    def latest(self, *, timeout_s: float | None = None) -> FrameSnapshot | None:
        with self._condition:
            if self._snapshot is None:
                self._condition.wait_for(lambda: self._snapshot is not None, timeout=timeout_s)
            if self._snapshot is None:
                return None
            return self._copy(self._snapshot)

    @staticmethod
    def _copy(snapshot: FrameSnapshot) -> FrameSnapshot:
        return FrameSnapshot(
            raw_observation=copy.deepcopy(snapshot.raw_observation),
            observation_frame=copy.deepcopy(snapshot.observation_frame),
            timestamp_s=snapshot.timestamp_s,
            sequence_id=snapshot.sequence_id,
        )


class ClientRuntimeState:
    def __init__(self) -> None:
        self.start_time_s = time.perf_counter()
        self.stop_event = threading.Event()
        self.first_chunk_ready = threading.Event()
        self.stop_reason: str | None = None
        self.last_error: str | None = None
        self.sensor_ticks = 0
        self.actor_ticks = 0
        self.sent_actions = 0
        self.inference_requests = 0
        self._lock = threading.RLock()

    @property
    def running(self) -> bool:
        return not self.stop_event.is_set()

    def request_stop(self, reason: str) -> None:
        with self._lock:
            if self.stop_event.is_set():
                return
            self.stop_reason = reason
            self.stop_event.set()
            self.first_chunk_ready.set()

    def record_error(self, source: str, exc: BaseException) -> None:
        with self._lock:
            self.last_error = f"{source}: {exc}"
        self.request_stop(self.last_error)


class ClientMetrics:
    def __init__(self, *, maxlen: int = 100) -> None:
        self._lock = threading.RLock()
        self.request_latencies = deque(maxlen=maxlen)
        self.latest_drop_steps = 0
        self.latest_predicted_delay_steps = 0
        self.latest_server_latency_s = 0.0

    def record_request(
        self,
        *,
        total_s: float,
        server_latency_s: float,
        drop_steps: int,
        predicted_delay_steps: int,
    ) -> None:
        with self._lock:
            self.request_latencies.append(max(0.0, float(total_s)))
            self.latest_server_latency_s = max(0.0, float(server_latency_s))
            self.latest_drop_steps = int(drop_steps)
            self.latest_predicted_delay_steps = int(predicted_delay_steps)

    def predicted_delay_steps(self, *, control_dt_s: float) -> int:
        with self._lock:
            if not self.request_latencies or control_dt_s <= 0:
                return 0
            values = sorted(self.request_latencies)
            p95 = values[min(len(values) - 1, int(round(0.95 * (len(values) - 1))))]
        return int(math.ceil(p95 / control_dt_s)) if p95 > 0 else 0

    def latest_request_s(self) -> float:
        with self._lock:
            return self.request_latencies[-1] if self.request_latencies else 0.0


def run_sensor_loop(
    *,
    config: ClientRuntimeConfig,
    state: ClientRuntimeState,
    robot_io: SerializedRobotIO,
    frame_buffer: FrameBuffer,
    dataset_features: dict[str, dict[str, Any]],
    robot_observation_processor: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
    perf_counter: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = precise_sleep,
) -> int:
    ticks = 0
    next_tick_s = perf_counter()
    while state.running:
        try:
            raw_observation = robot_io.get_observation()
            processed_observation = (
                dict(robot_observation_processor(raw_observation))
                if robot_observation_processor
                else dict(raw_observation)
            )
            observation_frame = build_dataset_frame(dataset_features, processed_observation, prefix=OBS_STR)
            frame_buffer.update(
                raw_observation=raw_observation,
                observation_frame=observation_frame,
                timestamp_s=perf_counter(),
            )
        except Exception as exc:
            state.record_error("sensor_loop", exc)
            break
        ticks += 1
        with state._lock:
            state.sensor_ticks += 1
        next_tick_s += config.camera_dt_s
        sleep_s = next_tick_s - perf_counter()
        if sleep_s > 0:
            sleep_fn(sleep_s)
        else:
            next_tick_s = perf_counter()
    return ticks


def run_actor_loop(
    *,
    config: ClientRuntimeConfig,
    state: ClientRuntimeState,
    robot_io: SerializedRobotIO,
    frame_buffer: FrameBuffer,
    action_queue: ActionChunkQueue,
    dataset_features: dict[str, dict[str, Any]],
    robot_action_processor: Callable[[tuple[dict[str, float], Any]], Any] | None,
    safety: ActionSafety,
    perf_counter: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = precise_sleep,
) -> int:
    ticks = 0
    next_tick_s = perf_counter()
    first_chunk_deadline_s = state.start_time_s + config.first_chunk_timeout_s
    while state.running:
        now_s = perf_counter()
        if config.run_time_s > 0 and now_s - state.start_time_s >= config.run_time_s:
            state.request_stop(f"run_time_s elapsed: {config.run_time_s:.3f}s")
            break
        if not state.first_chunk_ready.is_set() and now_s >= first_chunk_deadline_s:
            state.request_stop(f"first chunk timeout after {config.first_chunk_timeout_s:.3f}s")
            break

        sent = False
        try:
            if state.first_chunk_ready.is_set():
                action = action_queue.pop_processed_action()
                if action is not None:
                    frame_snapshot = frame_buffer.latest(timeout_s=0.0)
                    raw_observation = {} if frame_snapshot is None else frame_snapshot.raw_observation
                    robot_action = build_robot_action(
                        action=action,
                        dataset_features=dataset_features,
                        robot_action_processor=robot_action_processor,
                        observation=raw_observation,
                    )
                    safety.check_tensor(action)
                    safety.check_robot_action(robot_action)
                    robot_io.send_action(robot_action)
                    sent = True
        except Exception as exc:
            state.record_error("actor_loop", exc)
            break
        ticks += 1
        with state._lock:
            state.actor_ticks += 1
            if sent:
                state.sent_actions += 1
        next_tick_s += config.control_dt_s
        sleep_s = next_tick_s - perf_counter()
        if sleep_s > 0:
            sleep_fn(sleep_s)
        else:
            next_tick_s = perf_counter()
    return ticks


def run_producer_loop(
    *,
    config: ClientRuntimeConfig,
    state: ClientRuntimeState,
    metrics: ClientMetrics,
    remote_policy: RemotePolicyClient,
    frame_buffer: FrameBuffer,
    action_queue: ActionChunkQueue,
    robot_type: str,
    perf_counter: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    requests = 0
    request_id = 0
    while state.running:
        if action_queue.depth() > config.queue_low_watermark:
            sleep_fn(config.producer_idle_sleep_s)
            continue

        frame_snapshot = frame_buffer.latest(timeout_s=config.first_chunk_timeout_s if requests == 0 else 0.0)
        if frame_snapshot is None:
            sleep_fn(config.producer_idle_sleep_s)
            continue

        request_id += 1
        predicted_delay_steps = metrics.predicted_delay_steps(control_dt_s=config.control_dt_s)
        left_over = action_queue.get_raw_leftover()
        request = InferenceRequest(
            request_id=request_id,
            observation_frame=frame_snapshot.observation_frame,
            task=config.task,
            robot_type=robot_type,
            obs_timestamp_s=frame_snapshot.timestamp_s,
            obs_sequence_id=frame_snapshot.sequence_id,
            enable_rtc=config.enable_rtc,
            predicted_delay_steps=predicted_delay_steps,
            prev_chunk_left_over=None if left_over is None else left_over.detach().cpu().numpy(),
            execution_horizon=config.rtc_execution_horizon,
        )

        started_s = perf_counter()
        try:
            response = remote_policy.infer(request)
        except Exception as exc:
            state.record_error("producer_loop", exc)
            break
        ready_s = perf_counter()
        total_s = max(0.0, ready_s - started_s)
        drop_steps = latency_to_steps(ready_s - frame_snapshot.timestamp_s, config.control_dt_s)
        raw_actions = torch.as_tensor(response.raw_actions, dtype=torch.float32)
        processed_actions = torch.as_tensor(response.processed_actions, dtype=torch.float32)
        chunk = ActionChunk(
            raw_actions=raw_actions,
            processed_actions=processed_actions,
            obs_timestamp_s=frame_snapshot.timestamp_s,
            ready_timestamp_s=ready_s,
            drop_steps=drop_steps,
            rtc_inference_delay=predicted_delay_steps,
            source_observation_seq=frame_snapshot.sequence_id,
        )
        merge_result = action_queue.merge_rtc(chunk) if config.enable_rtc else action_queue.merge_plain(chunk)
        if merge_result.enqueued_steps > 0:
            state.first_chunk_ready.set()
        metrics.record_request(
            total_s=total_s,
            server_latency_s=response.server_latency_s,
            drop_steps=drop_steps,
            predicted_delay_steps=predicted_delay_steps,
        )
        requests += 1
        with state._lock:
            state.inference_requests += 1
    return requests


def build_robot_action(
    *,
    action: Tensor,
    dataset_features: dict[str, dict[str, Any]],
    robot_action_processor: Callable[[tuple[dict[str, float], Any]], Any] | None,
    observation: Any,
) -> Any:
    robot_action_dict = make_robot_action(action.unsqueeze(0), dataset_features)
    if robot_action_processor is None:
        return robot_action_dict
    return robot_action_processor((robot_action_dict, observation))


def latency_to_steps(latency_s: float, control_dt_s: float) -> int:
    if latency_s <= 0:
        return 0
    return int(math.ceil(float(latency_s) / float(control_dt_s)))
