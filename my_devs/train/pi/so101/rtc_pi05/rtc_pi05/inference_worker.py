from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import OBS_STR

from .action_chunk_queue import ActionChunk, ActionChunkQueue, MergeResult
from .config import RuntimeConfig
from .metrics import RuntimeMetrics
from .observation_buffer import ObservationBuffer, ObservationSnapshot
from .runtime_state import RuntimeState


@dataclass(frozen=True, slots=True)
class InferenceResult:
    merge_result: MergeResult
    total_s: float
    drop_steps: int
    predicted_delay_steps: int
    cursor_delta_steps: int | None
    observation_sequence_id: int


def prepare_policy_batch(
    *,
    observation: Mapping[str, Any],
    dataset_features: dict[str, dict[str, Any]],
    robot_observation_processor: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
    device: torch.device,
    task: str,
    robot_type: str,
) -> dict[str, Any]:
    processed_observation = (
        dict(robot_observation_processor(observation)) if robot_observation_processor else dict(observation)
    )
    observation_frame = build_dataset_frame(dataset_features, processed_observation, prefix=OBS_STR)
    return prepare_observation_for_inference(
        dict(observation_frame),
        device=device,
        task=task,
        robot_type=robot_type,
    )


def postprocess_action_chunk(postprocessor: Callable[[Tensor], Tensor], raw_chunk: Tensor) -> Tensor:
    raw_chunk = ensure_chunk_batch(raw_chunk)
    try:
        return ensure_chunk_batch(postprocessor(raw_chunk))
    except Exception as direct_error:
        batch_size, chunk_len, action_dim = raw_chunk.shape
        flattened = raw_chunk.reshape(batch_size * chunk_len, action_dim)
        try:
            processed = postprocessor(flattened)
        except Exception:
            raise direct_error
        return ensure_chunk_batch(processed).reshape(batch_size, chunk_len, action_dim)


def run_inference_once(
    *,
    config: RuntimeConfig,
    state: RuntimeState,
    metrics: RuntimeMetrics,
    action_queue: ActionChunkQueue,
    observation_buffer: ObservationBuffer,
    dataset_features: dict[str, dict[str, Any]],
    policy: Any,
    preprocessor: Callable[[dict[str, Any]], dict[str, Any]],
    postprocessor: Callable[[Tensor], Tensor],
    robot_observation_processor: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
    robot_type: str,
    device: torch.device,
    wait_timeout_s: float | None = None,
    perf_counter: Callable[[], float] = time.perf_counter,
) -> InferenceResult | None:
    if action_queue.depth() > config.queue_low_watermark:
        return None

    obs_snapshot = observation_buffer.latest(timeout_s=wait_timeout_s)
    if obs_snapshot is None:
        return None

    cursor_before = action_queue.action_cursor()
    predicted_delay_steps = metrics.predicted_delay_steps(control_dt_s=config.control_dt_s)
    raw_leftover = action_queue.get_raw_leftover()
    total_start_s = perf_counter()

    batch = prepare_policy_batch(
        observation=obs_snapshot.observation,
        dataset_features=dataset_features,
        robot_observation_processor=robot_observation_processor,
        device=device,
        task=config.task,
        robot_type=robot_type,
    )
    preprocessed_batch = preprocessor(batch)

    predict_kwargs: dict[str, Any] = {}
    if config.enable_rtc:
        predict_kwargs = {
            "inference_delay": predicted_delay_steps,
            "prev_chunk_left_over": raw_leftover,
            "execution_horizon": config.rtc_execution_horizon,
        }

    use_amp = bool(getattr(getattr(policy, "config", None), "use_amp", False))
    with torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext():
        raw_chunk = ensure_chunk_batch(policy.predict_action_chunk(preprocessed_batch, **predict_kwargs))
        processed_chunk = postprocess_action_chunk(postprocessor, raw_chunk)

    ready_s = perf_counter()
    total_s = max(0.0, ready_s - total_start_s)
    drop_steps = latency_to_steps(ready_s - obs_snapshot.timestamp_s, config.control_dt_s)
    raw_actions = raw_chunk.detach().clone().squeeze(0).to("cpu")
    processed_actions = processed_chunk.detach().clone().squeeze(0).to("cpu")
    chunk = ActionChunk(
        raw_actions=raw_actions,
        processed_actions=processed_actions,
        obs_timestamp_s=obs_snapshot.timestamp_s,
        ready_timestamp_s=ready_s,
        drop_steps=drop_steps,
        rtc_inference_delay=predicted_delay_steps,
        source_observation_seq=obs_snapshot.sequence_id,
    )

    cursor_after = action_queue.action_cursor()
    cursor_delta_steps = cursor_after - cursor_before
    merge_result = action_queue.merge_rtc(chunk) if config.enable_rtc else action_queue.merge_plain(chunk)
    state.note_inference_iteration(
        dropped_all=merge_result.dropped_all,
        drop_all_limit=config.drop_all_chunk_limit,
    )
    if merge_result.enqueued_steps > 0:
        state.mark_first_chunk_ready()
    metrics.record_inference(
        total_s=total_s,
        queue_depth=merge_result.queue_depth_after,
        drop_steps=drop_steps,
        predicted_delay_steps=predicted_delay_steps,
        cursor_delta_steps=cursor_delta_steps,
        dropped_all=merge_result.dropped_all,
    )
    return InferenceResult(
        merge_result=merge_result,
        total_s=total_s,
        drop_steps=drop_steps,
        predicted_delay_steps=predicted_delay_steps,
        cursor_delta_steps=cursor_delta_steps,
        observation_sequence_id=obs_snapshot.sequence_id,
    )


def run_inference_worker(
    *,
    config: RuntimeConfig,
    state: RuntimeState,
    metrics: RuntimeMetrics,
    action_queue: ActionChunkQueue,
    observation_buffer: ObservationBuffer,
    dataset_features: dict[str, dict[str, Any]],
    policy: Any,
    preprocessor: Callable[[dict[str, Any]], dict[str, Any]],
    postprocessor: Callable[[Tensor], Tensor],
    robot_observation_processor: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
    robot_type: str,
    device: torch.device,
    perf_counter: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    iterations = 0
    while state.running:
        try:
            result = run_inference_once(
                config=config,
                state=state,
                metrics=metrics,
                action_queue=action_queue,
                observation_buffer=observation_buffer,
                dataset_features=dataset_features,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                robot_observation_processor=robot_observation_processor,
                robot_type=robot_type,
                device=device,
                wait_timeout_s=config.first_chunk_timeout_s if iterations == 0 else 0.0,
                perf_counter=perf_counter,
            )
        except Exception as exc:
            state.record_exception("inference_worker", exc)
            break
        if result is not None:
            iterations += 1
            continue
        if config.inference_idle_sleep_s > 0:
            sleep_fn(config.inference_idle_sleep_s)
    return iterations


def ensure_chunk_batch(actions: Tensor) -> Tensor:
    if actions.ndim == 2:
        return actions.unsqueeze(0)
    if actions.ndim == 3:
        if actions.shape[0] != 1:
            raise ValueError(f"Only batch_size=1 is supported, got {tuple(actions.shape)}")
        return actions
    raise ValueError(f"Expected action chunk with shape (T,D) or (1,T,D), got {tuple(actions.shape)}")


def latency_to_steps(latency_s: float, control_dt_s: float) -> int:
    if latency_s <= 0:
        return 0
    return int(math.ceil(float(latency_s) / float(control_dt_s)))
