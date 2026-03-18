from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor

from my_devs.pi05_engineering.runtime.chunk_inference import (
    ChunkPolicy,
    build_chunk_observation_frame,
    prepare_policy_batch_from_frame,
)
from my_devs.pi05_engineering.runtime.metrics import PI05RuntimeMetrics
from my_devs.pi05_engineering.runtime.runtime_config import PI05RuntimeConfig
from my_devs.pi05_engineering.runtime.runtime_state import PI05RuntimeState

logger = logging.getLogger(__name__)


class QueueMergeResultLike(Protocol):
    mode: str
    input_steps: int
    trimmed_prefix_steps: int
    enqueued_steps: int
    queue_depth_after: int
    dropped_all: bool


class ProducerQueueController(Protocol):
    def qsize(self) -> int: ...

    def get_action_index(self) -> int: ...

    def get_left_over_original_actions(self) -> Tensor | None: ...

    def merge_plain(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        *,
        stale_prefix_steps: int = 0,
    ) -> QueueMergeResultLike: ...

    def merge_rtc(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        *,
        real_delay: int,
        action_index_before_inference: int | None = 0,
    ) -> QueueMergeResultLike: ...


ObservationProvider = Callable[[], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class ProducerIterationResult:
    refill_required: bool
    inference_ran: bool
    mode: str | None
    queue_depth_before: int
    queue_depth_after: int
    action_index_before_inference: int | None
    action_index_delta: int | None
    inference_delay: int | None
    real_delay: int | None
    original_chunk_length: int
    chunk_length: int
    enqueued_steps: int
    trimmed_prefix_steps: int
    dropped_all: bool
    first_chunk_ready_marked: bool
    total_latency_s: float | None
    preprocess_latency_s: float | None
    model_latency_s: float | None
    postprocess_latency_s: float | None

    @property
    def did_refill(self) -> bool:
        return self.enqueued_steps > 0


@dataclass(frozen=True, slots=True)
class TimedChunkInferenceResult:
    original_actions: Tensor
    processed_actions: Tensor
    total_s: float
    preprocess_s: float
    model_s: float
    postprocess_s: float
    finished_at_s: float


def run_producer_iteration(
    *,
    config: PI05RuntimeConfig,
    state: PI05RuntimeState,
    metrics: PI05RuntimeMetrics,
    queue_controller: ProducerQueueController,
    observation_provider: ObservationProvider,
    dataset_features: Mapping[str, dict[str, Any]],
    policy: ChunkPolicy,
    preprocessor: Callable[[dict[str, Any]], dict[str, Any]],
    postprocessor: Callable[[Tensor], Tensor],
    device: torch.device | str | None = None,
    task: str | None = None,
    robot_type: str | None = None,
    robot_observation_processor: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    perf_counter: Callable[[], float] = time.perf_counter,
) -> ProducerIterationResult:
    """Run one producer-side refill check and optional chunk inference.

    This function is fully offline and hardware-agnostic. The caller provides
    the observation source and all policy/processors explicitly.
    """

    iteration_start_s = perf_counter()
    state.note_producer_iteration(timestamp_s=iteration_start_s)
    metrics.record_producer_iteration(timestamp_s=iteration_start_s)

    queue_depth_before = queue_controller.qsize()
    metrics.record_queue_depth(queue_depth_before)
    refill_required = queue_depth_before <= config.queue_low_watermark
    chunk_step_budget = _resolve_chunk_step_budget(config, queue_depth_before)

    if not refill_required or chunk_step_budget <= 0:
        return ProducerIterationResult(
            refill_required=refill_required,
            inference_ran=False,
            mode=None,
            queue_depth_before=queue_depth_before,
            queue_depth_after=queue_depth_before,
            action_index_before_inference=None,
            action_index_delta=None,
            inference_delay=None,
            real_delay=None,
            original_chunk_length=0,
            chunk_length=0,
            enqueued_steps=0,
            trimmed_prefix_steps=0,
            dropped_all=False,
            first_chunk_ready_marked=False,
            total_latency_s=None,
            preprocess_latency_s=None,
            model_latency_s=None,
            postprocess_latency_s=None,
        )

    action_index_before_inference = queue_controller.get_action_index()
    prev_chunk_left_over = queue_controller.get_left_over_original_actions()
    observation = dict(observation_provider())

    inferred_delay = _latency_to_delay_steps(_latest_latency_estimate_s(metrics), config.step_dt_s)
    rtc_kwargs = _build_rtc_kwargs(
        config=config,
        prev_chunk_left_over=prev_chunk_left_over,
        inference_delay=inferred_delay,
    )
    timed_result = _run_timed_chunk_inference(
        observation=observation,
        dataset_features=dataset_features,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=device,
        task=task,
        robot_type=robot_type,
        robot_observation_processor=robot_observation_processor,
        rtc_kwargs=rtc_kwargs,
        perf_counter=perf_counter,
    )

    original_actions, processed_actions = _prepare_queue_actions(
        timed_result.original_actions,
        timed_result.processed_actions,
        max_steps=chunk_step_budget,
    )
    real_delay = _latency_to_delay_steps(timed_result.total_s, config.step_dt_s)
    action_index_after_inference = queue_controller.get_action_index()
    action_index_delta = _compute_action_index_delta(
        action_index_before_inference=action_index_before_inference,
        action_index_after_inference=action_index_after_inference,
    )
    delay_mismatch = _should_record_delay_mismatch(
        config=config,
        state=state,
        action_index_delta=action_index_delta,
        real_delay=real_delay,
    )
    if delay_mismatch:
        logger.warning(
            "[PI05_PRODUCER] action_index_delta and real_delay diverged: "
            "action_index_before_inference=%s action_index_after_inference=%s "
            "action_index_delta=%s real_delay=%s step_dt_s=%.6f enable_rtc=%s",
            action_index_before_inference,
            action_index_after_inference,
            action_index_delta,
            real_delay,
            config.step_dt_s,
            config.enable_rtc,
        )

    if config.enable_rtc:
        merge_result = queue_controller.merge_rtc(
            original_actions,
            processed_actions,
            real_delay=real_delay,
            action_index_before_inference=action_index_before_inference,
        )
    else:
        merge_result = queue_controller.merge_plain(
            original_actions,
            processed_actions,
            stale_prefix_steps=real_delay,
        )

    if merge_result.trimmed_prefix_steps > 0:
        metrics.record_stale_prefix_trim(merge_result.trimmed_prefix_steps)

    left_over_original = queue_controller.get_left_over_original_actions()
    left_over_length = 0 if left_over_original is None else len(left_over_original)

    state.note_chunk_inference(
        prev_chunk_left_over=prev_chunk_left_over,
        original_actions=original_actions,
        processed_actions=processed_actions,
        action_index_before_inference=action_index_before_inference,
        action_index_delta=action_index_delta,
        inference_delay=inferred_delay,
        real_delay=real_delay,
        merge_mode=merge_result.mode,
        queue_depth_before_merge=queue_depth_before,
        queue_depth_after_merge=merge_result.queue_depth_after,
        trimmed_prefix_steps=merge_result.trimmed_prefix_steps,
        enqueued_steps=merge_result.enqueued_steps,
        finished_at_s=timed_result.finished_at_s,
    )

    metrics.record_inference(
        total_s=timed_result.total_s,
        preprocess_s=timed_result.preprocess_s,
        model_s=timed_result.model_s,
        postprocess_s=timed_result.postprocess_s,
        chunk_length=len(processed_actions),
        leftover_length=left_over_length,
        original_backlog_length=left_over_length,
        executable_backlog_length=merge_result.queue_depth_after,
        action_index_before_inference=action_index_before_inference,
        action_index_delta=action_index_delta,
        inference_delay=inferred_delay,
        real_delay=real_delay,
        delay_mismatch=delay_mismatch,
        queue_depth_before=queue_depth_before,
        queue_depth_after=merge_result.queue_depth_after,
        inference_overrun_ratio=(timed_result.total_s / config.step_dt_s) if config.step_dt_s > 0 else None,
    )

    first_chunk_ready_marked = False
    if merge_result.enqueued_steps > 0:
        metrics.record_refill(
            chunk_length=len(processed_actions),
            queue_depth_after=merge_result.queue_depth_after,
            leftover_length=left_over_length,
            original_backlog_length=left_over_length,
            executable_backlog_length=merge_result.queue_depth_after,
        )
        state_marked = state.mark_first_chunk_ready(timestamp_s=timed_result.finished_at_s)
        metrics_marked = metrics.mark_first_chunk_ready(timestamp_s=timed_result.finished_at_s)
        first_chunk_ready_marked = state_marked or metrics_marked

    return ProducerIterationResult(
        refill_required=True,
        inference_ran=True,
        mode=merge_result.mode,
        queue_depth_before=queue_depth_before,
        queue_depth_after=merge_result.queue_depth_after,
        action_index_before_inference=action_index_before_inference,
        action_index_delta=action_index_delta,
        inference_delay=inferred_delay,
        real_delay=real_delay,
        original_chunk_length=len(original_actions),
        chunk_length=len(processed_actions),
        enqueued_steps=merge_result.enqueued_steps,
        trimmed_prefix_steps=merge_result.trimmed_prefix_steps,
        dropped_all=merge_result.dropped_all,
        first_chunk_ready_marked=first_chunk_ready_marked,
        total_latency_s=timed_result.total_s,
        preprocess_latency_s=timed_result.preprocess_s,
        model_latency_s=timed_result.model_s,
        postprocess_latency_s=timed_result.postprocess_s,
    )


def run_producer_loop(
    *,
    config: PI05RuntimeConfig,
    state: PI05RuntimeState,
    metrics: PI05RuntimeMetrics,
    queue_controller: ProducerQueueController,
    observation_provider: ObservationProvider,
    dataset_features: Mapping[str, dict[str, Any]],
    policy: ChunkPolicy,
    preprocessor: Callable[[dict[str, Any]], dict[str, Any]],
    postprocessor: Callable[[Tensor], Tensor],
    device: torch.device | str | None = None,
    task: str | None = None,
    robot_type: str | None = None,
    robot_observation_processor: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    sleep_interval_s: float | None = None,
    max_iterations: int | None = None,
    perf_counter: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    """Run the producer loop until stop, timeout, exception, or max iterations."""

    completed_iterations = 0
    loop_sleep_s = config.step_dt_s if sleep_interval_s is None else max(0.0, float(sleep_interval_s))

    while state.running:
        if config.should_stop_by_duration and perf_counter() - state.start_time_s >= config.run_time_s:
            state.request_stop("producer_loop duration elapsed")
            break

        try:
            run_producer_iteration(
                config=config,
                state=state,
                metrics=metrics,
                queue_controller=queue_controller,
                observation_provider=observation_provider,
                dataset_features=dataset_features,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                device=device,
                task=task,
                robot_type=robot_type,
                robot_observation_processor=robot_observation_processor,
                perf_counter=perf_counter,
            )
        except Exception as exc:  # pragma: no cover - exercised via public loop test
            state.record_exception("producer_loop", exc)
            break

        completed_iterations += 1
        if max_iterations is not None and completed_iterations >= max_iterations:
            break

        if loop_sleep_s > 0 and state.running:
            sleep_fn(loop_sleep_s)

    return completed_iterations


def _run_timed_chunk_inference(
    *,
    observation: Mapping[str, Any],
    dataset_features: Mapping[str, dict[str, Any]],
    policy: ChunkPolicy,
    preprocessor: Callable[[dict[str, Any]], dict[str, Any]],
    postprocessor: Callable[[Tensor], Tensor],
    device: torch.device | str | None,
    task: str | None,
    robot_type: str | None,
    robot_observation_processor: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
    rtc_kwargs: Mapping[str, Any] | None,
    perf_counter: Callable[[], float],
) -> TimedChunkInferenceResult:
    resolved_device = _resolve_device(policy, device)
    use_amp = bool(getattr(getattr(policy, "config", None), "use_amp", False))
    total_start_s = perf_counter()

    observation_frame = build_chunk_observation_frame(
        observation,
        dataset_features,
        robot_observation_processor=robot_observation_processor,
    )
    prepared_batch = prepare_policy_batch_from_frame(
        observation_frame,
        resolved_device,
        task=task,
        robot_type=robot_type,
    )
    preprocess_start_s = perf_counter()

    predict_kwargs = dict(rtc_kwargs or {})
    with (
        torch.inference_mode(),
        torch.autocast(device_type=resolved_device.type) if resolved_device.type == "cuda" and use_amp else nullcontext(),
    ):
        preprocessed_batch = preprocessor(dict(prepared_batch))
        preprocess_end_s = perf_counter()
        raw_action_chunk = _ensure_chunk_batch(policy.predict_action_chunk(preprocessed_batch, **predict_kwargs))
        model_end_s = perf_counter()
        processed_action_chunk = _ensure_chunk_batch(postprocessor(raw_action_chunk))
        postprocess_end_s = perf_counter()

    if raw_action_chunk.shape[0] != 1:
        raise ValueError(f"Phase 1 chunk runtime expects batch_size=1, got {tuple(raw_action_chunk.shape)}")
    if processed_action_chunk.shape[0] != 1:
        raise ValueError(
            "Postprocessor must preserve batch_size=1 for Phase 1 runtime, "
            f"got {tuple(processed_action_chunk.shape)}"
        )

    return TimedChunkInferenceResult(
        original_actions=raw_action_chunk.detach().clone().squeeze(0),
        processed_actions=processed_action_chunk.detach().clone().squeeze(0),
        total_s=max(0.0, postprocess_end_s - total_start_s),
        preprocess_s=max(0.0, preprocess_end_s - preprocess_start_s),
        model_s=max(0.0, model_end_s - preprocess_end_s),
        postprocess_s=max(0.0, postprocess_end_s - model_end_s),
        finished_at_s=postprocess_end_s,
    )


def _build_rtc_kwargs(
    *,
    config: PI05RuntimeConfig,
    prev_chunk_left_over: Tensor | None,
    inference_delay: int,
) -> dict[str, Any] | None:
    if not config.enable_rtc:
        return None

    if prev_chunk_left_over is not None:
        prev_chunk_left_over = prev_chunk_left_over.clone()

    return {
        "inference_delay": inference_delay,
        "prev_chunk_left_over": prev_chunk_left_over,
        "execution_horizon": config.rtc_execution_horizon,
    }


def _latest_latency_estimate_s(metrics: PI05RuntimeMetrics) -> float:
    return max(0.0, float(metrics.snapshot().total_latency.latest_s))


def _latency_to_delay_steps(latency_s: float, step_dt_s: float) -> int:
    if latency_s <= 0:
        return 0
    return int(math.ceil(float(latency_s) / float(step_dt_s)))


def _prepare_queue_actions(original_actions: Tensor, processed_actions: Tensor, *, max_steps: int) -> tuple[Tensor, Tensor]:
    if original_actions.ndim != 2 or processed_actions.ndim != 2:
        raise ValueError(
            "Producer loop expects 2D action tensors shaped as (time_steps, action_dim) after chunk inference."
        )
    if original_actions.shape != processed_actions.shape:
        raise ValueError(
            "original_actions and processed_actions must share the same shape, "
            f"got {tuple(original_actions.shape)} vs {tuple(processed_actions.shape)}."
        )
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    return original_actions.clone(), processed_actions[:max_steps].clone()


def _compute_action_index_delta(
    *,
    action_index_before_inference: int | None,
    action_index_after_inference: int | None,
) -> int | None:
    if action_index_before_inference is None or action_index_after_inference is None:
        return None
    return int(action_index_after_inference) - int(action_index_before_inference)


def _should_record_delay_mismatch(
    *,
    config: PI05RuntimeConfig,
    state: PI05RuntimeState,
    action_index_delta: int | None,
    real_delay: int,
) -> bool:
    if action_index_delta is None:
        return False
    if config.enable_rtc:
        return action_index_delta != real_delay

    # In plain mode `real_delay=ceil(latency / dt)` is intentionally conservative.
    # A one-step gap is expected from phase alignment between inference finish time
    # and the actor tick schedule, and startup should not warn before any action is sent.
    if state.actor_iterations <= 0:
        return False
    return abs(action_index_delta - real_delay) > 1


def _resolve_chunk_step_budget(config: PI05RuntimeConfig, queue_depth_before: int) -> int:
    if config.queue_max_size is None:
        return config.actions_per_chunk
    if config.enable_rtc:
        return min(config.actions_per_chunk, config.queue_max_size)
    remaining_capacity = max(config.queue_max_size - max(queue_depth_before, 0), 0)
    return min(config.actions_per_chunk, remaining_capacity)


def _resolve_device(policy: ChunkPolicy, device: torch.device | str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device(getattr(getattr(policy, "config", None), "device", "cpu"))


def _ensure_chunk_batch(actions: Tensor) -> Tensor:
    if actions.ndim == 2:
        return actions.unsqueeze(0)
    if actions.ndim == 3:
        return actions
    raise ValueError(f"Expected action chunk with shape (T, D) or (B, T, D), got {tuple(actions.shape)}")
