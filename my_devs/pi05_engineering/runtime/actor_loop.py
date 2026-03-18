from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from lerobot.policies.utils import make_robot_action

from .metrics import PI05RuntimeMetrics
from .queue_controller import QueueController
from .runtime_config import PI05RuntimeConfig
from .runtime_state import PI05RuntimeState

SendActionCallback = Callable[[Any], None]
RobotActionProcessor = Callable[[tuple[dict[str, float], Any]], Any]
TimeFn = Callable[[], float]
SleepFn = Callable[[float], None]


@dataclass(slots=True)
class ActorIterationResult:
    sent: bool
    queue_depth_before: int
    queue_depth_after: int
    action_available: bool
    stop_requested: bool
    stop_reason: str | None = None
    robot_action: Any | None = None


@dataclass(slots=True)
class ActorLoopResult:
    iterations: int
    startup_ready: bool
    stop_reason: str | None


def run_actor_iteration(
    *,
    config: PI05RuntimeConfig,
    runtime_state: PI05RuntimeState,
    metrics: PI05RuntimeMetrics,
    queue_controller: QueueController,
    dataset_features: Mapping[str, dict[str, Any]],
    send_action: SendActionCallback,
    robot_action_processor: RobotActionProcessor | None = None,
    observation_for_processor: Any | None = None,
    time_fn: TimeFn = time.perf_counter,
) -> ActorIterationResult:
    """Run one offline-safe actor step using the queue controller as the single source of truth."""

    queue_depth_before = _safe_queue_depth(queue_controller)
    metrics.record_queue_depth(queue_depth_before)
    _sync_first_chunk_metric(runtime_state, metrics)

    if not runtime_state.running:
        return ActorIterationResult(
            sent=False,
            queue_depth_before=queue_depth_before,
            queue_depth_after=queue_depth_before,
            action_available=False,
            stop_requested=True,
            stop_reason=runtime_state.stop_reason,
        )

    iteration_timestamp_s = time_fn()
    runtime_state.note_actor_iteration()
    metrics.record_actor_step(timestamp_s=iteration_timestamp_s)

    try:
        action = queue_controller.pop_next_action()
    except BaseException as exc:
        _record_runtime_exception(runtime_state, "actor_iteration.queue", exc)
        return ActorIterationResult(
            sent=False,
            queue_depth_before=queue_depth_before,
            queue_depth_after=queue_depth_before,
            action_available=False,
            stop_requested=True,
            stop_reason=runtime_state.stop_reason,
        )

    queue_depth_after = _safe_queue_depth(queue_controller)
    metrics.record_queue_depth(queue_depth_after)

    if action is None:
        metrics.record_starvation()
        return ActorIterationResult(
            sent=False,
            queue_depth_before=queue_depth_before,
            queue_depth_after=queue_depth_after,
            action_available=False,
            stop_requested=not runtime_state.running,
            stop_reason=runtime_state.stop_reason,
        )

    try:
        action_tensor = _normalize_action_tensor(action)
    except Exception as exc:
        runtime_state.record_exception("actor_iteration.action", exc)
        return ActorIterationResult(
            sent=False,
            queue_depth_before=queue_depth_before,
            queue_depth_after=queue_depth_after,
            action_available=True,
            stop_requested=True,
            stop_reason=runtime_state.stop_reason,
        )

    try:
        delta_error = _validate_action_delta(
            current_action=action_tensor,
            previous_action=runtime_state.get_last_sent_action(),
            max_action_delta=config.max_action_delta,
        )
    except Exception as exc:
        runtime_state.record_exception("actor_iteration.delta", exc)
        return ActorIterationResult(
            sent=False,
            queue_depth_before=queue_depth_before,
            queue_depth_after=queue_depth_after,
            action_available=True,
            stop_requested=True,
            stop_reason=runtime_state.stop_reason,
        )
    if delta_error is not None:
        runtime_state.request_stop(delta_error)
        return ActorIterationResult(
            sent=False,
            queue_depth_before=queue_depth_before,
            queue_depth_after=queue_depth_after,
            action_available=True,
            stop_requested=True,
            stop_reason=runtime_state.stop_reason,
        )

    try:
        robot_action = _build_robot_action(
            action_tensor=action_tensor,
            dataset_features=dataset_features,
            robot_action_processor=robot_action_processor,
            observation_for_processor=observation_for_processor,
        )
        send_action(robot_action)
    except BaseException as exc:
        _record_runtime_exception(runtime_state, "actor_iteration.send", exc)
        return ActorIterationResult(
            sent=False,
            queue_depth_before=queue_depth_before,
            queue_depth_after=queue_depth_after,
            action_available=True,
            stop_requested=True,
            stop_reason=runtime_state.stop_reason,
        )

    timestamp_s = time_fn()
    runtime_state.set_last_sent_action(action_tensor, timestamp_s=timestamp_s)
    return ActorIterationResult(
        sent=True,
        queue_depth_before=queue_depth_before,
        queue_depth_after=queue_depth_after,
        action_available=True,
        stop_requested=not runtime_state.running,
        stop_reason=runtime_state.stop_reason,
        robot_action=robot_action,
    )


def run_actor_loop(
    *,
    config: PI05RuntimeConfig,
    runtime_state: PI05RuntimeState,
    metrics: PI05RuntimeMetrics,
    queue_controller: QueueController,
    dataset_features: Mapping[str, dict[str, Any]],
    send_action: SendActionCallback,
    robot_action_processor: RobotActionProcessor | None = None,
    observation_for_processor: Any | None = None,
    time_fn: TimeFn = time.perf_counter,
    sleep_fn: SleepFn = time.sleep,
) -> ActorLoopResult:
    """Run the actor loop at the configured cadence without touching real hardware directly."""

    startup_ready = _wait_for_startup_barrier(
        config=config,
        runtime_state=runtime_state,
        metrics=metrics,
    )
    if not startup_ready:
        return ActorLoopResult(iterations=0, startup_ready=False, stop_reason=runtime_state.stop_reason)

    iterations = 0
    next_tick_s = time_fn()

    while runtime_state.running:
        if config.should_stop_by_duration and (time_fn() - runtime_state.start_time_s) >= config.run_time_s:
            runtime_state.request_stop(
                f"actor_loop duration limit reached after {config.run_time_s:.3f}s"
            )
            break

        try:
            run_actor_iteration(
                config=config,
                runtime_state=runtime_state,
                metrics=metrics,
                queue_controller=queue_controller,
                dataset_features=dataset_features,
                send_action=send_action,
                robot_action_processor=robot_action_processor,
                observation_for_processor=observation_for_processor,
                time_fn=time_fn,
            )
        except BaseException as exc:
            _record_runtime_exception(runtime_state, "actor_loop", exc)
            break

        iterations += 1
        if not runtime_state.running:
            break

        next_tick_s += config.step_dt_s
        sleep_s = next_tick_s - time_fn()
        if sleep_s > 0:
            try:
                sleep_fn(sleep_s)
            except BaseException as exc:
                _record_runtime_exception(runtime_state, "actor_loop.sleep", exc)
                break
        else:
            next_tick_s = time_fn()

    return ActorLoopResult(
        iterations=iterations,
        startup_ready=startup_ready,
        stop_reason=runtime_state.stop_reason,
    )


def _wait_for_startup_barrier(
    *,
    config: PI05RuntimeConfig,
    runtime_state: PI05RuntimeState,
    metrics: PI05RuntimeMetrics,
) -> bool:
    if runtime_state.first_chunk_ready_event.is_set():
        _sync_first_chunk_metric(runtime_state, metrics)
        return True

    if not config.startup_wait_for_first_chunk:
        return True

    ready = runtime_state.wait_for_first_chunk(timeout_s=config.startup_timeout_s)
    if not ready:
        runtime_state.request_stop(
            f"actor_loop startup timeout after {config.startup_timeout_s:.3f}s waiting for first chunk"
        )
        return False

    _sync_first_chunk_metric(runtime_state, metrics)
    return True


def _sync_first_chunk_metric(runtime_state: PI05RuntimeState, metrics: PI05RuntimeMetrics) -> None:
    if runtime_state.first_chunk_ready_event.is_set():
        metrics.mark_first_chunk_ready(timestamp_s=runtime_state.first_chunk_ready_at_s)


def _safe_queue_depth(queue_controller: QueueController) -> int:
    try:
        return max(0, int(queue_controller.qsize()))
    except Exception:
        return 0


def _normalize_action_tensor(action: Any) -> Tensor:
    tensor = torch.as_tensor(action, dtype=torch.float32).detach().clone()
    if tensor.ndim == 2:
        if tensor.shape[0] != 1:
            raise ValueError(
                "Actor loop expects a single processed action per step, "
                f"got tensor shape {tuple(tensor.shape)}"
            )
        tensor = tensor.squeeze(0)
    if tensor.ndim != 1:
        raise ValueError(
            "Actor loop expects a 1D action tensor after queue pop, "
            f"got tensor shape {tuple(tensor.shape)}"
        )
    return tensor


def _validate_action_delta(
    *,
    current_action: Tensor,
    previous_action: Any,
    max_action_delta: float | None,
) -> str | None:
    if max_action_delta is None or previous_action is None:
        return None

    previous_tensor = _normalize_action_tensor(previous_action)
    if previous_tensor.shape != current_action.shape:
        return (
            "actor_iteration max_action_delta shape mismatch: "
            f"{tuple(previous_tensor.shape)} vs {tuple(current_action.shape)}"
        )

    max_delta = float(torch.max(torch.abs(current_action - previous_tensor)).item())
    if max_delta > max_action_delta:
        return (
            "actor_iteration max_action_delta exceeded: "
            f"delta={max_delta:.6f} limit={max_action_delta:.6f}"
        )
    return None


def _build_robot_action(
    *,
    action_tensor: Tensor,
    dataset_features: Mapping[str, dict[str, Any]],
    robot_action_processor: RobotActionProcessor | None,
    observation_for_processor: Any | None,
) -> Any:
    action_dict = make_robot_action(action_tensor.unsqueeze(0), dict(dataset_features))
    if robot_action_processor is None:
        return action_dict
    return robot_action_processor((action_dict, observation_for_processor))


def _record_runtime_exception(
    runtime_state: PI05RuntimeState,
    source: str,
    exc: BaseException,
) -> None:
    runtime_state.record_exception(source, _normalize_exception_for_record(exc))


def _normalize_exception_for_record(exc: BaseException) -> BaseException:
    if isinstance(exc, KeyboardInterrupt) and not str(exc):
        return KeyboardInterrupt("KeyboardInterrupt").with_traceback(exc.__traceback__)
    return exc


__all__ = [
    "ActorIterationResult",
    "ActorLoopResult",
    "run_actor_iteration",
    "run_actor_loop",
]
