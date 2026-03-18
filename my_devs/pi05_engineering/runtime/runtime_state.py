from __future__ import annotations

import copy
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any


def _clone_value(value: Any) -> Any:
    if value is None:
        return None

    clone = getattr(value, "clone", None)
    if callable(clone):
        try:
            return clone()
        except TypeError:
            pass

    return copy.deepcopy(value)


@dataclass(slots=True)
class RuntimeErrorRecord:
    source: str
    message: str
    timestamp_s: float
    traceback_text: str


@dataclass(slots=True)
class PI05RuntimeStateSnapshot:
    running: bool
    start_time_s: float
    stop_requested_at_s: float | None
    stop_reason: str | None
    first_chunk_ready: bool
    first_chunk_ready_at_s: float | None
    producer_iterations: int
    actor_iterations: int
    last_producer_iteration_at_s: float | None
    last_action_sent_at_s: float | None
    last_action_index_before_inference: int | None
    last_action_index_delta: int | None
    last_inference_delay: int | None
    last_real_delay: int | None
    last_merge_mode: str | None
    last_queue_depth_before_merge: int | None
    last_queue_depth_after_merge: int | None
    last_trimmed_prefix_steps: int | None
    last_enqueued_steps: int | None
    last_chunk_finished_at_s: float | None
    has_prev_chunk_left_over: bool
    last_prev_chunk_left_over_length: int | None
    has_original_actions: bool
    last_original_actions_length: int | None
    has_processed_actions: bool
    last_processed_actions_length: int | None
    has_last_action: bool
    last_error: RuntimeErrorRecord | None


@dataclass(slots=True)
class PI05RuntimeState:
    """Thread-shared runtime state for producer/actor orchestration."""

    shutdown_event: threading.Event = field(default_factory=threading.Event, repr=False)
    first_chunk_ready_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    start_time_s: float = field(default_factory=time.perf_counter)
    stop_requested_at_s: float | None = None
    stop_reason: str | None = None
    first_chunk_ready_at_s: float | None = None
    producer_iterations: int = 0
    actor_iterations: int = 0
    last_producer_iteration_at_s: float | None = None
    last_action_sent_at_s: float | None = None
    last_action_index_before_inference: int | None = None
    last_action_index_delta: int | None = None
    last_inference_delay: int | None = None
    last_real_delay: int | None = None
    last_merge_mode: str | None = None
    last_queue_depth_before_merge: int | None = None
    last_queue_depth_after_merge: int | None = None
    last_trimmed_prefix_steps: int | None = None
    last_enqueued_steps: int | None = None
    last_chunk_finished_at_s: float | None = None
    _last_sent_action: Any = field(default=None, repr=False)
    _last_prev_chunk_left_over: Any = field(default=None, repr=False)
    _last_original_actions: Any = field(default=None, repr=False)
    _last_processed_actions: Any = field(default=None, repr=False)
    last_error: RuntimeErrorRecord | None = None

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    def request_stop(self, reason: str) -> bool:
        with self._lock:
            already_stopped = self.shutdown_event.is_set()
            if not already_stopped:
                self.stop_reason = reason
                self.stop_requested_at_s = time.perf_counter()
                self.shutdown_event.set()
            return not already_stopped

    def rebase_start_time(self, timestamp_s: float | None = None) -> float:
        with self._lock:
            self.start_time_s = timestamp_s if timestamp_s is not None else time.perf_counter()
            return self.start_time_s

    def mark_first_chunk_ready(self, timestamp_s: float | None = None) -> bool:
        with self._lock:
            if self.first_chunk_ready_event.is_set():
                return False
            self.first_chunk_ready_at_s = timestamp_s if timestamp_s is not None else time.perf_counter()
            self.first_chunk_ready_event.set()
            return True

    def wait_for_first_chunk(self, timeout_s: float | None = None) -> bool:
        return self.first_chunk_ready_event.wait(timeout=timeout_s)

    def set_last_sent_action(self, action: Any, timestamp_s: float | None = None) -> None:
        with self._lock:
            self._last_sent_action = _clone_value(action)
            self.last_action_sent_at_s = timestamp_s if timestamp_s is not None else time.perf_counter()

    def get_last_sent_action(self) -> Any:
        with self._lock:
            return _clone_value(self._last_sent_action)

    def note_actor_iteration(self, action: Any | None = None, timestamp_s: float | None = None) -> None:
        with self._lock:
            self.actor_iterations += 1
        if action is not None:
            self.set_last_sent_action(action, timestamp_s=timestamp_s)

    def note_producer_iteration(
        self,
        action_index_before_inference: int | None = None,
        timestamp_s: float | None = None,
    ) -> None:
        with self._lock:
            self.producer_iterations += 1
            self.last_producer_iteration_at_s = timestamp_s if timestamp_s is not None else time.perf_counter()
            if action_index_before_inference is not None:
                self.last_action_index_before_inference = action_index_before_inference

    def note_chunk_inference(
        self,
        *,
        prev_chunk_left_over: Any = None,
        original_actions: Any = None,
        processed_actions: Any = None,
        action_index_before_inference: int | None = None,
        action_index_delta: int | None = None,
        inference_delay: int | None = None,
        real_delay: int | None = None,
        merge_mode: str | None = None,
        queue_depth_before_merge: int | None = None,
        queue_depth_after_merge: int | None = None,
        trimmed_prefix_steps: int | None = None,
        enqueued_steps: int | None = None,
        finished_at_s: float | None = None,
    ) -> None:
        with self._lock:
            self._last_prev_chunk_left_over = _clone_value(prev_chunk_left_over)
            self._last_original_actions = _clone_value(original_actions)
            self._last_processed_actions = _clone_value(processed_actions)
            if action_index_before_inference is not None:
                self.last_action_index_before_inference = action_index_before_inference
            if action_index_delta is not None:
                self.last_action_index_delta = action_index_delta
            if inference_delay is not None:
                self.last_inference_delay = inference_delay
            if real_delay is not None:
                self.last_real_delay = real_delay
            self.last_merge_mode = merge_mode
            self.last_queue_depth_before_merge = queue_depth_before_merge
            self.last_queue_depth_after_merge = queue_depth_after_merge
            self.last_trimmed_prefix_steps = trimmed_prefix_steps
            self.last_enqueued_steps = enqueued_steps
            self.last_chunk_finished_at_s = finished_at_s

    def get_last_prev_chunk_left_over(self) -> Any:
        with self._lock:
            return _clone_value(self._last_prev_chunk_left_over)

    def get_last_original_actions(self) -> Any:
        with self._lock:
            return _clone_value(self._last_original_actions)

    def get_last_processed_actions(self) -> Any:
        with self._lock:
            return _clone_value(self._last_processed_actions)

    def record_exception(self, source: str, exc: BaseException) -> RuntimeErrorRecord:
        tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        record = RuntimeErrorRecord(
            source=source,
            message=str(exc),
            timestamp_s=time.perf_counter(),
            traceback_text=tb_text,
        )
        with self._lock:
            self.last_error = record
        self.request_stop(f"{source}: {exc}")
        return record

    def snapshot(self) -> PI05RuntimeStateSnapshot:
        with self._lock:
            return PI05RuntimeStateSnapshot(
                running=not self.shutdown_event.is_set(),
                start_time_s=self.start_time_s,
                stop_requested_at_s=self.stop_requested_at_s,
                stop_reason=self.stop_reason,
                first_chunk_ready=self.first_chunk_ready_event.is_set(),
                first_chunk_ready_at_s=self.first_chunk_ready_at_s,
                producer_iterations=self.producer_iterations,
                actor_iterations=self.actor_iterations,
                last_producer_iteration_at_s=self.last_producer_iteration_at_s,
                last_action_sent_at_s=self.last_action_sent_at_s,
                last_action_index_before_inference=self.last_action_index_before_inference,
                last_action_index_delta=self.last_action_index_delta,
                last_inference_delay=self.last_inference_delay,
                last_real_delay=self.last_real_delay,
                last_merge_mode=self.last_merge_mode,
                last_queue_depth_before_merge=self.last_queue_depth_before_merge,
                last_queue_depth_after_merge=self.last_queue_depth_after_merge,
                last_trimmed_prefix_steps=self.last_trimmed_prefix_steps,
                last_enqueued_steps=self.last_enqueued_steps,
                last_chunk_finished_at_s=self.last_chunk_finished_at_s,
                has_prev_chunk_left_over=self._last_prev_chunk_left_over is not None,
                last_prev_chunk_left_over_length=_safe_length(self._last_prev_chunk_left_over),
                has_original_actions=self._last_original_actions is not None,
                last_original_actions_length=_safe_length(self._last_original_actions),
                has_processed_actions=self._last_processed_actions is not None,
                last_processed_actions_length=_safe_length(self._last_processed_actions),
                has_last_action=self._last_sent_action is not None,
                last_error=self.last_error,
            )


def _safe_length(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return len(value)
    except TypeError:
        return None
