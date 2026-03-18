from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

from lerobot.policies.rtc.latency_tracker import LatencyTracker


@dataclass(frozen=True, slots=True)
class MetricSeriesSnapshot:
    count: int
    latest_s: float
    max_s: float
    p95_s: float


@dataclass(frozen=True, slots=True)
class PI05RuntimeMetricsSnapshot:
    uptime_s: float
    first_chunk_ready_at_s: float | None
    first_warmup_latency_s: float | None
    current_queue_depth: int
    max_queue_depth: int
    queue_depth_samples: int
    starvation_count: int
    refill_count: int
    stale_prefix_trim_events: int
    stale_prefix_trimmed_actions: int
    total_inference_count: int
    total_actor_steps: int
    total_producer_steps: int
    latest_chunk_length: int | None
    latest_leftover_length: int | None
    latest_original_backlog_length: int | None
    latest_executable_backlog_length: int | None
    latest_backlog_gap: int | None
    latest_action_index_before_inference: int | None
    latest_action_index_delta: int | None
    latest_inference_delay: int | None
    latest_real_delay: int | None
    latest_delay_error: int | None
    delay_mismatch_count: int
    delay_mismatch_ratio: float
    delay_error_mean: float
    delay_error_max_abs: int
    actor_rate_hz: float
    producer_rate_hz: float
    latest_inference_overrun_ratio: float | None
    total_latency: MetricSeriesSnapshot
    preprocess_latency: MetricSeriesSnapshot
    model_latency: MetricSeriesSnapshot
    postprocess_latency: MetricSeriesSnapshot


class _LatencySeries:
    def __init__(self, window_size: int) -> None:
        self._tracker = LatencyTracker(maxlen=window_size)
        self._latest = 0.0

    def add(self, value_s: float) -> None:
        value = float(value_s)
        if value < 0:
            return
        self._latest = value
        self._tracker.add(value)

    def snapshot(self) -> MetricSeriesSnapshot:
        return MetricSeriesSnapshot(
            count=len(self._tracker),
            latest_s=self._latest,
            max_s=float(self._tracker.max() or 0.0),
            p95_s=float(self._tracker.p95() or 0.0),
        )


class _RateTracker:
    def __init__(self, maxlen: int) -> None:
        self._timestamps: deque[float] = deque(maxlen=maxlen)

    def record(self, timestamp_s: float) -> None:
        self._timestamps.append(float(timestamp_s))

    def rate_hz(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed


class PI05RuntimeMetrics:
    """Thread-safe metrics collector for the local PI05 runtime."""

    def __init__(self, window_size: int = 100) -> None:
        self._window_size = window_size
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._start_time_s = time.perf_counter()
            self._first_chunk_ready_at_s: float | None = None
            self._first_warmup_latency_s: float | None = None

            self._current_queue_depth = 0
            self._max_queue_depth = 0
            self._queue_depth_samples = 0

            self._starvation_count = 0
            self._refill_count = 0
            self._stale_prefix_trim_events = 0
            self._stale_prefix_trimmed_actions = 0

            self._total_inference_count = 0
            self._total_actor_steps = 0
            self._total_producer_steps = 0

            self._latest_chunk_length: int | None = None
            self._latest_leftover_length: int | None = None
            self._latest_original_backlog_length: int | None = None
            self._latest_executable_backlog_length: int | None = None
            self._latest_backlog_gap: int | None = None
            self._latest_action_index_before_inference: int | None = None
            self._latest_action_index_delta: int | None = None
            self._latest_inference_delay: int | None = None
            self._latest_real_delay: int | None = None
            self._latest_delay_error: int | None = None
            self._delay_mismatch_count = 0
            self._delay_error_sum = 0
            self._delay_error_count = 0
            self._delay_error_max_abs = 0
            self._latest_inference_overrun_ratio: float | None = None

            self._actor_rate = _RateTracker(maxlen=self._window_size)
            self._producer_rate = _RateTracker(maxlen=self._window_size)

            self._total_latency = _LatencySeries(self._window_size)
            self._preprocess_latency = _LatencySeries(self._window_size)
            self._model_latency = _LatencySeries(self._window_size)
            self._postprocess_latency = _LatencySeries(self._window_size)

    def rebase_start_time(self, timestamp_s: float | None = None) -> float:
        with self._lock:
            self._start_time_s = timestamp_s if timestamp_s is not None else time.perf_counter()
            return self._start_time_s

    def mark_first_chunk_ready(self, timestamp_s: float | None = None) -> bool:
        with self._lock:
            if self._first_chunk_ready_at_s is not None:
                return False
            self._first_chunk_ready_at_s = timestamp_s if timestamp_s is not None else time.perf_counter()
            return True

    def record_queue_depth(self, depth: int) -> None:
        queue_depth = max(0, int(depth))
        with self._lock:
            self._current_queue_depth = queue_depth
            self._max_queue_depth = max(self._max_queue_depth, queue_depth)
            self._queue_depth_samples += 1

    def record_starvation(self, count: int = 1) -> None:
        with self._lock:
            self._starvation_count += max(0, int(count))

    def record_refill(
        self,
        *,
        chunk_length: int | None = None,
        queue_depth_after: int | None = None,
        leftover_length: int | None = None,
        original_backlog_length: int | None = None,
        executable_backlog_length: int | None = None,
    ) -> None:
        if original_backlog_length is None:
            original_backlog_length = leftover_length
        if executable_backlog_length is None:
            executable_backlog_length = queue_depth_after
        with self._lock:
            self._refill_count += 1
            if chunk_length is not None:
                self._latest_chunk_length = int(chunk_length)
            self._update_backlog_fields_locked(
                original_backlog_length=original_backlog_length,
                executable_backlog_length=executable_backlog_length,
            )
        if queue_depth_after is not None:
            self.record_queue_depth(queue_depth_after)

    def record_stale_prefix_trim(self, trimmed_actions: int) -> None:
        trimmed = max(0, int(trimmed_actions))
        if trimmed <= 0:
            return
        with self._lock:
            self._stale_prefix_trim_events += 1
            self._stale_prefix_trimmed_actions += trimmed

    def record_producer_iteration(self, timestamp_s: float | None = None) -> None:
        ts = timestamp_s if timestamp_s is not None else time.perf_counter()
        with self._lock:
            self._total_producer_steps += 1
            self._producer_rate.record(ts)

    def record_actor_step(self, timestamp_s: float | None = None, queue_depth_after: int | None = None) -> None:
        ts = timestamp_s if timestamp_s is not None else time.perf_counter()
        with self._lock:
            self._total_actor_steps += 1
            self._actor_rate.record(ts)
        if queue_depth_after is not None:
            self.record_queue_depth(queue_depth_after)

    def record_inference(
        self,
        *,
        total_s: float,
        preprocess_s: float | None = None,
        model_s: float | None = None,
        postprocess_s: float | None = None,
        chunk_length: int | None = None,
        leftover_length: int | None = None,
        original_backlog_length: int | None = None,
        executable_backlog_length: int | None = None,
        action_index_before_inference: int | None = None,
        action_index_delta: int | None = None,
        inference_delay: int | None = None,
        real_delay: int | None = None,
        delay_mismatch: bool = False,
        queue_depth_before: int | None = None,
        queue_depth_after: int | None = None,
        inference_overrun_ratio: float | None = None,
    ) -> None:
        if original_backlog_length is None:
            original_backlog_length = leftover_length
        if executable_backlog_length is None:
            executable_backlog_length = queue_depth_after
        with self._lock:
            self._total_inference_count += 1
            if self._first_warmup_latency_s is None:
                self._first_warmup_latency_s = float(total_s)
            self._total_latency.add(total_s)
            if preprocess_s is not None:
                self._preprocess_latency.add(preprocess_s)
            if model_s is not None:
                self._model_latency.add(model_s)
            if postprocess_s is not None:
                self._postprocess_latency.add(postprocess_s)
            if chunk_length is not None:
                self._latest_chunk_length = int(chunk_length)
            self._update_backlog_fields_locked(
                original_backlog_length=original_backlog_length,
                executable_backlog_length=executable_backlog_length,
            )
            if action_index_before_inference is not None:
                self._latest_action_index_before_inference = int(action_index_before_inference)
            if action_index_delta is not None:
                self._latest_action_index_delta = int(action_index_delta)
            if inference_delay is not None:
                self._latest_inference_delay = int(inference_delay)
            if real_delay is not None:
                self._latest_real_delay = int(real_delay)
            if action_index_delta is not None and real_delay is not None:
                delay_error = int(real_delay) - int(action_index_delta)
                self._latest_delay_error = delay_error
                self._delay_error_sum += delay_error
                self._delay_error_count += 1
                self._delay_error_max_abs = max(self._delay_error_max_abs, abs(delay_error))
            if delay_mismatch:
                self._delay_mismatch_count += 1
            if inference_overrun_ratio is not None:
                self._latest_inference_overrun_ratio = max(0.0, float(inference_overrun_ratio))

        if queue_depth_before is not None:
            self.record_queue_depth(queue_depth_before)
        if queue_depth_after is not None:
            self.record_queue_depth(queue_depth_after)

    def _update_backlog_fields_locked(
        self,
        *,
        original_backlog_length: int | None = None,
        executable_backlog_length: int | None = None,
    ) -> None:
        if original_backlog_length is not None:
            self._latest_leftover_length = int(original_backlog_length)
            self._latest_original_backlog_length = int(original_backlog_length)
        if executable_backlog_length is not None:
            self._latest_executable_backlog_length = int(executable_backlog_length)
        if self._latest_original_backlog_length is None or self._latest_executable_backlog_length is None:
            self._latest_backlog_gap = None
            return
        self._latest_backlog_gap = (
            int(self._latest_original_backlog_length) - int(self._latest_executable_backlog_length)
        )

    def snapshot(self, now_s: float | None = None) -> PI05RuntimeMetricsSnapshot:
        with self._lock:
            now = now_s if now_s is not None else time.perf_counter()
            delay_mismatch_ratio = (
                float(self._delay_mismatch_count) / float(self._total_inference_count)
                if self._total_inference_count > 0
                else 0.0
            )
            delay_error_mean = (
                float(self._delay_error_sum) / float(self._delay_error_count)
                if self._delay_error_count > 0
                else 0.0
            )
            return PI05RuntimeMetricsSnapshot(
                uptime_s=max(0.0, now - self._start_time_s),
                first_chunk_ready_at_s=self._first_chunk_ready_at_s,
                first_warmup_latency_s=self._first_warmup_latency_s,
                current_queue_depth=self._current_queue_depth,
                max_queue_depth=self._max_queue_depth,
                queue_depth_samples=self._queue_depth_samples,
                starvation_count=self._starvation_count,
                refill_count=self._refill_count,
                stale_prefix_trim_events=self._stale_prefix_trim_events,
                stale_prefix_trimmed_actions=self._stale_prefix_trimmed_actions,
                total_inference_count=self._total_inference_count,
                total_actor_steps=self._total_actor_steps,
                total_producer_steps=self._total_producer_steps,
                latest_chunk_length=self._latest_chunk_length,
                latest_leftover_length=self._latest_leftover_length,
                latest_original_backlog_length=self._latest_original_backlog_length,
                latest_executable_backlog_length=self._latest_executable_backlog_length,
                latest_backlog_gap=self._latest_backlog_gap,
                latest_action_index_before_inference=self._latest_action_index_before_inference,
                latest_action_index_delta=self._latest_action_index_delta,
                latest_inference_delay=self._latest_inference_delay,
                latest_real_delay=self._latest_real_delay,
                latest_delay_error=self._latest_delay_error,
                delay_mismatch_count=self._delay_mismatch_count,
                delay_mismatch_ratio=delay_mismatch_ratio,
                delay_error_mean=delay_error_mean,
                delay_error_max_abs=self._delay_error_max_abs,
                actor_rate_hz=self._actor_rate.rate_hz(),
                producer_rate_hz=self._producer_rate.rate_hz(),
                latest_inference_overrun_ratio=self._latest_inference_overrun_ratio,
                total_latency=self._total_latency.snapshot(),
                preprocess_latency=self._preprocess_latency.snapshot(),
                model_latency=self._model_latency.snapshot(),
                postprocess_latency=self._postprocess_latency.snapshot(),
            )

    def as_log_dict(self, now_s: float | None = None) -> dict[str, float | int | None]:
        snapshot = self.snapshot(now_s=now_s)
        return {
            "uptime_s": snapshot.uptime_s,
            "first_chunk_ready_at_s": snapshot.first_chunk_ready_at_s,
            "first_warmup_latency_s": snapshot.first_warmup_latency_s,
            "current_queue_depth": snapshot.current_queue_depth,
            "max_queue_depth": snapshot.max_queue_depth,
            "queue_depth_samples": snapshot.queue_depth_samples,
            "starvation_count": snapshot.starvation_count,
            "refill_count": snapshot.refill_count,
            "stale_prefix_trim_events": snapshot.stale_prefix_trim_events,
            "stale_prefix_trimmed_actions": snapshot.stale_prefix_trimmed_actions,
            "total_inference_count": snapshot.total_inference_count,
            "total_actor_steps": snapshot.total_actor_steps,
            "total_producer_steps": snapshot.total_producer_steps,
            "latest_chunk_length": snapshot.latest_chunk_length,
            "latest_leftover_length": snapshot.latest_leftover_length,
            "latest_original_backlog_length": snapshot.latest_original_backlog_length,
            "latest_executable_backlog_length": snapshot.latest_executable_backlog_length,
            "latest_backlog_gap": snapshot.latest_backlog_gap,
            "latest_action_index_before_inference": snapshot.latest_action_index_before_inference,
            "latest_action_index_delta": snapshot.latest_action_index_delta,
            "latest_inference_delay": snapshot.latest_inference_delay,
            "latest_real_delay": snapshot.latest_real_delay,
            "latest_delay_error": snapshot.latest_delay_error,
            "delay_mismatch_count": snapshot.delay_mismatch_count,
            "delay_mismatch_ratio": snapshot.delay_mismatch_ratio,
            "delay_error_mean": snapshot.delay_error_mean,
            "delay_error_max_abs": snapshot.delay_error_max_abs,
            "actor_rate_hz": snapshot.actor_rate_hz,
            "producer_rate_hz": snapshot.producer_rate_hz,
            "latest_inference_overrun_ratio": snapshot.latest_inference_overrun_ratio,
            "latency_total_latest_s": snapshot.total_latency.latest_s,
            "latency_total_max_s": snapshot.total_latency.max_s,
            "latency_total_p95_s": snapshot.total_latency.p95_s,
            "latency_preprocess_latest_s": snapshot.preprocess_latency.latest_s,
            "latency_model_latest_s": snapshot.model_latency.latest_s,
            "latency_postprocess_latest_s": snapshot.postprocess_latency.latest_s,
        }
