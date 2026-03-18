"""Phase 1 queue controller for the local PI05 chunk runtime.

This wrapper keeps `src/lerobot/policies/rtc/action_queue.py` as the low-level
queue primitive while adding the extra behavior Phase 1 needs:

- explicit plain vs. RTC merge entry points
- stale-prefix trimming for plain chunk mode
- configurable empty-queue behavior
- lightweight counters for runtime metrics

The controller is intentionally hardware-agnostic and safe to test fully
offline.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig

EMPTY_QUEUE_STRATEGIES = {"hold-last-action", "skip-send", "raise"}
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class QueueControllerCounters:
    """Monotonic counters that the runtime can snapshot for metrics/logging."""

    plain_merge_calls: int = 0
    rtc_merge_calls: int = 0
    stale_prefix_trim_events: int = 0
    stale_prefix_trimmed_actions: int = 0
    dropped_all_stale_chunks: int = 0
    empty_queue_events: int = 0
    held_last_action_events: int = 0
    skip_send_events: int = 0
    actions_popped: int = 0


@dataclass(slots=True)
class MergeResult:
    """Structured merge result for higher-level runtime bookkeeping."""

    mode: str
    input_steps: int
    trimmed_prefix_steps: int
    enqueued_steps: int
    queue_depth_after: int
    dropped_all: bool = False


class QueueController:
    """Runtime-facing wrapper around the RTC ActionQueue primitive."""

    def __init__(
        self,
        *,
        enable_rtc: bool = False,
        empty_queue_strategy: str = "hold-last-action",
        rtc_execution_horizon: int = 10,
        rtc_max_guidance_weight: float = 10.0,
    ) -> None:
        if empty_queue_strategy not in EMPTY_QUEUE_STRATEGIES:
            raise ValueError(
                f"Unsupported empty_queue_strategy={empty_queue_strategy!r}. "
                f"Expected one of {sorted(EMPTY_QUEUE_STRATEGIES)}."
            )

        self._rtc_cfg = RTCConfig(
            enabled=enable_rtc,
            execution_horizon=rtc_execution_horizon,
            max_guidance_weight=rtc_max_guidance_weight,
        )
        self._queue = ActionQueue(self._rtc_cfg)
        self.empty_queue_strategy = empty_queue_strategy
        self._last_sent_action: Tensor | None = None
        self._original_backlog: Tensor | None = None
        self._counters = QueueControllerCounters()
        self._lock = threading.RLock()

    @property
    def rtc_enabled(self) -> bool:
        with self._lock:
            return self._queue.cfg.enabled

    def set_rtc_enabled(self, enabled: bool) -> None:
        """Switch the underlying ActionQueue merge mode in-place."""
        with self._lock:
            self._queue.cfg.enabled = enabled

    def qsize(self) -> int:
        with self._lock:
            return self._queue.qsize()

    def empty(self) -> bool:
        with self._lock:
            return self._queue.empty()

    def get_action_index(self) -> int:
        with self._lock:
            return self._queue.get_action_index()

    def get_left_over_original_actions(self) -> Tensor | None:
        with self._lock:
            return self._current_original_left_over_locked()

    def has_last_action(self) -> bool:
        with self._lock:
            return self._last_sent_action is not None

    def counters(self) -> QueueControllerCounters:
        """Return a copy-friendly view of the controller counters."""
        with self._lock:
            return QueueControllerCounters(**asdict(self._counters))

    def merge_plain(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        *,
        stale_prefix_steps: int = 0,
    ) -> MergeResult:
        """Append a new chunk in plain mode with explicit stale-prefix trimming."""
        with self._lock:
            self._validate_actions(original_actions, processed_actions)
            self._counters.plain_merge_calls += 1

            queue_depth_before = self._queue.qsize()
            existing_original_left_over = self._current_original_left_over_locked()
            kept_existing_original = existing_original_left_over if queue_depth_before > 0 else None

            original_trimmed_steps = self._normalize_trim_steps(stale_prefix_steps, len(original_actions))
            processed_trimmed_steps = self._normalize_trim_steps(stale_prefix_steps, len(processed_actions))

            if original_trimmed_steps:
                self._counters.stale_prefix_trim_events += 1
                self._counters.stale_prefix_trimmed_actions += original_trimmed_steps

            trimmed_original = original_actions[original_trimmed_steps:].clone()
            trimmed_processed = processed_actions[processed_trimmed_steps:].clone()
            self._original_backlog = self._concat_original_left_overs(kept_existing_original, trimmed_original)

            if len(trimmed_processed) == 0:
                self._counters.dropped_all_stale_chunks += 1
                return MergeResult(
                    mode="plain",
                    input_steps=len(original_actions),
                    trimmed_prefix_steps=original_trimmed_steps,
                    enqueued_steps=0,
                    queue_depth_after=self._queue.qsize(),
                    dropped_all=True,
                )

            previous_mode = self._queue.cfg.enabled
            self._queue.cfg.enabled = False
            try:
                self._queue.merge(
                    self._original_for_processed_queue(trimmed_original, trimmed_processed),
                    trimmed_processed,
                    real_delay=0,
                    action_index_before_inference=None,
                )
            finally:
                self._queue.cfg.enabled = previous_mode

            return MergeResult(
                mode="plain",
                input_steps=len(original_actions),
                trimmed_prefix_steps=original_trimmed_steps,
                enqueued_steps=len(trimmed_processed),
                queue_depth_after=self._queue.qsize(),
            )

    def merge_rtc(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        *,
        real_delay: int,
        action_index_before_inference: int | None = 0,
    ) -> MergeResult:
        """Replace the queue using RTC semantics from the low-level ActionQueue."""
        with self._lock:
            self._validate_actions(original_actions, processed_actions)
            self._counters.rtc_merge_calls += 1

            original_trimmed_steps = self._normalize_trim_steps(real_delay, len(original_actions))
            processed_trimmed_steps = self._normalize_trim_steps(real_delay, len(processed_actions))
            self._original_backlog = original_actions[original_trimmed_steps:].clone()

            previous_mode = self._queue.cfg.enabled
            self._queue.cfg.enabled = True
            try:
                self._queue.merge(
                    self._original_for_processed_queue(original_actions, processed_actions),
                    processed_actions,
                    real_delay=real_delay,
                    action_index_before_inference=action_index_before_inference,
                )
            finally:
                self._queue.cfg.enabled = previous_mode

            enqueued_steps = max(len(processed_actions) - processed_trimmed_steps, 0)
            return MergeResult(
                mode="rtc",
                input_steps=len(original_actions),
                trimmed_prefix_steps=original_trimmed_steps,
                enqueued_steps=enqueued_steps,
                queue_depth_after=self._queue.qsize(),
                dropped_all=enqueued_steps == 0,
            )

    def pop_next_action(self) -> Tensor | None:
        """Pop the next action or apply the configured empty-queue strategy."""
        with self._lock:
            action = self._queue.get()
            if action is not None:
                self._last_sent_action = action.clone()
                self._counters.actions_popped += 1
                return action

            self._counters.empty_queue_events += 1

            if self.empty_queue_strategy == "hold-last-action" and self._last_sent_action is not None:
                self._counters.held_last_action_events += 1
                return self._last_sent_action.clone()

            if self.empty_queue_strategy == "skip-send" or (
                self.empty_queue_strategy == "hold-last-action" and self._last_sent_action is None
            ):
                self._counters.skip_send_events += 1
                logger.warning(
                    "[PI05_QUEUE] Queue empty and no last action is available; skipping send "
                    "(strategy=%s).",
                    self.empty_queue_strategy,
                )
                return None

            raise RuntimeError("Action queue is empty and empty_queue_strategy='raise'.")

    @staticmethod
    def _normalize_trim_steps(stale_prefix_steps: int, num_steps: int) -> int:
        if stale_prefix_steps < 0:
            raise ValueError(f"stale_prefix_steps must be >= 0, got {stale_prefix_steps}")
        return min(stale_prefix_steps, num_steps)

    @staticmethod
    def _validate_actions(original_actions: Tensor, processed_actions: Tensor) -> None:
        if original_actions.ndim != 2 or processed_actions.ndim != 2:
            raise ValueError(
                "QueueController expects 2D action tensors shaped as (time_steps, action_dim)."
            )
        if original_actions.shape[1:] != processed_actions.shape[1:]:
            raise ValueError(
                "original_actions and processed_actions must have the same action dimension, "
                f"got {tuple(original_actions.shape)} vs {tuple(processed_actions.shape)}."
            )
        if len(original_actions) < len(processed_actions):
            raise ValueError(
                "original_actions must be at least as long as processed_actions so the full "
                "model-space chunk remains available for leftover/merge bookkeeping, "
                f"got {len(original_actions)} < {len(processed_actions)}."
            )

    def _current_original_left_over_locked(self) -> Tensor | None:
        if self._original_backlog is None:
            return None

        action_index = min(self._queue.get_action_index(), len(self._original_backlog))
        return self._original_backlog[action_index:].clone()

    @staticmethod
    def _concat_original_left_overs(existing: Tensor | None, new_chunk: Tensor) -> Tensor | None:
        if existing is None:
            return None if len(new_chunk) == 0 else new_chunk.clone()
        if len(new_chunk) == 0:
            return existing.clone()
        return torch.cat([existing.clone(), new_chunk.clone()])

    @staticmethod
    def _original_for_processed_queue(original_actions: Tensor, processed_actions: Tensor) -> Tensor:
        if len(processed_actions) == 0:
            return original_actions[:0].clone()
        return original_actions[: len(processed_actions)].clone()
