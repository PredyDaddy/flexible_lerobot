from __future__ import annotations

import threading
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ActionChunk:
    raw_actions: Tensor
    processed_actions: Tensor
    obs_timestamp_s: float
    ready_timestamp_s: float
    drop_steps: int
    rtc_inference_delay: int
    source_observation_seq: int


@dataclass(frozen=True, slots=True)
class MergeResult:
    mode: str
    input_steps: int
    dropped_steps: int
    enqueued_steps: int
    queue_depth_after: int
    dropped_all: bool


@dataclass(frozen=True, slots=True)
class QueueSnapshot:
    depth: int
    action_cursor: int
    has_last_action: bool
    raw_leftover_len: int
    empty_events: int
    hold_last_events: int
    skip_send_events: int
    stop_events: int
    popped_actions: int
    merge_plain_calls: int
    merge_rtc_calls: int
    dropped_all_chunks: int


class ActionChunkQueue:
    """Owns raw RTC leftovers and processed executable actions with one cursor."""

    def __init__(self, *, empty_queue_strategy: str = "hold-last-action", max_queue_size: int = 50) -> None:
        if empty_queue_strategy not in {"hold-last-action", "skip-send", "stop"}:
            raise ValueError(f"Unsupported empty_queue_strategy: {empty_queue_strategy}")
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        self.empty_queue_strategy = empty_queue_strategy
        self.max_queue_size = int(max_queue_size)
        self._raw_queue: Tensor | None = None
        self._processed_queue: Tensor | None = None
        self._cursor = 0
        self._absolute_cursor = 0
        self._last_action: Tensor | None = None
        self._lock = threading.RLock()
        self._empty_events = 0
        self._hold_last_events = 0
        self._skip_send_events = 0
        self._stop_events = 0
        self._popped_actions = 0
        self._merge_plain_calls = 0
        self._merge_rtc_calls = 0
        self._dropped_all_chunks = 0

    def depth(self) -> int:
        with self._lock:
            return self._depth_locked()

    def action_cursor(self) -> int:
        with self._lock:
            return self._absolute_cursor

    def get_raw_leftover(self) -> Tensor | None:
        with self._lock:
            return self._raw_leftover_locked()

    def pop_processed_action(self) -> Tensor | None:
        with self._lock:
            if self._processed_queue is not None and self._cursor < len(self._processed_queue):
                action = self._processed_queue[self._cursor].detach().clone()
                self._cursor += 1
                self._absolute_cursor += 1
                self._popped_actions += 1
                self._last_action = action.detach().clone()
                if self._cursor > 0 and self._cursor >= len(self._processed_queue):
                    self._raw_queue = None
                    self._processed_queue = None
                    self._cursor = 0
                return action

            self._empty_events += 1
            if self.empty_queue_strategy == "hold-last-action" and self._last_action is not None:
                self._hold_last_events += 1
                return self._last_action.detach().clone()
            if self.empty_queue_strategy == "skip-send" or self.empty_queue_strategy == "hold-last-action":
                self._skip_send_events += 1
                return None
            self._stop_events += 1
            raise RuntimeError("Action queue is empty and empty_queue_strategy='stop'.")

    def merge_plain(self, chunk: ActionChunk) -> MergeResult:
        with self._lock:
            self._merge_plain_calls += 1
            raw, processed, dropped_steps = self._trim_chunk(chunk)
            input_steps = int(chunk.raw_actions.shape[0])
            if len(processed) == 0:
                self._dropped_all_chunks += 1
                return MergeResult("plain", input_steps, dropped_steps, 0, self._depth_locked(), True)

            existing_raw = self._raw_leftover_locked()
            existing_processed = self._processed_leftover_locked()
            raw = self._cat_optional(existing_raw, raw)
            processed = self._cat_optional(existing_processed, processed)
            raw, processed = self._cap_to_max_size(raw, processed)
            self._raw_queue = raw
            self._processed_queue = processed
            self._cursor = 0
            return MergeResult("plain", input_steps, dropped_steps, len(processed), self._depth_locked(), False)

    def merge_rtc(self, chunk: ActionChunk) -> MergeResult:
        with self._lock:
            self._merge_rtc_calls += 1
            raw, processed, dropped_steps = self._trim_chunk(chunk)
            input_steps = int(chunk.raw_actions.shape[0])
            raw, processed = self._cap_to_max_size(raw, processed)
            self._raw_queue = raw if len(raw) else None
            self._processed_queue = processed if len(processed) else None
            self._cursor = 0
            dropped_all = len(processed) == 0
            if dropped_all:
                self._dropped_all_chunks += 1
            return MergeResult("rtc", input_steps, dropped_steps, len(processed), self._depth_locked(), dropped_all)

    def snapshot(self) -> QueueSnapshot:
        with self._lock:
            raw_leftover = self._raw_leftover_locked()
            return QueueSnapshot(
                depth=self._depth_locked(),
                action_cursor=self._absolute_cursor,
                has_last_action=self._last_action is not None,
                raw_leftover_len=0 if raw_leftover is None else len(raw_leftover),
                empty_events=self._empty_events,
                hold_last_events=self._hold_last_events,
                skip_send_events=self._skip_send_events,
                stop_events=self._stop_events,
                popped_actions=self._popped_actions,
                merge_plain_calls=self._merge_plain_calls,
                merge_rtc_calls=self._merge_rtc_calls,
                dropped_all_chunks=self._dropped_all_chunks,
            )

    def _depth_locked(self) -> int:
        if self._processed_queue is None:
            return 0
        return max(0, len(self._processed_queue) - self._cursor)

    def _raw_leftover_locked(self) -> Tensor | None:
        if self._raw_queue is None:
            return None
        return self._raw_queue[self._cursor :].detach().clone()

    def _processed_leftover_locked(self) -> Tensor | None:
        if self._processed_queue is None:
            return None
        return self._processed_queue[self._cursor :].detach().clone()

    def _trim_chunk(self, chunk: ActionChunk) -> tuple[Tensor, Tensor, int]:
        self._validate_chunk(chunk)
        dropped_steps = min(max(int(chunk.drop_steps), 0), len(chunk.raw_actions))
        raw = chunk.raw_actions[dropped_steps:].detach().clone()
        processed = chunk.processed_actions[dropped_steps:].detach().clone()
        return raw, processed, dropped_steps

    def _validate_chunk(self, chunk: ActionChunk) -> None:
        if chunk.raw_actions.ndim != 2 or chunk.processed_actions.ndim != 2:
            raise ValueError("ActionChunk tensors must be 2D shaped as (time_steps, action_dim)")
        if chunk.raw_actions.shape != chunk.processed_actions.shape:
            raise ValueError(
                "raw_actions and processed_actions must have identical shapes, "
                f"got {tuple(chunk.raw_actions.shape)} vs {tuple(chunk.processed_actions.shape)}"
            )

    @staticmethod
    def _cat_optional(existing: Tensor | None, new: Tensor) -> Tensor:
        if existing is None or len(existing) == 0:
            return new.detach().clone()
        if len(new) == 0:
            return existing.detach().clone()
        return torch.cat([existing.detach().clone(), new.detach().clone()], dim=0)

    def _cap_to_max_size(self, raw: Tensor, processed: Tensor) -> tuple[Tensor, Tensor]:
        if len(processed) <= self.max_queue_size:
            return raw, processed
        return raw[: self.max_queue_size].detach().clone(), processed[: self.max_queue_size].detach().clone()
