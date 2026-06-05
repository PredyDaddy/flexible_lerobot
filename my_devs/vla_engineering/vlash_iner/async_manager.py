#!/usr/bin/env python

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .safety import validate_action_chunk


PredictChunkFn = Callable[[dict, np.ndarray | None], np.ndarray]


@dataclass
class AsyncChunkStats:
    inference_count: int = 0
    last_inference_s: float = 0.0
    last_switch_step: int = 0
    pending_inference: bool = False
    wait_count: int = 0
    switch_count: int = 0
    last_switch_delta_abs_max: float = 0.0
    max_switch_delta_abs_max: float = 0.0
    last_switch_delta_l2: float = 0.0
    max_switch_delta_l2: float = 0.0
    switch_direction_flip_count: int = 0


class AsyncChunkManager:
    """VLASH-style action chunk manager with optional background prefetch."""

    def __init__(
        self,
        predict_chunk_fn: PredictChunkFn,
        *,
        n_action_steps: int,
        overlap_steps: int = 0,
        action_dim: int | None = None,
        background_inference: bool = False,
        blend_steps: int = 0,
        future_state_aware: bool = False,
    ):
        if n_action_steps <= 0:
            raise ValueError("n_action_steps must be positive")
        if overlap_steps < 0:
            raise ValueError("overlap_steps must be non-negative")
        if overlap_steps > n_action_steps:
            raise ValueError("overlap_steps cannot exceed n_action_steps")
        if blend_steps < 0:
            raise ValueError("blend_steps must be non-negative")
        self.predict_chunk_fn = predict_chunk_fn
        self.n_action_steps = int(n_action_steps)
        self.overlap_steps = int(overlap_steps)
        self.action_dim = action_dim
        self.background_inference = bool(background_inference)
        self.blend_steps = min(int(blend_steps), self.n_action_steps)
        self.future_state_aware = bool(future_state_aware)
        self.executor = ThreadPoolExecutor(max_workers=1) if self.background_inference else None
        self.current_chunk: np.ndarray | None = None
        self.next_chunk: np.ndarray | None = None
        self.next_future: Future[np.ndarray] | None = None
        self.chunk_index = 0
        self.total_steps = 0
        self.previous_action: np.ndarray | None = None
        self.previous_delta: np.ndarray | None = None
        self.stats = AsyncChunkStats()

    def reset(self) -> None:
        if self.next_future is not None:
            self.next_future.cancel()
        self.current_chunk = None
        self.next_chunk = None
        self.next_future = None
        self.chunk_index = 0
        self.total_steps = 0
        self.previous_action = None
        self.previous_delta = None
        self.stats = AsyncChunkStats()

    def close(self) -> None:
        self.reset()
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)

    def is_running(self) -> bool:
        return (
            self.current_chunk is not None
            or self.next_chunk is not None
            or self.next_future is not None
        )

    def should_switch_chunk(self) -> bool:
        return self.chunk_index == 0 and self.current_chunk is None and self.next_chunk is not None

    def should_launch_next_inference(self) -> bool:
        if self.current_chunk is None:
            return False
        if self.overlap_steps == 0:
            return False
        if self.next_chunk is not None or self.next_future is not None:
            return False
        return self.chunk_index == self.n_action_steps - self.overlap_steps

    def should_fetch_observation(self) -> bool:
        return (not self.is_running()) or self.should_launch_next_inference()

    def _future_state(self) -> np.ndarray | None:
        if not self.future_state_aware or self.current_chunk is None:
            return None
        final_index = min(self.n_action_steps, self.current_chunk.shape[0]) - 1
        return self.current_chunk[final_index].copy()

    def _predict(self, observation: dict, future_state: np.ndarray | None = None) -> np.ndarray:
        start = time.perf_counter()
        chunk = self.predict_chunk_fn(observation, future_state)
        elapsed = time.perf_counter() - start
        chunk = validate_action_chunk(chunk, action_dim=self.action_dim)
        self.stats.inference_count += 1
        self.stats.last_inference_s = elapsed
        return chunk[: self.n_action_steps].copy()

    def _launch_next_inference(self, observation: dict) -> None:
        if self.next_chunk is not None or self.next_future is not None:
            return

        if self.executor is None:
            self.next_chunk = self._predict(observation, self._future_state())
            self.stats.pending_inference = False
            return

        self.next_future = self.executor.submit(self._predict, dict(observation), self._future_state())
        self.stats.pending_inference = True

    def _collect_next_if_ready(self, *, wait: bool = False) -> None:
        if self.next_future is None:
            self.stats.pending_inference = False
            return
        if not wait and not self.next_future.done():
            self.stats.pending_inference = True
            return

        if wait and not self.next_future.done():
            self.stats.wait_count += 1
        self.next_chunk = self.next_future.result()
        self.next_future = None
        self.stats.pending_inference = False

    def _bootstrap_or_switch(self, observation: dict) -> None:
        if self.current_chunk is None and self.next_chunk is None:
            self.current_chunk = self._predict(observation)
            self.chunk_index = 0
            return
        self._collect_next_if_ready(wait=self.current_chunk is None)
        if self.current_chunk is None and self.next_chunk is not None:
            self.current_chunk = self.next_chunk
            self.next_chunk = None
            self.chunk_index = 0
            self.stats.last_switch_step = self.total_steps

    def _blend_action(self, action: np.ndarray) -> np.ndarray:
        if self.previous_action is None or self.blend_steps <= 0 or self.chunk_index >= self.blend_steps:
            return action

        alpha = (self.chunk_index + 1) / (self.blend_steps + 1)
        return (1.0 - alpha) * self.previous_action + alpha * action

    def _record_switch_discontinuity(self, raw_action: np.ndarray) -> None:
        if self.chunk_index != 0 or self.previous_action is None:
            return
        delta = raw_action - self.previous_action
        delta_abs_max = float(np.max(np.abs(delta)))
        delta_l2 = float(np.linalg.norm(delta))
        self.stats.switch_count += 1
        self.stats.last_switch_delta_abs_max = delta_abs_max
        self.stats.max_switch_delta_abs_max = max(self.stats.max_switch_delta_abs_max, delta_abs_max)
        self.stats.last_switch_delta_l2 = delta_l2
        self.stats.max_switch_delta_l2 = max(self.stats.max_switch_delta_l2, delta_l2)
        if self.previous_delta is not None and np.any(delta * self.previous_delta < 0):
            self.stats.switch_direction_flip_count += 1

    def get_action(self, observation: dict) -> np.ndarray:
        self._bootstrap_or_switch(observation)
        if self.current_chunk is None:
            raise RuntimeError("No current action chunk is available")

        if self.should_launch_next_inference():
            self._launch_next_inference(observation)
        self._collect_next_if_ready(wait=False)

        action = self.current_chunk[self.chunk_index].copy()
        self._record_switch_discontinuity(action)
        action = self._blend_action(action)
        previous_action = self.previous_action.copy() if self.previous_action is not None else None
        self.chunk_index += 1
        self.total_steps += 1
        if previous_action is not None:
            self.previous_delta = action - previous_action
        self.previous_action = action.copy()
        if self.chunk_index >= min(self.n_action_steps, self.current_chunk.shape[0]):
            self.current_chunk = None
            self.chunk_index = 0
        return action
