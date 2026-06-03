#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ActionSafetyConfig:
    action_dim: int | None = None
    max_abs: float | None = None
    max_delta: float | None = None
    reject_nan: bool = True
    reject_inf: bool = True


class ActionSafetyChecker:
    def __init__(self, config: ActionSafetyConfig):
        self.config = config
        self._last_action: np.ndarray | None = None

    def reset(self) -> None:
        self._last_action = None

    def validate(self, action: np.ndarray | list[float], *, update: bool = True) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.config.action_dim is not None and arr.shape[0] != self.config.action_dim:
            raise ValueError(f"Action dim mismatch: got {arr.shape[0]}, expected {self.config.action_dim}")
        if self.config.reject_nan and np.isnan(arr).any():
            raise ValueError("Action contains NaN")
        if self.config.reject_inf and np.isinf(arr).any():
            raise ValueError("Action contains Inf")
        if self.config.max_abs is not None and np.max(np.abs(arr)) > self.config.max_abs:
            raise ValueError(
                f"Action abs limit exceeded: max_abs={np.max(np.abs(arr)):.6f}, limit={self.config.max_abs}"
            )
        if self.config.max_delta is not None and self._last_action is not None:
            delta = np.max(np.abs(arr - self._last_action))
            if delta > self.config.max_delta:
                raise ValueError(f"Action delta limit exceeded: delta={delta:.6f}, limit={self.config.max_delta}")
        if update:
            self._last_action = arr.copy()
        return arr


def validate_action_chunk(
    chunk: np.ndarray | list[list[float]],
    *,
    action_dim: int | None = None,
    min_len: int = 1,
) -> np.ndarray:
    arr = np.asarray(chunk, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Action chunk must be 2D [T,A], got shape={arr.shape}")
    if arr.shape[0] < min_len:
        raise ValueError(f"Action chunk is too short: got {arr.shape[0]}, min_len={min_len}")
    if action_dim is not None and arr.shape[1] != action_dim:
        raise ValueError(f"Action chunk dim mismatch: got {arr.shape[1]}, expected {action_dim}")
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError("Action chunk contains NaN or Inf")
    return arr

