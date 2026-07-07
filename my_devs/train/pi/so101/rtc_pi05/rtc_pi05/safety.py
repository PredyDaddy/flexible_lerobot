from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


class ActionSafety:
    def __init__(self, *, max_action_delta: float | None = None) -> None:
        self.max_action_delta = max_action_delta
        self._last_action: Tensor | None = None

    def check_tensor(self, action: Tensor) -> None:
        action = torch.as_tensor(action, dtype=torch.float32).detach().clone()
        if self.max_action_delta is not None and self._last_action is not None:
            if self._last_action.shape != action.shape:
                raise ValueError(
                    f"Action shape changed between sends: {tuple(self._last_action.shape)} vs {tuple(action.shape)}"
                )
            max_delta = float(torch.max(torch.abs(action - self._last_action)).item())
            if max_delta > self.max_action_delta:
                raise RuntimeError(
                    f"max_action_delta exceeded: delta={max_delta:.6f}, limit={self.max_action_delta:.6f}"
                )
        self._last_action = action

    def check_robot_action(self, _robot_action: Any) -> None:
        return
