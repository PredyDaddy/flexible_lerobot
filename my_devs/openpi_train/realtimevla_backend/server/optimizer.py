from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any


class BaseOptimizer(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, inference_cfg: Any) -> "BaseOptimizer":
        raise NotImplementedError

    @abstractmethod
    def optimize(self, actions: list[list[float]]) -> list[list[float]]:
        raise NotImplementedError


@dataclass
class PassThroughOptimizer(BaseOptimizer):
    @classmethod
    def from_config(cls, inference_cfg: Any) -> "PassThroughOptimizer":
        return cls()

    def optimize(self, actions: list[list[float]]) -> list[list[float]]:
        return actions

