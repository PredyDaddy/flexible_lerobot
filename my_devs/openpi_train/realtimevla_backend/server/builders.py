from __future__ import annotations

from config import Config
from model import BaseModelAdapter
from model import MockSO101Adapter
from model import OpenPISO101Adapter
from optimizer import BaseOptimizer
from optimizer import PassThroughOptimizer


def build_model(cfg: Config) -> BaseModelAdapter:
    adapter_map = {
        "mock_so101": MockSO101Adapter,
        "openpi_so101": OpenPISO101Adapter,
    }
    adapter_cls = adapter_map.get(cfg.model.adapter)
    if adapter_cls is None:
        raise ValueError(f"Unsupported model.adapter={cfg.model.adapter!r}")
    return adapter_cls.from_config(cfg.model)


def build_optimizer(cfg: Config) -> BaseOptimizer:
    optimizer_map = {
        "pass_through": PassThroughOptimizer,
    }
    optimizer_cls = optimizer_map.get(cfg.inference.optimizer)
    if optimizer_cls is None:
        raise ValueError(f"Unsupported inference.optimizer={cfg.inference.optimizer!r}")
    return optimizer_cls.from_config(cfg.inference)

