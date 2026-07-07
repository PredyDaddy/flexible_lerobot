from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig

EmptyQueueStrategy = Literal["hold-last-action", "skip-send", "stop"]


@dataclass(slots=True)
class RuntimeConfig:
    policy_path: Path
    task: str
    fps: int = 30
    run_time_s: float = 0.0
    queue_low_watermark: int = 8
    queue_target_size: int = 24
    max_queue_size: int = 50
    first_chunk_timeout_s: float = 30.0
    enable_rtc: bool = True
    rtc_execution_horizon: int = 10
    rtc_max_guidance_weight: float = 10.0
    rtc_prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.LINEAR
    rtc_debug: bool = False
    rtc_debug_maxlen: int = 100
    empty_queue_strategy: EmptyQueueStrategy = "hold-last-action"
    max_action_delta: float | None = None
    metrics_log_interval_s: float = 5.0
    drop_all_chunk_limit: int = 3
    inference_idle_sleep_s: float = 0.005

    def __post_init__(self) -> None:
        self.policy_path = Path(self.policy_path).expanduser()
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.run_time_s < 0:
            raise ValueError(f"run_time_s must be non-negative, got {self.run_time_s}")
        if self.queue_low_watermark < 0:
            raise ValueError(f"queue_low_watermark must be non-negative, got {self.queue_low_watermark}")
        if self.queue_target_size <= 0:
            raise ValueError(f"queue_target_size must be positive, got {self.queue_target_size}")
        if self.max_queue_size <= 0:
            raise ValueError(f"max_queue_size must be positive, got {self.max_queue_size}")
        if self.queue_low_watermark >= self.max_queue_size:
            raise ValueError("queue_low_watermark must be smaller than max_queue_size")
        if self.queue_target_size > self.max_queue_size:
            raise ValueError("queue_target_size must be <= max_queue_size")
        if self.first_chunk_timeout_s <= 0:
            raise ValueError("first_chunk_timeout_s must be positive")
        if self.rtc_execution_horizon <= 0:
            raise ValueError("rtc_execution_horizon must be positive")
        if self.rtc_max_guidance_weight <= 0:
            raise ValueError("rtc_max_guidance_weight must be positive")
        if self.rtc_debug_maxlen <= 0:
            raise ValueError("rtc_debug_maxlen must be positive")
        if self.empty_queue_strategy not in {"hold-last-action", "skip-send", "stop"}:
            raise ValueError(f"Unsupported empty_queue_strategy: {self.empty_queue_strategy}")
        if self.max_action_delta is not None and self.max_action_delta <= 0:
            raise ValueError("max_action_delta must be positive when set")
        if self.metrics_log_interval_s <= 0:
            raise ValueError("metrics_log_interval_s must be positive")
        if self.drop_all_chunk_limit <= 0:
            raise ValueError("drop_all_chunk_limit must be positive")
        if self.inference_idle_sleep_s < 0:
            raise ValueError("inference_idle_sleep_s must be non-negative")

    @property
    def control_dt_s(self) -> float:
        return 1.0 / self.fps

    def build_rtc_config(self) -> RTCConfig:
        return RTCConfig(
            enabled=self.enable_rtc,
            prefix_attention_schedule=self.rtc_prefix_attention_schedule,
            max_guidance_weight=self.rtc_max_guidance_weight,
            execution_horizon=self.rtc_execution_horizon,
            debug=self.rtc_debug,
            debug_maxlen=self.rtc_debug_maxlen,
        )
