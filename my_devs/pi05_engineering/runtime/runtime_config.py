from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig

EmptyQueueStrategy = Literal["hold-last-action", "skip-send", "raise"]

_LEGACY_EMPTY_QUEUE_STRATEGY_ALIASES = {
    "raise-error": "raise",
}


@dataclass(slots=True)
class PI05RuntimeConfig:
    """Configuration for the local Phase 1 PI05 chunk runtime.

    This config intentionally stays runtime-scoped. Model structure and robot
    wiring remain in existing repo modules and helpers.
    """

    policy_path: str | Path | None = None
    fps: int = 30
    run_time_s: float = 0.0
    dry_run: bool = False

    actions_per_chunk: int = 8
    queue_low_watermark: int = 0
    queue_max_size: int | None = None

    startup_wait_for_first_chunk: bool = True
    startup_timeout_s: float = 15.0
    empty_queue_strategy: EmptyQueueStrategy = "hold-last-action"
    max_action_delta: float | None = None

    enable_rtc: bool = False
    rtc_execution_horizon: int = 10
    rtc_max_guidance_weight: float = 10.0
    rtc_prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.LINEAR
    rtc_debug: bool = False
    rtc_debug_maxlen: int = 100

    metrics_log_interval: float = 5.0
    latency_window_size: int = 100

    def __post_init__(self) -> None:
        if self.empty_queue_strategy in _LEGACY_EMPTY_QUEUE_STRATEGY_ALIASES:
            self.empty_queue_strategy = _LEGACY_EMPTY_QUEUE_STRATEGY_ALIASES[self.empty_queue_strategy]

        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.run_time_s < 0:
            raise ValueError(f"run_time_s must be non-negative, got {self.run_time_s}")
        if self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")
        if self.queue_low_watermark < 0:
            raise ValueError(f"queue_low_watermark must be non-negative, got {self.queue_low_watermark}")
        if self.queue_max_size is not None and self.queue_max_size <= 0:
            raise ValueError(f"queue_max_size must be positive when set, got {self.queue_max_size}")
        if self.queue_max_size is not None and self.queue_low_watermark > self.queue_max_size:
            raise ValueError(
                "queue_low_watermark cannot be greater than queue_max_size: "
                f"{self.queue_low_watermark} > {self.queue_max_size}"
            )
        if self.startup_timeout_s <= 0:
            raise ValueError(f"startup_timeout_s must be positive, got {self.startup_timeout_s}")
        if self.empty_queue_strategy not in {"hold-last-action", "skip-send", "raise"}:
            raise ValueError(f"Unsupported empty_queue_strategy: {self.empty_queue_strategy}")
        if self.max_action_delta is not None and self.max_action_delta <= 0:
            raise ValueError(f"max_action_delta must be positive when set, got {self.max_action_delta}")
        if self.rtc_execution_horizon <= 0:
            raise ValueError(
                f"rtc_execution_horizon must be positive, got {self.rtc_execution_horizon}"
            )
        if self.rtc_max_guidance_weight <= 0:
            raise ValueError(
                f"rtc_max_guidance_weight must be positive, got {self.rtc_max_guidance_weight}"
            )
        if self.rtc_debug_maxlen <= 0:
            raise ValueError(f"rtc_debug_maxlen must be positive, got {self.rtc_debug_maxlen}")
        if self.metrics_log_interval <= 0:
            raise ValueError(f"metrics_log_interval must be positive, got {self.metrics_log_interval}")
        if self.latency_window_size <= 0:
            raise ValueError(f"latency_window_size must be positive, got {self.latency_window_size}")

    @property
    def step_dt_s(self) -> float:
        return 1.0 / self.fps

    @property
    def should_stop_by_duration(self) -> bool:
        return self.run_time_s > 0

    def build_rtc_config(self) -> RTCConfig:
        """Build the repo RTC config object from runtime-facing knobs."""
        return RTCConfig(
            enabled=self.enable_rtc,
            prefix_attention_schedule=self.rtc_prefix_attention_schedule,
            max_guidance_weight=self.rtc_max_guidance_weight,
            execution_horizon=self.rtc_execution_horizon,
            debug=self.rtc_debug,
            debug_maxlen=self.rtc_debug_maxlen,
        )
