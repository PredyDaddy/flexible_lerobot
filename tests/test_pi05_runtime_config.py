from __future__ import annotations

import pytest

from lerobot.configs.types import RTCAttentionSchedule

from my_devs.pi05_engineering.runtime.runtime_config import PI05RuntimeConfig


def test_empty_queue_strategy_uses_raise_as_canonical_value() -> None:
    cfg = PI05RuntimeConfig(empty_queue_strategy="raise")

    assert cfg.empty_queue_strategy == "raise"


def test_empty_queue_strategy_normalizes_legacy_raise_error_value() -> None:
    cfg = PI05RuntimeConfig(empty_queue_strategy="raise-error")  # type: ignore[arg-type]

    assert cfg.empty_queue_strategy == "raise"


def test_invalid_empty_queue_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported empty_queue_strategy"):
        PI05RuntimeConfig(empty_queue_strategy="boom")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field_name", "kwargs", "match"),
    [
        ("fps", {"fps": 0}, "fps must be positive"),
        ("run_time_s", {"run_time_s": -1.0}, "run_time_s must be non-negative"),
        ("actions_per_chunk", {"actions_per_chunk": 0}, "actions_per_chunk must be positive"),
        ("queue_low_watermark", {"queue_low_watermark": -1}, "queue_low_watermark must be non-negative"),
        ("queue_max_size", {"queue_max_size": 0}, "queue_max_size must be positive"),
        ("startup_timeout_s", {"startup_timeout_s": 0}, "startup_timeout_s must be positive"),
        ("max_action_delta", {"max_action_delta": 0}, "max_action_delta must be positive"),
        ("rtc_execution_horizon", {"rtc_execution_horizon": 0}, "rtc_execution_horizon must be positive"),
        ("rtc_max_guidance_weight", {"rtc_max_guidance_weight": 0}, "rtc_max_guidance_weight must be positive"),
        ("rtc_debug_maxlen", {"rtc_debug_maxlen": 0}, "rtc_debug_maxlen must be positive"),
        ("metrics_log_interval", {"metrics_log_interval": 0}, "metrics_log_interval must be positive"),
        ("latency_window_size", {"latency_window_size": 0}, "latency_window_size must be positive"),
    ],
)
def test_invalid_scalar_fields_are_rejected(field_name: str, kwargs: dict, match: str) -> None:
    del field_name
    with pytest.raises(ValueError, match=match):
        PI05RuntimeConfig(**kwargs)


def test_queue_low_watermark_cannot_exceed_queue_max_size() -> None:
    with pytest.raises(ValueError, match="queue_low_watermark cannot be greater than queue_max_size"):
        PI05RuntimeConfig(queue_low_watermark=4, queue_max_size=3)


def test_step_dt_and_duration_properties() -> None:
    cfg = PI05RuntimeConfig(fps=25, run_time_s=2.0)

    assert cfg.step_dt_s == pytest.approx(0.04)
    assert cfg.should_stop_by_duration is True


def test_build_rtc_config_reflects_runtime_knobs() -> None:
    cfg = PI05RuntimeConfig(
        enable_rtc=True,
        rtc_execution_horizon=12,
        rtc_max_guidance_weight=7.5,
        rtc_prefix_attention_schedule=RTCAttentionSchedule.EXP,
        rtc_debug=True,
        rtc_debug_maxlen=32,
    )

    rtc_cfg = cfg.build_rtc_config()

    assert rtc_cfg.enabled is True
    assert rtc_cfg.execution_horizon == 12
    assert rtc_cfg.max_guidance_weight == pytest.approx(7.5)
    assert rtc_cfg.prefix_attention_schedule == RTCAttentionSchedule.EXP
    assert rtc_cfg.debug is True
    assert rtc_cfg.debug_maxlen == 32
