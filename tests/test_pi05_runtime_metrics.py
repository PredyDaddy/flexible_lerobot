from __future__ import annotations

import pytest

from my_devs.pi05_engineering.runtime.metrics import PI05RuntimeMetrics


def test_metrics_start_empty_and_snapshot_is_consistent() -> None:
    metrics = PI05RuntimeMetrics(window_size=4)

    snapshot = metrics.snapshot(now_s=10.0)

    assert snapshot.uptime_s >= 0.0
    assert snapshot.current_queue_depth == 0
    assert snapshot.max_queue_depth == 0
    assert snapshot.starvation_count == 0
    assert snapshot.refill_count == 0
    assert snapshot.total_inference_count == 0
    assert snapshot.total_actor_steps == 0
    assert snapshot.total_producer_steps == 0
    assert snapshot.first_warmup_latency_s is None
    assert snapshot.latest_original_backlog_length is None
    assert snapshot.latest_executable_backlog_length is None
    assert snapshot.latest_backlog_gap is None
    assert snapshot.latest_delay_error is None
    assert snapshot.delay_mismatch_ratio == pytest.approx(0.0)
    assert snapshot.delay_error_mean == pytest.approx(0.0)
    assert snapshot.delay_error_max_abs == 0
    assert snapshot.latest_inference_overrun_ratio is None
    assert snapshot.total_latency.count == 0


def test_metrics_records_queue_starvation_refill_and_trim() -> None:
    metrics = PI05RuntimeMetrics(window_size=8)

    metrics.record_queue_depth(3)
    metrics.record_queue_depth(-5)
    metrics.record_starvation()
    metrics.record_starvation(2)
    metrics.record_refill(chunk_length=8, queue_depth_after=5, leftover_length=2)
    metrics.record_stale_prefix_trim(0)
    metrics.record_stale_prefix_trim(3)

    snapshot = metrics.snapshot()
    assert snapshot.current_queue_depth == 5
    assert snapshot.max_queue_depth == 5
    assert snapshot.queue_depth_samples == 3
    assert snapshot.starvation_count == 3
    assert snapshot.refill_count == 1
    assert snapshot.latest_chunk_length == 8
    assert snapshot.latest_leftover_length == 2
    assert snapshot.latest_original_backlog_length == 2
    assert snapshot.latest_executable_backlog_length == 5
    assert snapshot.latest_backlog_gap == -3
    assert snapshot.stale_prefix_trim_events == 1
    assert snapshot.stale_prefix_trimmed_actions == 3


def test_metrics_records_inference_series_and_latest_fields() -> None:
    metrics = PI05RuntimeMetrics(window_size=8)

    metrics.record_inference(
        total_s=0.20,
        preprocess_s=0.03,
        model_s=0.12,
        postprocess_s=0.05,
        chunk_length=6,
        leftover_length=1,
        action_index_before_inference=7,
        action_index_delta=3,
        inference_delay=2,
        real_delay=3,
        delay_mismatch=True,
        queue_depth_before=1,
        queue_depth_after=4,
    )

    snapshot = metrics.snapshot()
    assert snapshot.total_inference_count == 1
    assert snapshot.first_warmup_latency_s == pytest.approx(0.20)
    assert snapshot.latest_chunk_length == 6
    assert snapshot.latest_leftover_length == 1
    assert snapshot.latest_original_backlog_length == 1
    assert snapshot.latest_executable_backlog_length == 4
    assert snapshot.latest_backlog_gap == -3
    assert snapshot.latest_action_index_before_inference == 7
    assert snapshot.latest_action_index_delta == 3
    assert snapshot.latest_inference_delay == 2
    assert snapshot.latest_real_delay == 3
    assert snapshot.latest_delay_error == 0
    assert snapshot.delay_mismatch_count == 1
    assert snapshot.delay_mismatch_ratio == pytest.approx(1.0)
    assert snapshot.delay_error_mean == pytest.approx(0.0)
    assert snapshot.delay_error_max_abs == 0
    assert snapshot.latest_inference_overrun_ratio is None
    assert snapshot.current_queue_depth == 4
    assert snapshot.max_queue_depth == 4
    assert snapshot.total_latency.latest_s == pytest.approx(0.20)
    assert snapshot.preprocess_latency.latest_s == pytest.approx(0.03)
    assert snapshot.model_latency.latest_s == pytest.approx(0.12)
    assert snapshot.postprocess_latency.latest_s == pytest.approx(0.05)


def test_metrics_records_actor_and_producer_steps() -> None:
    metrics = PI05RuntimeMetrics(window_size=8)

    metrics.record_producer_iteration(timestamp_s=0.5)
    metrics.record_producer_iteration(timestamp_s=1.0)
    metrics.record_actor_step(timestamp_s=1.0)
    metrics.record_actor_step(timestamp_s=1.5, queue_depth_after=2)

    snapshot = metrics.snapshot()
    assert snapshot.total_producer_steps == 2
    assert snapshot.total_actor_steps == 2
    assert snapshot.current_queue_depth == 2
    assert snapshot.actor_rate_hz == pytest.approx(2.0)
    assert snapshot.producer_rate_hz == pytest.approx(2.0)


def test_metrics_rebase_start_time_updates_uptime_baseline() -> None:
    metrics = PI05RuntimeMetrics(window_size=8)

    rebased = metrics.rebase_start_time(timestamp_s=5.0)

    assert rebased == pytest.approx(5.0)
    assert metrics.snapshot(now_s=8.0).uptime_s == pytest.approx(3.0)


def test_mark_first_chunk_ready_only_applies_once() -> None:
    metrics = PI05RuntimeMetrics(window_size=8)

    assert metrics.mark_first_chunk_ready(timestamp_s=4.0) is True
    assert metrics.mark_first_chunk_ready(timestamp_s=9.0) is False

    snapshot = metrics.snapshot()
    assert snapshot.first_chunk_ready_at_s == pytest.approx(4.0)


def test_metrics_tracks_delay_error_aggregate_fields() -> None:
    metrics = PI05RuntimeMetrics(window_size=8)

    metrics.record_inference(total_s=0.10, action_index_delta=1, real_delay=2, delay_mismatch=True)
    metrics.record_inference(total_s=0.10, action_index_delta=3, real_delay=2, delay_mismatch=False)

    snapshot = metrics.snapshot()
    assert snapshot.total_inference_count == 2
    assert snapshot.latest_delay_error == -1
    assert snapshot.delay_mismatch_count == 1
    assert snapshot.delay_mismatch_ratio == pytest.approx(0.5)
    assert snapshot.delay_error_mean == pytest.approx(0.0)
    assert snapshot.delay_error_max_abs == 1


def test_as_log_dict_contains_expected_keys() -> None:
    metrics = PI05RuntimeMetrics(window_size=8)
    metrics.record_inference(total_s=0.10, preprocess_s=0.01, model_s=0.07, postprocess_s=0.02)
    metrics.record_queue_depth(3)
    metrics.record_starvation()

    log_dict = metrics.as_log_dict()

    assert log_dict["current_queue_depth"] == 3
    assert log_dict["first_warmup_latency_s"] == pytest.approx(0.10)
    assert log_dict["starvation_count"] == 1
    assert log_dict["producer_rate_hz"] == pytest.approx(0.0)
    assert log_dict["delay_mismatch_ratio"] == pytest.approx(0.0)
    assert log_dict["delay_error_mean"] == pytest.approx(0.0)
    assert log_dict["delay_error_max_abs"] == 0
    assert log_dict["latest_original_backlog_length"] is None
    assert log_dict["latest_executable_backlog_length"] is None
    assert log_dict["latency_total_latest_s"] == pytest.approx(0.10)
    assert log_dict["latency_preprocess_latest_s"] == pytest.approx(0.01)
    assert log_dict["latency_model_latest_s"] == pytest.approx(0.07)
    assert log_dict["latency_postprocess_latest_s"] == pytest.approx(0.02)
