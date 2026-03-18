from __future__ import annotations

import pytest

from my_devs.pi05_engineering.runtime.runtime_state import PI05RuntimeState


def test_request_stop_is_idempotent_and_keeps_first_reason() -> None:
    state = PI05RuntimeState()

    assert state.running is True
    assert state.request_stop("first-stop") is True
    assert state.request_stop("second-stop") is False

    snapshot = state.snapshot()
    assert snapshot.running is False
    assert snapshot.stop_reason == "first-stop"
    assert snapshot.stop_requested_at_s is not None


def test_mark_first_chunk_ready_only_sets_once() -> None:
    state = PI05RuntimeState()

    assert state.mark_first_chunk_ready(timestamp_s=1.25) is True
    assert state.mark_first_chunk_ready(timestamp_s=2.50) is False
    assert state.wait_for_first_chunk(timeout_s=0.0) is True

    snapshot = state.snapshot()
    assert snapshot.first_chunk_ready is True
    assert snapshot.first_chunk_ready_at_s == pytest.approx(1.25)


def test_last_sent_action_is_cloned_on_set_and_get() -> None:
    state = PI05RuntimeState()
    action = {"joint_0": [1.0, 2.0]}

    state.set_last_sent_action(action, timestamp_s=3.0)
    action["joint_0"].append(99.0)

    stored = state.get_last_sent_action()
    stored["joint_0"].append(77.0)

    snapshot = state.snapshot()
    assert snapshot.has_last_action is True
    assert snapshot.last_action_sent_at_s == pytest.approx(3.0)
    assert state.get_last_sent_action()["joint_0"] == [1.0, 2.0]


def test_note_actor_and_producer_iterations_update_snapshot() -> None:
    state = PI05RuntimeState()

    state.note_producer_iteration(action_index_before_inference=4, timestamp_s=2.5)
    state.note_actor_iteration(action={"joint_1": 0.5}, timestamp_s=5.0)

    snapshot = state.snapshot()
    assert snapshot.producer_iterations == 1
    assert snapshot.actor_iterations == 1
    assert snapshot.last_producer_iteration_at_s == pytest.approx(2.5)
    assert snapshot.last_action_index_before_inference == 4
    assert snapshot.last_action_sent_at_s == pytest.approx(5.0)
    assert state.get_last_sent_action() == {"joint_1": 0.5}


def test_rebase_start_time_updates_snapshot() -> None:
    state = PI05RuntimeState()

    rebased = state.rebase_start_time(timestamp_s=12.5)

    assert rebased == pytest.approx(12.5)
    assert state.snapshot().start_time_s == pytest.approx(12.5)


def test_note_chunk_inference_records_rtc_state_chain_with_cloned_values() -> None:
    state = PI05RuntimeState()
    prev_left_over = [{"joint_0": 1.0}]
    original_actions = [{"joint_0": 2.0}, {"joint_0": 3.0}]
    processed_actions = [{"joint_0": 4.0}]

    state.note_chunk_inference(
        prev_chunk_left_over=prev_left_over,
        original_actions=original_actions,
        processed_actions=processed_actions,
        action_index_before_inference=7,
        action_index_delta=2,
        inference_delay=1,
        real_delay=2,
        merge_mode="rtc",
        queue_depth_before_merge=0,
        queue_depth_after_merge=1,
        trimmed_prefix_steps=2,
        enqueued_steps=1,
        finished_at_s=8.5,
    )

    prev_left_over.append({"joint_0": 9.0})
    original_actions.append({"joint_0": 10.0})
    processed_actions.append({"joint_0": 11.0})

    stored_original = state.get_last_original_actions()
    stored_original.append({"joint_0": 12.0})

    snapshot = state.snapshot()
    assert snapshot.last_action_index_before_inference == 7
    assert snapshot.last_action_index_delta == 2
    assert snapshot.last_inference_delay == 1
    assert snapshot.last_real_delay == 2
    assert snapshot.last_merge_mode == "rtc"
    assert snapshot.last_queue_depth_before_merge == 0
    assert snapshot.last_queue_depth_after_merge == 1
    assert snapshot.last_trimmed_prefix_steps == 2
    assert snapshot.last_enqueued_steps == 1
    assert snapshot.last_chunk_finished_at_s == pytest.approx(8.5)
    assert snapshot.has_prev_chunk_left_over is True
    assert snapshot.last_prev_chunk_left_over_length == 1
    assert snapshot.has_original_actions is True
    assert snapshot.last_original_actions_length == 2
    assert snapshot.has_processed_actions is True
    assert snapshot.last_processed_actions_length == 1
    assert state.get_last_prev_chunk_left_over() == [{"joint_0": 1.0}]
    assert state.get_last_original_actions() == [{"joint_0": 2.0}, {"joint_0": 3.0}]
    assert state.get_last_processed_actions() == [{"joint_0": 4.0}]


def test_record_exception_stores_error_and_requests_stop() -> None:
    state = PI05RuntimeState()

    try:
        raise RuntimeError("broken loop")
    except RuntimeError as exc:
        record = state.record_exception("actor_loop", exc)

    assert record.source == "actor_loop"
    assert record.message == "broken loop"
    assert "RuntimeError: broken loop" in record.traceback_text

    snapshot = state.snapshot()
    assert snapshot.running is False
    assert snapshot.last_error is not None
    assert snapshot.stop_reason == "actor_loop: broken loop"
