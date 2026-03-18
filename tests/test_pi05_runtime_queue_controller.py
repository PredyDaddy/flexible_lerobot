#!/usr/bin/env python

"""Offline tests for the PI05 Phase 1 queue controller."""

import pytest
import torch

from my_devs.pi05_engineering.runtime.queue_controller import (
    MergeResult,
    QueueController,
)


@pytest.fixture
def sample_actions():
    original = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    processed = original + 100.0
    return original, processed


def test_merge_plain_appends_without_trimming(sample_actions):
    original, processed = sample_actions
    controller = QueueController(enable_rtc=False)

    result = controller.merge_plain(original, processed)

    assert isinstance(result, MergeResult)
    assert result.mode == "plain"
    assert result.trimmed_prefix_steps == 0
    assert result.enqueued_steps == 4
    assert controller.qsize() == 4
    assert torch.equal(controller.get_left_over_original_actions(), original)


def test_merge_plain_trims_stale_prefix_and_tracks_counters(sample_actions):
    original, processed = sample_actions
    controller = QueueController(enable_rtc=False)

    result = controller.merge_plain(original, processed, stale_prefix_steps=2)

    assert result.trimmed_prefix_steps == 2
    assert result.enqueued_steps == 2
    assert controller.qsize() == 2
    assert torch.equal(controller.get_left_over_original_actions(), original[2:])
    counters = controller.counters()
    assert counters.plain_merge_calls == 1
    assert counters.stale_prefix_trim_events == 1
    assert counters.stale_prefix_trimmed_actions == 2


def test_merge_plain_keeps_full_original_chunk_when_processed_prefix_is_shorter(sample_actions):
    original, processed = sample_actions
    controller = QueueController(enable_rtc=False)

    result = controller.merge_plain(original, processed[:2], stale_prefix_steps=1)

    assert result.trimmed_prefix_steps == 1
    assert result.enqueued_steps == 1
    assert controller.qsize() == 1
    assert torch.equal(controller.get_left_over_original_actions(), original[1:])


def test_merge_plain_can_drop_fully_stale_chunk(sample_actions):
    original, processed = sample_actions
    controller = QueueController(enable_rtc=False)

    result = controller.merge_plain(original, processed, stale_prefix_steps=99)

    assert result.dropped_all is True
    assert result.enqueued_steps == 0
    assert controller.qsize() == 0
    counters = controller.counters()
    assert counters.dropped_all_stale_chunks == 1
    assert counters.stale_prefix_trimmed_actions == len(original)


def test_merge_plain_preserves_unconsumed_suffix_when_appending(sample_actions):
    original, processed = sample_actions
    controller = QueueController(enable_rtc=False)
    controller.merge_plain(original, processed)

    first = controller.pop_next_action()
    assert torch.equal(first, processed[0])

    new_original = original + 1000.0
    new_processed = processed + 1000.0
    result = controller.merge_plain(new_original, new_processed, stale_prefix_steps=1)

    assert result.enqueued_steps == 3
    assert controller.qsize() == 6
    leftover = controller.get_left_over_original_actions()
    expected = torch.cat([original[1:], new_original[1:]])
    assert torch.equal(leftover, expected)


def test_merge_rtc_replaces_queue_using_real_delay(sample_actions):
    original, processed = sample_actions
    controller = QueueController(enable_rtc=False)
    controller.merge_plain(original, processed)

    new_original = original + 500.0
    new_processed = processed + 500.0
    result = controller.merge_rtc(
        new_original,
        new_processed,
        real_delay=2,
        action_index_before_inference=0,
    )

    assert result.mode == "rtc"
    assert result.trimmed_prefix_steps == 2
    assert result.enqueued_steps == 2
    assert controller.qsize() == 2
    assert torch.equal(controller.get_left_over_original_actions(), new_original[2:])


def test_pop_next_action_holds_last_action_when_configured(sample_actions):
    original, processed = sample_actions
    controller = QueueController(enable_rtc=False, empty_queue_strategy="hold-last-action")
    controller.merge_plain(original, processed)

    popped = [controller.pop_next_action() for _ in range(4)]
    assert all(action is not None for action in popped)

    held = controller.pop_next_action()

    assert held is not None
    assert torch.equal(held, processed[-1])
    counters = controller.counters()
    assert counters.empty_queue_events == 1
    assert counters.held_last_action_events == 1
    assert counters.skip_send_events == 0


def test_pop_next_action_skips_send_without_last_action(caplog):
    controller = QueueController(enable_rtc=False, empty_queue_strategy="skip-send")

    with caplog.at_level("WARNING"):
        assert controller.pop_next_action() is None

    counters = controller.counters()
    assert counters.empty_queue_events == 1
    assert counters.skip_send_events == 1
    assert "skipping send" in caplog.text


def test_pop_next_action_hold_last_action_falls_back_to_skip_when_no_last_action(caplog):
    controller = QueueController(enable_rtc=False, empty_queue_strategy="hold-last-action")

    with caplog.at_level("WARNING"):
        assert controller.pop_next_action() is None

    counters = controller.counters()
    assert counters.empty_queue_events == 1
    assert counters.held_last_action_events == 0
    assert counters.skip_send_events == 1
    assert "strategy=hold-last-action" in caplog.text


def test_pop_next_action_can_raise_when_configured():
    controller = QueueController(enable_rtc=False, empty_queue_strategy="raise")

    with pytest.raises(RuntimeError, match="Action queue is empty"):
        controller.pop_next_action()


def test_controller_rejects_shape_mismatch(sample_actions):
    original, processed = sample_actions
    controller = QueueController(enable_rtc=False)

    with pytest.raises(ValueError, match="same action dimension"):
        controller.merge_plain(original, processed[:, :-1])


def test_controller_rejects_negative_stale_prefix(sample_actions):
    original, processed = sample_actions
    controller = QueueController(enable_rtc=False)

    with pytest.raises(ValueError, match="stale_prefix_steps"):
        controller.merge_plain(original, processed, stale_prefix_steps=-1)
