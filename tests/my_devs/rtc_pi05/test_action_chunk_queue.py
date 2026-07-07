from __future__ import annotations

import torch

from my_devs.train.pi.so101.rtc_pi05.rtc_pi05.action_chunk_queue import ActionChunk, ActionChunkQueue


def _chunk(raw_values: list[float], *, drop_steps: int = 0) -> ActionChunk:
    raw = torch.tensor(raw_values, dtype=torch.float32).unsqueeze(1)
    return ActionChunk(
        raw_actions=raw,
        processed_actions=raw + 10.0,
        obs_timestamp_s=1.0,
        ready_timestamp_s=1.1,
        drop_steps=drop_steps,
        rtc_inference_delay=0,
        source_observation_seq=1,
    )


def test_plain_merge_appends_to_unconsumed_raw_leftover() -> None:
    queue = ActionChunkQueue(max_queue_size=10)

    result = queue.merge_plain(_chunk([1.0, 2.0, 3.0], drop_steps=1))

    assert result.mode == "plain"
    assert result.dropped_steps == 1
    assert result.enqueued_steps == 2
    assert queue.depth() == 2
    assert torch.equal(queue.get_raw_leftover(), torch.tensor([[2.0], [3.0]]))
    assert torch.equal(queue.pop_processed_action(), torch.tensor([12.0]))
    assert torch.equal(queue.get_raw_leftover(), torch.tensor([[3.0]]))

    result = queue.merge_plain(_chunk([4.0, 5.0]))

    assert result.enqueued_steps == 3
    assert torch.equal(queue.get_raw_leftover(), torch.tensor([[3.0], [4.0], [5.0]]))
    assert torch.equal(queue.pop_processed_action(), torch.tensor([13.0]))
    assert torch.equal(queue.pop_processed_action(), torch.tensor([14.0]))
    assert torch.equal(queue.pop_processed_action(), torch.tensor([15.0]))


def test_rtc_merge_replaces_queue_after_stale_prefix_drop() -> None:
    queue = ActionChunkQueue(max_queue_size=10)
    queue.merge_plain(_chunk([1.0, 2.0, 3.0]))
    assert torch.equal(queue.pop_processed_action(), torch.tensor([11.0]))

    result = queue.merge_rtc(_chunk([7.0, 8.0, 9.0, 10.0], drop_steps=2))

    assert result.mode == "rtc"
    assert result.dropped_steps == 2
    assert result.enqueued_steps == 2
    assert queue.depth() == 2
    assert torch.equal(queue.get_raw_leftover(), torch.tensor([[9.0], [10.0]]))
    assert torch.equal(queue.pop_processed_action(), torch.tensor([19.0]))


def test_empty_queue_holds_last_action_when_available() -> None:
    queue = ActionChunkQueue(empty_queue_strategy="hold-last-action")
    queue.merge_plain(_chunk([1.0]))

    assert torch.equal(queue.pop_processed_action(), torch.tensor([11.0]))
    assert torch.equal(queue.pop_processed_action(), torch.tensor([11.0]))

    snapshot = queue.snapshot()
    assert snapshot.empty_events == 1
    assert snapshot.hold_last_events == 1
