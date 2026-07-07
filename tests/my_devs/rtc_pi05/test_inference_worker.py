from __future__ import annotations

import torch

from my_devs.train.pi.so101.rtc_pi05.rtc_pi05.action_chunk_queue import ActionChunkQueue
from my_devs.train.pi.so101.rtc_pi05.rtc_pi05.config import RuntimeConfig
from my_devs.train.pi.so101.rtc_pi05.rtc_pi05.inference_worker import run_inference_once
from my_devs.train.pi.so101.rtc_pi05.rtc_pi05.metrics import RuntimeMetrics
from my_devs.train.pi.so101.rtc_pi05.rtc_pi05.observation_buffer import ObservationBuffer
from my_devs.train.pi.so101.rtc_pi05.rtc_pi05.runtime_state import RuntimeState


class FakePolicy:
    def __init__(self) -> None:
        self.config = type("Config", (), {"use_amp": False})()
        self.calls = 0
        self.last_kwargs = None

    def predict_action_chunk(self, batch, **kwargs):  # noqa: ANN001
        self.calls += 1
        self.last_kwargs = dict(kwargs)
        return torch.tensor([[[1.0], [2.0], [3.0], [4.0]]], dtype=torch.float32)


class SequenceClock:
    def __init__(self, values: list[float]) -> None:
        self.values = list(values)

    def __call__(self) -> float:
        if not self.values:
            raise AssertionError("clock exhausted")
        return self.values.pop(0)


def test_inference_once_passes_rtc_kwargs_and_drops_stale_prefix(monkeypatch) -> None:
    from my_devs.train.pi.so101.rtc_pi05.rtc_pi05 import inference_worker

    monkeypatch.setattr(
        inference_worker,
        "prepare_policy_batch",
        lambda **_kwargs: {"prepared": True},
    )

    cfg = RuntimeConfig(
        policy_path="dummy",
        task="test task",
        fps=10,
        enable_rtc=True,
        queue_low_watermark=8,
    )
    state = RuntimeState()
    metrics = RuntimeMetrics()
    queue = ActionChunkQueue()
    buffer = ObservationBuffer()
    buffer.update({"obs": 1.0}, timestamp_s=1.0)
    policy = FakePolicy()

    result = run_inference_once(
        config=cfg,
        state=state,
        metrics=metrics,
        action_queue=queue,
        observation_buffer=buffer,
        dataset_features={},
        policy=policy,
        preprocessor=lambda batch: batch,
        postprocessor=lambda actions: actions + 10.0,
        robot_observation_processor=None,
        robot_type="so101_follower",
        device=torch.device("cpu"),
        wait_timeout_s=0.0,
        perf_counter=SequenceClock([1.0, 1.15]),
    )

    assert result is not None
    assert result.drop_steps == 2
    assert result.merge_result.enqueued_steps == 2
    assert torch.equal(queue.get_raw_leftover(), torch.tensor([[3.0], [4.0]]))
    assert torch.equal(queue.pop_processed_action(), torch.tensor([13.0]))
    assert policy.last_kwargs == {
        "inference_delay": 0,
        "prev_chunk_left_over": None,
        "execution_horizon": cfg.rtc_execution_horizon,
    }
