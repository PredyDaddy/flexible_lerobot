from __future__ import annotations

import torch

from my_devs.train.pi.so101.rtc_pi05.rtc_pi05.action_chunk_queue import ActionChunk, ActionChunkQueue
from my_devs.train.pi.so101.rtc_pi05.rtc_pi05.actor_loop import run_actor_loop
from my_devs.train.pi.so101.rtc_pi05.rtc_pi05.config import RuntimeConfig
from my_devs.train.pi.so101.rtc_pi05.rtc_pi05.metrics import RuntimeMetrics
from my_devs.train.pi.so101.rtc_pi05.rtc_pi05.observation_buffer import ObservationBuffer
from my_devs.train.pi.so101.rtc_pi05.rtc_pi05.runtime_state import RuntimeState
from my_devs.train.pi.so101.rtc_pi05.rtc_pi05.safety import ActionSafety


class FakeRobotIO:
    def __init__(self) -> None:
        self.sent = []

    def get_observation(self):
        return {"state": torch.tensor([0.0]).numpy()}

    def send_action(self, action):
        self.sent.append(action)


class SequenceClock:
    def __init__(self, values: list[float]) -> None:
        self.values = list(values)

    def __call__(self) -> float:
        if not self.values:
            return 99.0
        return self.values.pop(0)


def test_actor_loop_sends_processed_action_once() -> None:
    cfg = RuntimeConfig(policy_path="dummy", task="task", fps=10, run_time_s=0.05)
    state = RuntimeState()
    state.start_time_s = 0.0
    state.mark_first_chunk_ready()
    metrics = RuntimeMetrics()
    queue = ActionChunkQueue()
    raw = torch.tensor([[1.0]], dtype=torch.float32)
    queue.merge_plain(
        ActionChunk(
            raw_actions=raw,
            processed_actions=raw + 10.0,
            obs_timestamp_s=0.0,
            ready_timestamp_s=0.0,
            drop_steps=0,
            rtc_inference_delay=0,
            source_observation_seq=1,
        )
    )
    robot_io = FakeRobotIO()
    dataset_features = {"action": {"names": ["joint"]}}

    result = run_actor_loop(
        config=cfg,
        state=state,
        metrics=metrics,
        action_queue=queue,
        observation_buffer=ObservationBuffer(),
        robot_io=robot_io,
        dataset_features=dataset_features,
        robot_action_processor=None,
        safety=ActionSafety(),
        perf_counter=SequenceClock([0.0, 0.0, 0.01, 0.02, 0.20]),
        sleep_fn=lambda _seconds: None,
    )

    assert result.sent_actions == 1
    assert robot_io.sent == [{"joint": 11.0}]
