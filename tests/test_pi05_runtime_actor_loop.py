from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from my_devs.pi05_engineering.runtime.actor_loop import (
    ActorLoopResult,
    run_actor_iteration,
    run_actor_loop,
)
from my_devs.pi05_engineering.runtime.metrics import PI05RuntimeMetrics
from my_devs.pi05_engineering.runtime.runtime_config import PI05RuntimeConfig
from my_devs.pi05_engineering.runtime.runtime_state import PI05RuntimeState


@dataclass
class FakeQueueController:
    responses: list[Any]
    qsize_values: list[int]

    def __post_init__(self) -> None:
        self.pop_calls = 0
        self.qsize_calls = 0
        self._last_qsize = self.qsize_values[-1] if self.qsize_values else 0

    def pop_next_action(self) -> Any:
        self.pop_calls += 1
        if not self.responses:
            return None

        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def qsize(self) -> int:
        self.qsize_calls += 1
        if self.qsize_values:
            self._last_qsize = self.qsize_values.pop(0)
        return self._last_qsize


@pytest.fixture
def dataset_features() -> dict[str, dict[str, list[str]]]:
    return {"action": {"names": ["joint_0", "joint_1"]}}


def test_run_actor_iteration_sends_processed_action_and_updates_state_metrics(
    dataset_features: dict[str, dict[str, list[str]]],
) -> None:
    config = PI05RuntimeConfig(startup_wait_for_first_chunk=False)
    runtime_state = PI05RuntimeState()
    metrics = PI05RuntimeMetrics(window_size=8)
    queue_controller = FakeQueueController(
        responses=[torch.tensor([0.25, -0.5], dtype=torch.float32)],
        qsize_values=[1, 0],
    )
    sent_payloads: list[Any] = []
    processor_inputs: list[tuple[dict[str, float], Any]] = []

    def robot_action_processor(payload: tuple[dict[str, float], Any]) -> dict[str, Any]:
        processor_inputs.append(payload)
        action_dict, observation = payload
        return {"action": action_dict, "tag": observation["tag"]}

    result = run_actor_iteration(
        config=config,
        runtime_state=runtime_state,
        metrics=metrics,
        queue_controller=queue_controller,
        dataset_features=dataset_features,
        send_action=sent_payloads.append,
        robot_action_processor=robot_action_processor,
        observation_for_processor={"tag": "offline"},
        time_fn=lambda: 12.5,
    )

    assert result.sent is True
    assert result.action_available is True
    assert result.queue_depth_before == 1
    assert result.queue_depth_after == 0
    assert sent_payloads == [{"action": {"joint_0": 0.25, "joint_1": -0.5}, "tag": "offline"}]
    assert processor_inputs == [({"joint_0": 0.25, "joint_1": -0.5}, {"tag": "offline"})]

    state_snapshot = runtime_state.snapshot()
    metrics_snapshot = metrics.snapshot(now_s=13.0)
    assert state_snapshot.actor_iterations == 1
    assert state_snapshot.has_last_action is True
    assert state_snapshot.last_action_sent_at_s == pytest.approx(12.5)
    assert torch.equal(runtime_state.get_last_sent_action(), torch.tensor([0.25, -0.5]))
    assert metrics_snapshot.total_actor_steps == 1
    assert metrics_snapshot.current_queue_depth == 0


def test_run_actor_iteration_records_starvation_when_queue_returns_none(
    dataset_features: dict[str, dict[str, list[str]]],
) -> None:
    config = PI05RuntimeConfig(startup_wait_for_first_chunk=False)
    runtime_state = PI05RuntimeState()
    metrics = PI05RuntimeMetrics(window_size=8)
    queue_controller = FakeQueueController(responses=[None], qsize_values=[0, 0])
    sent_payloads: list[Any] = []

    result = run_actor_iteration(
        config=config,
        runtime_state=runtime_state,
        metrics=metrics,
        queue_controller=queue_controller,
        dataset_features=dataset_features,
        send_action=sent_payloads.append,
    )

    assert result.sent is False
    assert result.action_available is False
    assert result.stop_requested is False
    assert sent_payloads == []
    assert metrics.snapshot().starvation_count == 1
    assert metrics.snapshot().total_actor_steps == 1
    assert runtime_state.snapshot().running is True
    assert runtime_state.snapshot().actor_iterations == 1


def test_run_actor_iteration_stops_when_queue_raises(
    dataset_features: dict[str, dict[str, list[str]]],
) -> None:
    config = PI05RuntimeConfig(startup_wait_for_first_chunk=False)
    runtime_state = PI05RuntimeState()
    metrics = PI05RuntimeMetrics(window_size=8)
    queue_controller = FakeQueueController(
        responses=[RuntimeError("queue broken")],
        qsize_values=[0],
    )

    result = run_actor_iteration(
        config=config,
        runtime_state=runtime_state,
        metrics=metrics,
        queue_controller=queue_controller,
        dataset_features=dataset_features,
        send_action=lambda _: None,
    )

    assert result.sent is False
    assert result.stop_requested is True
    snapshot = runtime_state.snapshot()
    assert snapshot.running is False
    assert snapshot.stop_reason == "actor_iteration.queue: queue broken"
    assert snapshot.last_error is not None
    assert snapshot.last_error.message == "queue broken"


def test_run_actor_iteration_stops_on_max_action_delta_without_sending(
    dataset_features: dict[str, dict[str, list[str]]],
) -> None:
    config = PI05RuntimeConfig(startup_wait_for_first_chunk=False, max_action_delta=0.2)
    runtime_state = PI05RuntimeState()
    runtime_state.set_last_sent_action(torch.tensor([0.0, 0.0], dtype=torch.float32), timestamp_s=1.0)
    metrics = PI05RuntimeMetrics(window_size=8)
    queue_controller = FakeQueueController(
        responses=[torch.tensor([0.0, 0.5], dtype=torch.float32)],
        qsize_values=[1, 0],
    )
    sent_payloads: list[Any] = []

    result = run_actor_iteration(
        config=config,
        runtime_state=runtime_state,
        metrics=metrics,
        queue_controller=queue_controller,
        dataset_features=dataset_features,
        send_action=sent_payloads.append,
    )

    assert result.sent is False
    assert result.stop_requested is True
    assert sent_payloads == []
    snapshot = runtime_state.snapshot()
    assert snapshot.running is False
    assert snapshot.stop_reason is not None
    assert "max_action_delta exceeded" in snapshot.stop_reason
    assert metrics.snapshot().total_actor_steps == 1
    assert snapshot.actor_iterations == 1


def test_run_actor_loop_waits_for_first_chunk_and_times_out_without_sending(
    dataset_features: dict[str, dict[str, list[str]]],
) -> None:
    config = PI05RuntimeConfig(
        startup_wait_for_first_chunk=True,
        startup_timeout_s=0.01,
    )
    runtime_state = PI05RuntimeState()
    metrics = PI05RuntimeMetrics(window_size=8)
    queue_controller = FakeQueueController(
        responses=[torch.tensor([0.1, 0.2], dtype=torch.float32)],
        qsize_values=[1, 0],
    )
    sent_payloads: list[Any] = []

    result = run_actor_loop(
        config=config,
        runtime_state=runtime_state,
        metrics=metrics,
        queue_controller=queue_controller,
        dataset_features=dataset_features,
        send_action=sent_payloads.append,
    )

    assert isinstance(result, ActorLoopResult)
    assert result.startup_ready is False
    assert result.iterations == 0
    assert sent_payloads == []
    assert queue_controller.pop_calls == 0
    assert runtime_state.snapshot().stop_reason is not None
    assert "startup timeout" in runtime_state.snapshot().stop_reason


def test_run_actor_loop_stops_cleanly_when_send_requests_stop(
    dataset_features: dict[str, dict[str, list[str]]],
) -> None:
    config = PI05RuntimeConfig(startup_wait_for_first_chunk=False, fps=20)
    runtime_state = PI05RuntimeState()
    metrics = PI05RuntimeMetrics(window_size=8)
    queue_controller = FakeQueueController(
        responses=[torch.tensor([0.3, 0.4], dtype=torch.float32)],
        qsize_values=[1, 0],
    )
    sent_payloads: list[Any] = []
    sleep_calls: list[float] = []

    def send_action(payload: Any) -> None:
        sent_payloads.append(payload)
        runtime_state.request_stop("test-stop")

    result = run_actor_loop(
        config=config,
        runtime_state=runtime_state,
        metrics=metrics,
        queue_controller=queue_controller,
        dataset_features=dataset_features,
        send_action=send_action,
        sleep_fn=sleep_calls.append,
    )

    assert result.startup_ready is True
    assert result.iterations == 1
    assert len(sent_payloads) == 1
    assert sent_payloads[0]["joint_0"] == pytest.approx(0.3)
    assert sent_payloads[0]["joint_1"] == pytest.approx(0.4)
    assert runtime_state.snapshot().stop_reason == "test-stop"
    assert sleep_calls == []


def test_run_actor_loop_records_keyboard_interrupt_from_send(
    dataset_features: dict[str, dict[str, list[str]]],
) -> None:
    config = PI05RuntimeConfig(startup_wait_for_first_chunk=False)
    runtime_state = PI05RuntimeState()
    metrics = PI05RuntimeMetrics(window_size=8)
    queue_controller = FakeQueueController(
        responses=[torch.tensor([0.3, 0.4], dtype=torch.float32)],
        qsize_values=[1, 0],
    )

    result = run_actor_loop(
        config=config,
        runtime_state=runtime_state,
        metrics=metrics,
        queue_controller=queue_controller,
        dataset_features=dataset_features,
        send_action=lambda _: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    assert result.startup_ready is True
    assert result.iterations == 1
    snapshot = runtime_state.snapshot()
    assert snapshot.running is False
    assert snapshot.stop_reason == "actor_iteration.send: KeyboardInterrupt"
    assert snapshot.last_error is not None
    assert snapshot.last_error.message == "KeyboardInterrupt"
