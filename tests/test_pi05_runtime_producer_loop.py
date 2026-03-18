from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from my_devs.pi05_engineering.runtime.metrics import PI05RuntimeMetrics
from my_devs.pi05_engineering.runtime.producer_loop import (
    run_producer_iteration,
    run_producer_loop,
)
from my_devs.pi05_engineering.runtime.runtime_config import PI05RuntimeConfig
from my_devs.pi05_engineering.runtime.runtime_state import PI05RuntimeState


def _make_dataset_features() -> dict[str, dict]:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": [2],
            "names": ["joint_1", "joint_2"],
        }
    }


def _make_raw_observation() -> dict[str, object]:
    return {
        "joint_1": 1.25,
        "joint_2": -0.75,
    }


class RecordingObservationProvider:
    def __init__(self, observation: dict[str, object], *, error: BaseException | None = None) -> None:
        self._observation = dict(observation)
        self._error = error
        self.calls = 0

    def __call__(self) -> dict[str, object]:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return dict(self._observation)


class RecordingPreprocessor:
    def __init__(self) -> None:
        self.calls = 0
        self.last_batch: dict[str, object] | None = None

    def __call__(self, batch: dict[str, object]) -> dict[str, object]:
        self.calls += 1
        self.last_batch = dict(batch)
        out = dict(batch)
        out["preprocessed"] = True
        return out


class RecordingPolicy:
    def __init__(self, output: torch.Tensor) -> None:
        self.output = output.clone()
        self.calls = 0
        self.last_batch: dict[str, object] | None = None
        self.last_kwargs: dict[str, object] | None = None
        self.config = SimpleNamespace(device="cpu", use_amp=False)

    def predict_action_chunk(self, batch: dict[str, object], **kwargs: object) -> torch.Tensor:
        self.calls += 1
        self.last_batch = dict(batch)
        self.last_kwargs = dict(kwargs)
        return self.output.clone()


class RecordingPostprocessor:
    def __init__(self, *, delta: float = 10.0) -> None:
        self.delta = delta
        self.calls = 0
        self.last_actions: torch.Tensor | None = None

    def __call__(self, actions: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.last_actions = actions.clone()
        return actions + self.delta


class SequencePerfCounter:
    def __init__(self, values: list[float]) -> None:
        self._values = list(values)
        self._index = 0

    def __call__(self) -> float:
        if self._index >= len(self._values):
            raise AssertionError("perf_counter called more times than expected")
        value = self._values[self._index]
        self._index += 1
        return value


@dataclass(slots=True)
class FakeMergeResult:
    mode: str
    input_steps: int
    trimmed_prefix_steps: int
    enqueued_steps: int
    queue_depth_after: int
    dropped_all: bool = False


class FakeQueueController:
    def __init__(
        self,
        *,
        qsize: int,
        action_index: int = 0,
        action_index_values: list[int] | None = None,
        left_over: torch.Tensor | None = None,
    ) -> None:
        self._qsize = qsize
        self._action_index = action_index
        self._action_index_values = list(action_index_values or [])
        self._left_over = left_over.clone() if left_over is not None else None
        self.plain_calls: list[dict[str, object]] = []
        self.rtc_calls: list[dict[str, object]] = []

    def qsize(self) -> int:
        return self._qsize

    def get_action_index(self) -> int:
        if self._action_index_values:
            self._action_index = self._action_index_values.pop(0)
        return self._action_index

    def get_left_over_original_actions(self) -> torch.Tensor | None:
        return None if self._left_over is None else self._left_over.clone()

    def merge_plain(
        self,
        original_actions: torch.Tensor,
        processed_actions: torch.Tensor,
        *,
        stale_prefix_steps: int = 0,
    ) -> FakeMergeResult:
        trimmed_prefix_steps = min(max(int(stale_prefix_steps), 0), len(original_actions))
        processed_trimmed_steps = min(max(int(stale_prefix_steps), 0), len(processed_actions))
        enqueued_steps = len(processed_actions) - processed_trimmed_steps
        queue_depth_after = self._qsize + enqueued_steps
        self.plain_calls.append(
            {
                "original_actions": original_actions.clone(),
                "processed_actions": processed_actions.clone(),
                "stale_prefix_steps": stale_prefix_steps,
            }
        )
        self._left_over = original_actions[trimmed_prefix_steps:].clone()
        if enqueued_steps > 0:
            self._qsize = queue_depth_after
        return FakeMergeResult(
            mode="plain",
            input_steps=len(original_actions),
            trimmed_prefix_steps=trimmed_prefix_steps,
            enqueued_steps=enqueued_steps,
            queue_depth_after=self._qsize if enqueued_steps == 0 else queue_depth_after,
            dropped_all=enqueued_steps == 0,
        )

    def merge_rtc(
        self,
        original_actions: torch.Tensor,
        processed_actions: torch.Tensor,
        *,
        real_delay: int,
        action_index_before_inference: int | None = 0,
    ) -> FakeMergeResult:
        trimmed_prefix_steps = min(max(int(real_delay), 0), len(original_actions))
        processed_trimmed_steps = min(max(int(real_delay), 0), len(processed_actions))
        enqueued_steps = len(processed_actions) - processed_trimmed_steps
        self.rtc_calls.append(
            {
                "original_actions": original_actions.clone(),
                "processed_actions": processed_actions.clone(),
                "real_delay": real_delay,
                "action_index_before_inference": action_index_before_inference,
            }
        )
        self._left_over = original_actions[trimmed_prefix_steps:].clone()
        self._qsize = enqueued_steps
        return FakeMergeResult(
            mode="rtc",
            input_steps=len(original_actions),
            trimmed_prefix_steps=trimmed_prefix_steps,
            enqueued_steps=enqueued_steps,
            queue_depth_after=enqueued_steps,
            dropped_all=enqueued_steps == 0,
        )


def test_run_producer_iteration_skips_refill_when_queue_is_above_watermark() -> None:
    cfg = PI05RuntimeConfig(queue_low_watermark=1, actions_per_chunk=3)
    state = PI05RuntimeState()
    metrics = PI05RuntimeMetrics(window_size=8)
    queue = FakeQueueController(qsize=2)
    observation_provider = RecordingObservationProvider(_make_raw_observation())
    preprocessor = RecordingPreprocessor()
    policy = RecordingPolicy(output=torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32))
    postprocessor = RecordingPostprocessor()

    result = run_producer_iteration(
        config=cfg,
        state=state,
        metrics=metrics,
        queue_controller=queue,
        observation_provider=observation_provider,
        dataset_features=_make_dataset_features(),
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )

    assert result.refill_required is False
    assert result.inference_ran is False
    assert result.queue_depth_before == 2
    assert result.queue_depth_after == 2
    assert observation_provider.calls == 0
    assert preprocessor.calls == 0
    assert policy.calls == 0
    assert postprocessor.calls == 0
    assert queue.plain_calls == []
    assert queue.rtc_calls == []

    metrics_snapshot = metrics.snapshot()
    assert metrics_snapshot.current_queue_depth == 2
    assert metrics_snapshot.total_inference_count == 0
    assert metrics_snapshot.total_producer_steps == 1
    assert state.snapshot().producer_iterations == 1


def test_run_producer_iteration_respects_plain_queue_capacity_budget() -> None:
    cfg = PI05RuntimeConfig(
        queue_low_watermark=1,
        queue_max_size=2,
        actions_per_chunk=4,
        enable_rtc=False,
    )
    state = PI05RuntimeState()
    metrics = PI05RuntimeMetrics(window_size=8)
    queue = FakeQueueController(qsize=1, action_index=2, action_index_values=[2, 2])
    observation_provider = RecordingObservationProvider(_make_raw_observation())
    preprocessor = RecordingPreprocessor()
    policy = RecordingPolicy(
        output=torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]],
            dtype=torch.float32,
        )
    )
    postprocessor = RecordingPostprocessor()

    result = run_producer_iteration(
        config=cfg,
        state=state,
        metrics=metrics,
        queue_controller=queue,
        observation_provider=observation_provider,
        dataset_features=_make_dataset_features(),
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        perf_counter=SequencePerfCounter([0.99, 1.00, 1.00, 1.00, 1.00, 1.00]),
    )

    assert result.refill_required is True
    assert result.original_chunk_length == 4
    assert result.chunk_length == 1
    assert result.enqueued_steps == 1
    assert len(queue.plain_calls) == 1
    assert tuple(queue.plain_calls[0]["original_actions"].shape) == (4, 2)
    assert tuple(queue.plain_calls[0]["processed_actions"].shape) == (1, 2)


def test_run_producer_iteration_plain_mode_crops_chunk_and_trims_stale_prefix() -> None:
    cfg = PI05RuntimeConfig(fps=10, queue_low_watermark=0, actions_per_chunk=3, enable_rtc=False)
    state = PI05RuntimeState()
    metrics = PI05RuntimeMetrics(window_size=8)
    metrics.record_inference(total_s=0.21)
    queue = FakeQueueController(qsize=0, action_index=5, action_index_values=[5, 7])
    observation_provider = RecordingObservationProvider(_make_raw_observation())
    preprocessor = RecordingPreprocessor()
    policy = RecordingPolicy(
        output=torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]],
            dtype=torch.float32,
        )
    )
    postprocessor = RecordingPostprocessor(delta=100.0)

    result = run_producer_iteration(
        config=cfg,
        state=state,
        metrics=metrics,
        queue_controller=queue,
        observation_provider=observation_provider,
        dataset_features=_make_dataset_features(),
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        task="stack blocks",
        robot_type="so101_follower",
        perf_counter=SequencePerfCounter([0.90, 1.00, 1.02, 1.04, 1.12, 1.16]),
    )

    assert result.refill_required is True
    assert result.inference_ran is True
    assert result.mode == "plain"
    assert result.original_chunk_length == 5
    assert result.chunk_length == 3
    assert result.action_index_delta == 2
    assert result.inference_delay == 3
    assert result.real_delay == 2
    assert result.trimmed_prefix_steps == 2
    assert result.enqueued_steps == 1
    assert result.first_chunk_ready_marked is True
    assert result.total_latency_s == pytest.approx(0.16)
    assert result.preprocess_latency_s == pytest.approx(0.02)
    assert result.model_latency_s == pytest.approx(0.08)
    assert result.postprocess_latency_s == pytest.approx(0.04)
    assert result.total_latency_s - (
        result.preprocess_latency_s + result.model_latency_s + result.postprocess_latency_s
    ) == pytest.approx(0.02)

    assert len(queue.plain_calls) == 1
    assert queue.rtc_calls == []
    plain_call = queue.plain_calls[0]
    assert plain_call["stale_prefix_steps"] == 2
    assert tuple(plain_call["original_actions"].shape) == (5, 2)
    assert tuple(plain_call["processed_actions"].shape) == (3, 2)

    assert preprocessor.last_batch is not None
    assert preprocessor.last_batch["task"] == ["stack blocks"]
    assert preprocessor.last_batch["robot_type"] == "so101_follower"
    assert policy.last_kwargs == {}

    state_snapshot = state.snapshot()
    assert state_snapshot.producer_iterations == 1
    assert state_snapshot.first_chunk_ready is True
    assert state_snapshot.first_chunk_ready_at_s == pytest.approx(1.16)
    assert state_snapshot.last_action_index_before_inference == 5
    assert state_snapshot.last_action_index_delta == 2
    assert state_snapshot.last_inference_delay == 3
    assert state_snapshot.last_real_delay == 2
    assert state_snapshot.last_original_actions_length == 5
    assert state_snapshot.last_processed_actions_length == 3

    metrics_snapshot = metrics.snapshot()
    assert metrics_snapshot.total_producer_steps == 1
    assert metrics_snapshot.refill_count == 1
    assert metrics_snapshot.total_inference_count == 2
    assert metrics_snapshot.first_warmup_latency_s == pytest.approx(0.21)
    assert metrics_snapshot.latest_chunk_length == 3
    assert metrics_snapshot.latest_leftover_length == 3
    assert metrics_snapshot.latest_original_backlog_length == 3
    assert metrics_snapshot.latest_executable_backlog_length == 1
    assert metrics_snapshot.latest_backlog_gap == 2
    assert metrics_snapshot.latest_action_index_before_inference == 5
    assert metrics_snapshot.latest_action_index_delta == 2
    assert metrics_snapshot.latest_inference_delay == 3
    assert metrics_snapshot.latest_real_delay == 2
    assert metrics_snapshot.latest_delay_error == 0
    assert metrics_snapshot.delay_mismatch_count == 0
    assert metrics_snapshot.delay_mismatch_ratio == pytest.approx(0.0)
    assert metrics_snapshot.delay_error_mean == pytest.approx(0.0)
    assert metrics_snapshot.delay_error_max_abs == 0
    assert metrics_snapshot.latest_inference_overrun_ratio == pytest.approx(1.6)
    assert metrics_snapshot.stale_prefix_trim_events == 1
    assert metrics_snapshot.stale_prefix_trimmed_actions == 2
    assert metrics_snapshot.first_chunk_ready_at_s == pytest.approx(1.16)
    assert metrics_snapshot.total_latency.latest_s == pytest.approx(0.16)
    assert metrics_snapshot.preprocess_latency.latest_s == pytest.approx(0.02)
    assert metrics_snapshot.model_latency.latest_s == pytest.approx(0.08)
    assert metrics_snapshot.postprocess_latency.latest_s == pytest.approx(0.04)


def test_run_producer_iteration_rtc_mode_forwards_plumbing_and_uses_rtc_merge() -> None:
    cfg = PI05RuntimeConfig(
        fps=10,
        queue_low_watermark=0,
        actions_per_chunk=2,
        enable_rtc=True,
        rtc_execution_horizon=8,
    )
    state = PI05RuntimeState()
    metrics = PI05RuntimeMetrics(window_size=8)
    metrics.record_inference(total_s=0.09)
    prev_left_over = torch.tensor([[0.5, 0.6], [0.7, 0.8]], dtype=torch.float32)
    queue = FakeQueueController(qsize=0, action_index=7, action_index_values=[7, 9], left_over=prev_left_over)
    observation_provider = RecordingObservationProvider(_make_raw_observation())
    preprocessor = RecordingPreprocessor()
    policy = RecordingPolicy(
        output=torch.tensor(
            [[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]]],
            dtype=torch.float32,
        )
    )
    postprocessor = RecordingPostprocessor(delta=5.0)

    result = run_producer_iteration(
        config=cfg,
        state=state,
        metrics=metrics,
        queue_controller=queue,
        observation_provider=observation_provider,
        dataset_features=_make_dataset_features(),
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        perf_counter=SequencePerfCounter([1.90, 2.00, 2.01, 2.03, 2.13, 2.18]),
    )

    assert result.mode == "rtc"
    assert result.original_chunk_length == 4
    assert result.chunk_length == 2
    assert result.action_index_delta == 2
    assert result.inference_delay == 1
    assert result.real_delay == 2
    assert result.trimmed_prefix_steps == 2
    assert result.enqueued_steps == 0
    assert result.did_refill is False

    assert len(queue.rtc_calls) == 1
    assert queue.plain_calls == []
    rtc_call = queue.rtc_calls[0]
    assert rtc_call["real_delay"] == 2
    assert rtc_call["action_index_before_inference"] == 7
    assert tuple(rtc_call["original_actions"].shape) == (4, 2)
    assert tuple(rtc_call["processed_actions"].shape) == (2, 2)

    assert policy.last_kwargs is not None
    assert policy.last_kwargs["inference_delay"] == 1
    assert policy.last_kwargs["execution_horizon"] == 8
    assert torch.equal(policy.last_kwargs["prev_chunk_left_over"], prev_left_over)

    metrics_snapshot = metrics.snapshot()
    assert metrics_snapshot.total_producer_steps == 1
    assert metrics_snapshot.total_inference_count == 2
    assert metrics_snapshot.refill_count == 0
    assert metrics_snapshot.latest_original_backlog_length == 2
    assert metrics_snapshot.latest_executable_backlog_length == 0
    assert metrics_snapshot.latest_backlog_gap == 2
    assert metrics_snapshot.latest_action_index_before_inference == 7
    assert metrics_snapshot.latest_action_index_delta == 2
    assert metrics_snapshot.latest_inference_delay == 1
    assert metrics_snapshot.latest_real_delay == 2
    assert metrics_snapshot.latest_leftover_length == 2
    assert metrics_snapshot.latest_delay_error == 0
    assert metrics_snapshot.delay_mismatch_ratio == pytest.approx(0.0)
    assert metrics_snapshot.latest_inference_overrun_ratio == pytest.approx(1.8)
    assert metrics_snapshot.first_chunk_ready_at_s is None
    state_snapshot = state.snapshot()
    assert state_snapshot.first_chunk_ready is False
    assert state_snapshot.last_prev_chunk_left_over_length == 2
    assert state_snapshot.last_original_actions_length == 4
    assert state_snapshot.last_processed_actions_length == 2


def test_run_producer_iteration_does_not_mark_first_chunk_ready_when_chunk_is_fully_trimmed() -> None:
    cfg = PI05RuntimeConfig(fps=10, queue_low_watermark=0, actions_per_chunk=2, enable_rtc=False)
    state = PI05RuntimeState()
    metrics = PI05RuntimeMetrics(window_size=8)
    queue = FakeQueueController(qsize=0, action_index=3, action_index_values=[3, 7])
    observation_provider = RecordingObservationProvider(_make_raw_observation())
    preprocessor = RecordingPreprocessor()
    policy = RecordingPolicy(output=torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32))
    postprocessor = RecordingPostprocessor()

    result = run_producer_iteration(
        config=cfg,
        state=state,
        metrics=metrics,
        queue_controller=queue,
        observation_provider=observation_provider,
        dataset_features=_make_dataset_features(),
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        perf_counter=SequencePerfCounter([4.90, 5.00, 5.01, 5.05, 5.15, 5.31]),
    )

    assert result.dropped_all is True
    assert result.action_index_delta == 4
    assert result.real_delay == 4
    assert result.enqueued_steps == 0
    assert result.first_chunk_ready_marked is False
    assert state.snapshot().first_chunk_ready is False

    metrics_snapshot = metrics.snapshot()
    assert metrics_snapshot.refill_count == 0
    assert metrics_snapshot.total_inference_count == 1
    assert metrics_snapshot.stale_prefix_trim_events == 1
    assert metrics_snapshot.stale_prefix_trimmed_actions == 2
    assert metrics_snapshot.first_chunk_ready_at_s is None
    assert metrics_snapshot.latest_action_index_delta == 4
    assert metrics_snapshot.latest_delay_error == 0
    assert metrics_snapshot.latest_inference_overrun_ratio == pytest.approx(3.1)


def test_run_producer_iteration_warns_when_action_index_delta_disagrees_with_real_delay(caplog) -> None:
    cfg = PI05RuntimeConfig(fps=10, queue_low_watermark=0, actions_per_chunk=2, enable_rtc=True)
    state = PI05RuntimeState()
    metrics = PI05RuntimeMetrics(window_size=8)
    queue = FakeQueueController(qsize=0, action_index=4, action_index_values=[4, 5])
    observation_provider = RecordingObservationProvider(_make_raw_observation())
    preprocessor = RecordingPreprocessor()
    policy = RecordingPolicy(output=torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=torch.float32))
    postprocessor = RecordingPostprocessor()

    with caplog.at_level("WARNING"):
        result = run_producer_iteration(
            config=cfg,
            state=state,
            metrics=metrics,
            queue_controller=queue,
            observation_provider=observation_provider,
            dataset_features=_make_dataset_features(),
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            perf_counter=SequencePerfCounter([9.00, 9.10, 9.11, 9.15, 9.25, 9.32]),
        )

    assert result.action_index_delta == 1
    assert result.real_delay == 3
    assert "action_index_delta and real_delay diverged" in caplog.text
    assert metrics.snapshot().delay_mismatch_count == 1
    assert metrics.snapshot().latest_delay_error == 2
    assert metrics.snapshot().delay_mismatch_ratio == pytest.approx(1.0)
    assert state.snapshot().last_action_index_delta == 1


def test_run_producer_iteration_plain_mode_ignores_one_step_phase_gap(caplog) -> None:
    cfg = PI05RuntimeConfig(fps=10, queue_low_watermark=0, actions_per_chunk=2, enable_rtc=False)
    state = PI05RuntimeState()
    state.note_actor_iteration(timestamp_s=1.0)
    metrics = PI05RuntimeMetrics(window_size=8)
    queue = FakeQueueController(qsize=0, action_index=4, action_index_values=[4, 5])
    observation_provider = RecordingObservationProvider(_make_raw_observation())
    preprocessor = RecordingPreprocessor()
    policy = RecordingPolicy(output=torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=torch.float32))
    postprocessor = RecordingPostprocessor()

    with caplog.at_level("WARNING"):
        result = run_producer_iteration(
            config=cfg,
            state=state,
            metrics=metrics,
            queue_controller=queue,
            observation_provider=observation_provider,
            dataset_features=_make_dataset_features(),
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            perf_counter=SequencePerfCounter([9.00, 9.10, 9.11, 9.15, 9.25, 9.26]),
        )

    assert result.action_index_delta == 1
    assert result.real_delay == 2
    assert "action_index_delta and real_delay diverged" not in caplog.text
    assert metrics.snapshot().delay_mismatch_count == 0
    assert metrics.snapshot().latest_delay_error == 1
    assert metrics.snapshot().delay_mismatch_ratio == pytest.approx(0.0)


def test_run_producer_loop_records_exception_and_requests_stop() -> None:
    cfg = PI05RuntimeConfig(queue_low_watermark=0, actions_per_chunk=2)
    state = PI05RuntimeState()
    metrics = PI05RuntimeMetrics(window_size=8)
    queue = FakeQueueController(qsize=0)
    observation_provider = RecordingObservationProvider(
        _make_raw_observation(),
        error=RuntimeError("observation failed"),
    )
    preprocessor = RecordingPreprocessor()
    policy = RecordingPolicy(output=torch.tensor([[[1.0, 2.0]]], dtype=torch.float32))
    postprocessor = RecordingPostprocessor()

    completed_iterations = run_producer_loop(
        config=cfg,
        state=state,
        metrics=metrics,
        queue_controller=queue,
        observation_provider=observation_provider,
        dataset_features=_make_dataset_features(),
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        sleep_interval_s=0.0,
        max_iterations=3,
    )

    assert completed_iterations == 0
    state_snapshot = state.snapshot()
    assert state_snapshot.running is False
    assert state_snapshot.stop_reason == "producer_loop: observation failed"
    assert state_snapshot.last_error is not None
    assert state_snapshot.last_error.message == "observation failed"
