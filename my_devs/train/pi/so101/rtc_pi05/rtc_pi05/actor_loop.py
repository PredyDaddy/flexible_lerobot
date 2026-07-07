from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from torch import Tensor

from lerobot.policies.utils import make_robot_action
from lerobot.utils.robot_utils import precise_sleep

from .action_chunk_queue import ActionChunkQueue
from .config import RuntimeConfig
from .metrics import RuntimeMetrics
from .observation_buffer import ObservationBuffer
from .runtime_state import RuntimeState
from .safety import ActionSafety


@dataclass(frozen=True, slots=True)
class ActorLoopResult:
    iterations: int
    sent_actions: int


def run_actor_loop(
    *,
    config: RuntimeConfig,
    state: RuntimeState,
    metrics: RuntimeMetrics,
    action_queue: ActionChunkQueue,
    observation_buffer: ObservationBuffer,
    robot_io: Any,
    dataset_features: Mapping[str, dict[str, Any]],
    robot_action_processor: Callable[[tuple[dict[str, float], Any]], Any] | None,
    safety: ActionSafety,
    perf_counter: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = precise_sleep,
) -> ActorLoopResult:
    iterations = 0
    sent_actions = 0
    next_tick_s = perf_counter()
    first_chunk_deadline_s = state.start_time_s + config.first_chunk_timeout_s

    while state.running:
        now_s = perf_counter()
        if config.run_time_s > 0 and now_s - state.start_time_s >= config.run_time_s:
            state.request_stop(f"run_time_s elapsed: {config.run_time_s:.3f}s")
            break
        if not state.first_chunk_ready.is_set() and now_s >= first_chunk_deadline_s:
            state.request_stop(f"first chunk timeout after {config.first_chunk_timeout_s:.3f}s")
            break

        sent = False
        empty = False
        try:
            obs = robot_io.get_observation()
            observation_buffer.update(obs, timestamp_s=perf_counter())
            action = None if not state.first_chunk_ready.is_set() else action_queue.pop_processed_action()
            if action is None:
                empty = True
            else:
                robot_action = build_robot_action(
                    action=action,
                    dataset_features=dict(dataset_features),
                    robot_action_processor=robot_action_processor,
                    observation=obs,
                )
                safety.check_tensor(action)
                safety.check_robot_action(robot_action)
                robot_io.send_action(robot_action)
                sent = True
                sent_actions += 1
        except Exception as exc:
            state.record_exception("actor_loop", exc)
            break

        iterations += 1
        state.note_actor_iteration(sent=sent)
        metrics.record_actor_tick(queue_depth=action_queue.depth(), empty=empty)

        next_tick_s += config.control_dt_s
        sleep_s = next_tick_s - perf_counter()
        if sleep_s > 0:
            sleep_fn(sleep_s)
        else:
            next_tick_s = perf_counter()

    return ActorLoopResult(iterations=iterations, sent_actions=sent_actions)


def build_robot_action(
    *,
    action: Tensor,
    dataset_features: dict[str, dict[str, Any]],
    robot_action_processor: Callable[[tuple[dict[str, float], Any]], Any] | None,
    observation: Any,
) -> Any:
    robot_action_dict = make_robot_action(action.unsqueeze(0), dataset_features)
    if robot_action_processor is None:
        return robot_action_dict
    return robot_action_processor((robot_action_dict, observation))
