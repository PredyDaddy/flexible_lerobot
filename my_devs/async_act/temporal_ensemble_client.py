from __future__ import annotations

from queue import Queue
from typing import Any

import torch

try:
    from lerobot.async_inference.helpers import TimedAction
    from lerobot.async_inference.robot_client import RobotClient
    _ROBOT_CLIENT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-specific import mismatch
    TimedAction = Any  # type: ignore[assignment]
    RobotClient = object  # type: ignore[assignment,misc]
    _ROBOT_CLIENT_IMPORT_ERROR = exc


class TemporalEnsembleAggregator:
    """Online ACT temporal-ensemble update for a single overlapping timestep."""

    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int):
        self.temporal_ensemble_coeff = temporal_ensemble_coeff
        self.chunk_size = int(chunk_size)
        self._ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(self.chunk_size))
        self._ensemble_weights_cumsum = torch.cumsum(self._ensemble_weights, dim=0)

    def aggregate(
        self,
        old_action: torch.Tensor,
        new_action: torch.Tensor,
        ensemble_count: int,
    ) -> tuple[torch.Tensor, int]:
        if self.chunk_size <= 1:
            return new_action, 1

        clamped_count = max(1, min(int(ensemble_count), self.chunk_size - 1))
        weights = self._ensemble_weights.to(device=new_action.device, dtype=new_action.dtype)
        weights_cumsum = self._ensemble_weights_cumsum.to(device=new_action.device, dtype=new_action.dtype)
        old_weight_sum = weights_cumsum[clamped_count - 1]
        new_weight = weights[clamped_count]
        total_weight = weights_cumsum[clamped_count]
        aggregated = (old_action * old_weight_sum + new_action * new_weight) / total_weight
        return aggregated, min(clamped_count + 1, self.chunk_size)


class TemporalEnsembleRobotClient(RobotClient):
    """Robot client with ACT-style temporal ensembling over overlapping action chunks."""

    def __init__(
        self,
        config,
        *,
        temporal_ensemble_coeff: float,
        temporal_chunk_size: int,
        force_send_every_step: bool = True,
    ):
        if _ROBOT_CLIENT_IMPORT_ERROR is not None:  # pragma: no cover - guarded by runtime environment
            raise RuntimeError(_ROBOT_CLIENT_IMPORT_ERROR)
        super().__init__(config)
        self.temporal_ensemble_coeff = temporal_ensemble_coeff
        self.temporal_chunk_size = temporal_chunk_size
        self.force_send_every_step = force_send_every_step
        self._temporal_counts: dict[int, int] = {}
        self._aggregator = TemporalEnsembleAggregator(temporal_ensemble_coeff, temporal_chunk_size)
        self.logger.info(
            "Temporal ensemble mode enabled | coeff=%s | chunk_size=%s | force_send_every_step=%s",
            temporal_ensemble_coeff,
            temporal_chunk_size,
            force_send_every_step,
        )

    def _ready_to_send_observation(self):
        if self.force_send_every_step:
            return True
        return super()._ready_to_send_observation()

    def control_loop_action(self, verbose: bool = False):
        performed = super().control_loop_action(verbose)
        with self.latest_action_lock:
            latest_action = self.latest_action
        self._temporal_counts.pop(latest_action, None)
        return performed

    def _aggregate_action_queues(self, incoming_actions: list[TimedAction], aggregate_fn=None):  # noqa: ARG002
        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = list(self.action_queue.queue)

        current_actions = {action.get_timestep(): action for action in internal_queue}
        next_counts: dict[int, int] = {}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            timestep = new_action.get_timestep()
            if timestep <= latest_action:
                continue

            if timestep not in current_actions:
                future_action_queue.put(new_action)
                next_counts[timestep] = 1
                continue

            previous_action = current_actions[timestep]
            current_count = self._temporal_counts.get(timestep, 1)
            aggregated, next_count = self._aggregator.aggregate(
                previous_action.get_action(),
                new_action.get_action(),
                current_count,
            )
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=timestep,
                    action=aggregated,
                )
            )
            next_counts[timestep] = next_count

        with self.action_queue_lock:
            self.action_queue = future_action_queue
        self._temporal_counts = next_counts
