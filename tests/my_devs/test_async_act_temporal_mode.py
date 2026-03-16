from __future__ import annotations

import importlib
import threading
from queue import Queue

import torch

from lerobot.async_inference.helpers import TimedAction


def load_temporal_module():
    return importlib.import_module("my_devs.async_act.temporal_ensemble_client")


def make_temporal_client(module):
    client = module.TemporalEnsembleRobotClient.__new__(module.TemporalEnsembleRobotClient)
    client.temporal_chunk_size = 100
    client.force_send_every_step = True
    client._aggregator = module.TemporalEnsembleAggregator(temporal_ensemble_coeff=0.01, chunk_size=100)
    client._temporal_counts = {}
    client.action_queue = Queue()
    client.action_queue_lock = threading.Lock()
    client.latest_action = 0
    client.latest_action_lock = threading.Lock()
    return client


def test_temporal_module_import_is_safe():
    module = load_temporal_module()

    assert hasattr(module, "TemporalEnsembleAggregator")
    assert hasattr(module, "TemporalEnsembleRobotClient")


def test_temporal_overlap_aggregator_matches_weighted_online_update():
    module = load_temporal_module()
    aggregator = module.TemporalEnsembleAggregator(temporal_ensemble_coeff=0.01, chunk_size=100)
    old = torch.tensor([8.0, 9.0], dtype=torch.float32)
    new = torch.tensor([10.0, 13.0], dtype=torch.float32)

    updated, next_count = aggregator.aggregate(old, new, ensemble_count=1)

    weights = torch.exp(-0.01 * torch.arange(100, dtype=torch.float32))
    expected = (old * weights[0] + new * weights[1]) / weights[:2].sum()

    torch.testing.assert_close(updated, expected)
    assert next_count == 2


def test_temporal_overlap_client_updates_queue_state_for_overlap_and_new_actions():
    module = load_temporal_module()
    client = make_temporal_client(module)
    client.latest_action = 4
    client.action_queue.put(
        TimedAction(
            timestamp=1.0,
            timestep=5,
            action=torch.tensor([8.0], dtype=torch.float32),
        )
    )
    client._temporal_counts = {5: 1}

    incoming_actions = [
        TimedAction(
            timestamp=2.0,
            timestep=4,
            action=torch.tensor([99.0], dtype=torch.float32),
        ),
        TimedAction(
            timestamp=2.0,
            timestep=5,
            action=torch.tensor([10.0], dtype=torch.float32),
        ),
        TimedAction(
            timestamp=2.0,
            timestep=6,
            action=torch.tensor([20.0], dtype=torch.float32),
        ),
    ]

    client._aggregate_action_queues(incoming_actions)

    with client.action_queue_lock:
        queued_actions = list(client.action_queue.queue)

    weights = torch.exp(-0.01 * torch.arange(100, dtype=torch.float32))
    expected_overlap = (
        torch.tensor([8.0], dtype=torch.float32) * weights[0]
        + torch.tensor([10.0], dtype=torch.float32) * weights[1]
    ) / weights[:2].sum()

    assert [action.get_timestep() for action in queued_actions] == [5, 6]
    torch.testing.assert_close(queued_actions[0].get_action(), expected_overlap)
    torch.testing.assert_close(queued_actions[1].get_action(), torch.tensor([20.0], dtype=torch.float32))
    assert client._temporal_counts == {5: 2, 6: 1}
