from __future__ import annotations

from my_devs.train.pi.so101.rtc_pi05.rtc_pi05.observation_buffer import ObservationBuffer


def test_observation_buffer_deep_copies_latest_snapshot() -> None:
    buffer = ObservationBuffer()
    observation = {"joint": [1.0, 2.0]}

    snapshot = buffer.update(observation, timestamp_s=3.0)
    observation["joint"].append(99.0)
    snapshot.observation["joint"].append(77.0)

    latest = buffer.latest(timeout_s=0.0)

    assert latest is not None
    assert latest.timestamp_s == 3.0
    assert latest.sequence_id == 1
    assert latest.observation == {"joint": [1.0, 2.0]}
