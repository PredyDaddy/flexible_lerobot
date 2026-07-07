from __future__ import annotations

import numpy as np

from my_devs.train.pi.so101.rtc_pi05.server.client_runtime import ClientRuntimeConfig, FrameBuffer


def test_client_runtime_config_splits_camera_and_control_fps() -> None:
    config = ClientRuntimeConfig(task="task", camera_fps=30, control_fps=40)

    assert config.camera_dt_s == 1 / 30
    assert config.control_dt_s == 1 / 40


def test_frame_buffer_keeps_observation_frame_snapshot() -> None:
    buffer = FrameBuffer()
    raw = {"joint": [1.0]}
    frame = {"observation.state": np.array([1.0], dtype=np.float32)}

    snapshot = buffer.update(raw_observation=raw, observation_frame=frame, timestamp_s=2.0)
    raw["joint"].append(99.0)
    frame["observation.state"][0] = 99.0

    latest = buffer.latest(timeout_s=0.0)
    assert latest is not None
    assert latest.sequence_id == snapshot.sequence_id
    assert latest.raw_observation == {"joint": [1.0]}
    assert np.allclose(latest.observation_frame["observation.state"], [1.0])
