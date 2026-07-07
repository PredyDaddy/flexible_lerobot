from __future__ import annotations

import numpy as np

from my_devs.train.pi.so101.rtc_pi05.server.protocol import (
    InferenceRequest,
    dumps_payload,
    loads_payload,
)


def test_protocol_roundtrip_preserves_numpy_payload() -> None:
    request = InferenceRequest(
        request_id=7,
        observation_frame={"observation.state": np.array([1.0, 2.0], dtype=np.float32)},
        task="task",
        robot_type="so101_follower",
        obs_timestamp_s=1.25,
        obs_sequence_id=3,
        enable_rtc=True,
        predicted_delay_steps=2,
        prev_chunk_left_over=np.array([[0.1], [0.2]], dtype=np.float32),
        execution_horizon=10,
    )

    restored = loads_payload(dumps_payload(request))

    assert restored.request_id == 7
    assert restored.enable_rtc is True
    assert np.allclose(restored.observation_frame["observation.state"], [1.0, 2.0])
    assert np.allclose(restored.prev_chunk_left_over, [[0.1], [0.2]])
