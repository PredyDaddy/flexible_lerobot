from __future__ import annotations

import threading

import numpy as np

from my_devs.train.pi.so101.rtc_pi05.server.http_policy_server import make_server
from my_devs.train.pi.so101.rtc_pi05.server.protocol import InferenceRequest, InferenceResponse
from my_devs.train.pi.so101.rtc_pi05.server.remote_policy_client import RemotePolicyClient


class FakeService:
    def __init__(self) -> None:
        self.inference_count = 0
        self.last_latency_s = 0.0
        self.config = type("Config", (), {"enable_rtc": True})()
        self.device = "cpu"

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        self.inference_count += 1
        return InferenceResponse(
            request_id=request.request_id,
            raw_actions=np.array([[1.0], [2.0]], dtype=np.float32),
            processed_actions=np.array([[11.0], [12.0]], dtype=np.float32),
            server_latency_s=0.01,
            model_latency_s=0.01,
            action_shape=(2, 1),
        )


def test_http_policy_server_roundtrip() -> None:
    service = FakeService()
    server = make_server(host="127.0.0.1", port=0, service=service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        client = RemotePolicyClient(f"http://{host}:{port}", timeout_s=5)
        health = client.health()
        assert health["ok"] is True

        response = client.infer(
            InferenceRequest(
                request_id=42,
                observation_frame={"observation.state": np.array([0.0], dtype=np.float32)},
                task="task",
                robot_type="so101_follower",
                obs_timestamp_s=1.0,
                obs_sequence_id=1,
                enable_rtc=True,
                predicted_delay_steps=0,
                prev_chunk_left_over=None,
                execution_horizon=10,
            )
        )

        assert response.request_id == 42
        assert np.allclose(response.processed_actions, [[11.0], [12.0]])
        assert service.inference_count == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
