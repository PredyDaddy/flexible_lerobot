from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

from .policy_service import PolicyService
from .protocol import CONTENT_TYPE, InferenceRequest, InferenceResponse, dumps_payload, loads_payload


class PolicyHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], service: PolicyService) -> None:
        super().__init__(server_address, PolicyRequestHandler)
        self.service = service


class PolicyRequestHandler(BaseHTTPRequestHandler):
    server: PolicyHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        service = self.server.service
        self._send_json(
            {
                "ok": True,
                "inference_count": service.inference_count,
                "last_latency_s": service.last_latency_s,
                "enable_rtc": service.config.enable_rtc,
                "device": str(service.device),
            }
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/infer":
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = loads_payload(self.rfile.read(length))
            if not isinstance(request, InferenceRequest):
                raise ValueError(f"Expected InferenceRequest, got {type(request)}")
            response = self.server.service.infer(request)
            self._send_pickle(response)
        except Exception as exc:
            response = InferenceResponse(
                request_id=-1,
                raw_actions=np.zeros((0, 0), dtype="float32"),
                processed_actions=np.zeros((0, 0), dtype="float32"),
                server_latency_s=0.0,
                model_latency_s=0.0,
                action_shape=(0, 0),
                error=str(exc),
            )
            self._send_pickle(response, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_pickle(self, payload: Any, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = dumps_payload(payload)
        self.send_response(int(status))
        self.send_header("Content-Type", CONTENT_TYPE)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def make_server(*, host: str, port: int, service: PolicyService) -> PolicyHTTPServer:
    return PolicyHTTPServer((host, port), service)
