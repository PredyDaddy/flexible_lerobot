from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .protocol import CONTENT_TYPE, InferenceRequest, InferenceResponse, dumps_payload, loads_payload


@dataclass(slots=True)
class RemotePolicyClient:
    server_url: str
    timeout_s: float = 120.0

    def __post_init__(self) -> None:
        self.server_url = self.server_url.rstrip("/")

    def health(self) -> dict[str, Any]:
        with urllib.request.urlopen(f"{self.server_url}/health", timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        body = dumps_payload(request)
        http_request = urllib.request.Request(
            f"{self.server_url}/infer",
            data=body,
            method="POST",
            headers={"Content-Type": CONTENT_TYPE},
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_s) as response:
                payload = loads_payload(response.read())
        except urllib.error.HTTPError as exc:
            payload = loads_payload(exc.read())
        if not isinstance(payload, InferenceResponse):
            raise ValueError(f"Expected InferenceResponse, got {type(payload)}")
        if payload.error:
            raise RuntimeError(payload.error)
        return payload
