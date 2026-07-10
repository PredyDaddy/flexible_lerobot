from __future__ import annotations

import pickle

import requests


class InferClient:
    def __init__(self, base_url: str, endpoint: str, timeout_s: float):
        self._session = requests.Session()
        self._url = f"{base_url.rstrip('/')}{endpoint}"
        self._timeout_s = float(timeout_s)

    @property
    def url(self) -> str:
        return self._url

    def infer(self, payload: dict) -> dict:
        try:
            response = self._session.post(
                self._url,
                data=pickle.dumps(payload),
                headers={"Content-Type": "application/octet-stream"},
                timeout=self._timeout_s,
            )
        except requests.exceptions.ReadTimeout as exc:
            raise TimeoutError(
                f"HTTP inference timed out after {self._timeout_s:.1f}s. "
                "The first real OpenPI request can take 10+ seconds while JAX compiles; "
                "restart the server with the latest warmup-enabled script or pass --timeout-s 60."
            ) from exc
        response.raise_for_status()
        result = pickle.loads(response.content)
        if isinstance(result, dict):
            return result
        return {"action_list": result}
