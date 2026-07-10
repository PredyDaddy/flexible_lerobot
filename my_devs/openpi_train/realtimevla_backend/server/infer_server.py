from __future__ import annotations

import argparse
import pickle
import threading
import time
from typing import Callable

from fastapi import FastAPI
from fastapi import Request
from fastapi import Response
import numpy as np
import uvicorn

from builders import build_model
from builders import build_optimizer
from config import Config
from config import load_config


INFER_LOCK = threading.Lock()


def create_app(handler: Callable[[dict], dict], endpoint: str = "/infer") -> FastAPI:
    app = FastAPI(title="RealtimeVLA SO101 Backend")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post(endpoint)
    async def infer(request: Request) -> Response:
        body = await request.body()
        if not body:
            return Response(content=b"empty request body", status_code=400)
        try:
            data = pickle.loads(body)
        except Exception as exc:
            return Response(content=f"invalid pickle payload: {exc}".encode(), status_code=400)
        if not isinstance(data, dict):
            return Response(content=b"request payload must be a dict", status_code=400)

        start_time = time.perf_counter()
        try:
            with INFER_LOCK:
                output = handler(data)
        except Exception as exc:
            return Response(content=f"inference failed: {type(exc).__name__}: {exc}".encode(), status_code=500)

        infer_time = time.perf_counter() - start_time
        result = dict(output)
        result["infer_time"] = infer_time
        return Response(content=pickle.dumps(result), media_type="application/octet-stream")

    return app


class InferPipeline:
    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._model = build_model(cfg)
        self._optimizer = build_optimizer(cfg)

    def __call__(self, request: dict) -> dict:
        raw_actions = self._model.infer_actions(request)
        action_list = self._optimizer.optimize(raw_actions)
        result = {"action_list": action_list}
        if self._cfg.inference.include_raw_actions:
            result["raw_action_list"] = raw_actions
        return result

    def warmup(self) -> None:
        height = int(self._cfg.inference.warmup_image_height)
        width = int(self._cfg.inference.warmup_image_width)
        state_dim = int(self._cfg.model.state_dim)
        payload = {
            "images": {
                "top": np.zeros((height, width, 3), dtype=np.uint8),
                "wrist": np.zeros((height, width, 3), dtype=np.uint8),
            },
            "state": np.zeros((state_dim,), dtype=np.float32),
            "prompt": self._cfg.model.default_prompt,
            "timestamp": time.time(),
        }
        start = time.perf_counter()
        result = self(payload)
        elapsed = time.perf_counter() - start
        action_count = len(result.get("action_list", []))
        print(f"[infer_server] warmup complete in {elapsed:.2f}s action_count={action_count}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve OpenPI SO101 through a RealtimeVLA-style HTTP backend.")
    parser.add_argument("--config", required=True, help="Path to server YAML config.")
    parser.add_argument("--host", default=None, help="Override config server.host.")
    parser.add_argument("--port", type=int, default=None, help="Override config server.port.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    host = args.host or cfg.server.host
    port = args.port or cfg.server.port
    pipeline = InferPipeline(cfg)
    if cfg.inference.warmup_on_start:
        pipeline.warmup()
    app = create_app(pipeline, endpoint=cfg.server.endpoint)
    print(f"[infer_server] listening on {host}:{port}{cfg.server.endpoint}")
    uvicorn.run(app, host=host, port=port, access_log=bool(cfg.server.access_log))


if __name__ == "__main__":
    main()
