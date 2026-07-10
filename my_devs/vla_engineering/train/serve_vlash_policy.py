#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path("/data/cqy_workspace/flexible_lerobot")
VLASH_ROOT = REPO_ROOT / "my_devs/vla_engineering/vlash-main"
for path in (VLASH_ROOT, REPO_ROOT / "src"):
    path_str = path.as_posix()
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from lerobot.configs.policies import PreTrainedConfig

from vlash.policies.factory import get_policy_class


DEFAULT_POLICY_PATH = (
    REPO_ROOT
    / "my_devs/vla_engineering/train/outputs/pi05_so101_vlash_lora_10epochs_r192_bs1_accum8/"
    "checkpoints/033275/pretrained_model"
)
DEFAULT_PROMPT = "Put the eraser into the small box"


def ensure_offline_defaults() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def decode_array(payload: dict[str, Any]) -> np.ndarray:
    shape = tuple(int(v) for v in payload["shape"])
    dtype = np.dtype(payload["dtype"])
    raw = base64.b64decode(payload["data_b64"])
    array = np.frombuffer(raw, dtype=dtype).reshape(shape)
    return np.ascontiguousarray(array)


def image_hwc_uint8_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape={image.shape}")
    if image.dtype != np.uint8:
        image = image.astype(np.uint8, copy=False)
    tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float().div_(255.0)
    return tensor.unsqueeze(0).to(device)


def state_to_tensor(state: np.ndarray, device: torch.device) -> torch.Tensor:
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (6,):
        raise ValueError(f"Expected SO101 state shape=(6,), got shape={state.shape}")
    return torch.from_numpy(state).unsqueeze(0).to(device)


class VLASHPolicyService:
    def __init__(
        self,
        policy_path: Path,
        *,
        device: str,
        default_prompt: str,
        num_inference_steps: int,
        compile_model: bool,
        fuse_qkv: bool,
        fuse_gate_up: bool,
    ) -> None:
        ensure_offline_defaults()
        self.policy_path = policy_path
        self.default_prompt = default_prompt
        self.device = torch.device(device)
        if not policy_path.is_dir():
            raise FileNotFoundError(f"Policy path does not exist: {policy_path}")

        cli_overrides = [
            f"--device={device}",
            f"--num_inference_steps={num_inference_steps}",
            f"--compile_model={str(compile_model).lower()}",
            f"--fuse_qkv={str(fuse_qkv).lower()}",
            f"--fuse_gate_up={str(fuse_gate_up).lower()}",
        ]
        cfg = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
        policy_cls = get_policy_class(cfg.type)
        self.policy = policy_cls.from_pretrained(policy_path, config=cfg, dataset_stats=None)
        self.policy.eval()
        self.cfg = cfg
        self.request_count = 0

    def metadata(self) -> dict[str, Any]:
        return {
            "policy_path": str(self.policy_path),
            "policy_type": self.cfg.type,
            "device": str(self.device),
            "num_inference_steps": self.cfg.num_inference_steps,
            "n_action_steps": self.cfg.n_action_steps,
            "compile_model": self.cfg.compile_model,
            "fuse_qkv": self.cfg.fuse_qkv,
            "fuse_gate_up": self.cfg.fuse_gate_up,
            "default_prompt": self.default_prompt,
            "request_count": self.request_count,
        }

    @torch.inference_mode()
    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        prompt = request.get("prompt") or self.default_prompt
        top = decode_array(request["observation.images.top"])
        wrist = decode_array(request["observation.images.wrist"])
        state = decode_array(request["observation.state"])

        decode_s = time.perf_counter() - start
        batch = {
            "observation.images.top": image_hwc_uint8_to_tensor(top, self.device),
            "observation.images.wrist": image_hwc_uint8_to_tensor(wrist, self.device),
            "observation.state": state_to_tensor(state, self.device),
            "task": [prompt],
        }

        infer_start = time.perf_counter()
        actions = self.policy.predict_action_chunk(batch)
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize()
        infer_s = time.perf_counter() - infer_start

        actions_np = actions[0].detach().float().cpu().numpy().astype(np.float32, copy=False)
        self.request_count += 1
        return {
            "actions": actions_np.tolist(),
            "shape": list(actions_np.shape),
            "dtype": "float32",
            "server_timing": {
                "decode_ms": decode_s * 1000,
                "infer_ms": infer_s * 1000,
                "total_ms": (time.perf_counter() - start) * 1000,
            },
        }


def make_handler(service: VLASHPolicyService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "VLASHPolicyHTTP/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[HTTP] {self.address_string()} - {fmt % args}", flush=True)

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in {"/", "/health", "/metadata"}:
                self._send_json(200, {"ok": True, "metadata": service.metadata()})
                return
            self._send_json(404, {"ok": False, "error": f"Unknown path: {self.path}"})

        def do_POST(self) -> None:
            if self.path != "/infer":
                self._send_json(404, {"ok": False, "error": f"Unknown path: {self.path}"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                response = service.infer(request)
                self._send_json(200, {"ok": True, **response})
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a VLASH PI0.5 policy over local HTTP.")
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--default-prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--fuse-qkv", action="store_true")
    parser.add_argument("--fuse-gate-up", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    service = VLASHPolicyService(
        args.policy_path,
        device=args.device,
        default_prompt=args.default_prompt,
        num_inference_steps=args.num_inference_steps,
        compile_model=args.compile_model,
        fuse_qkv=args.fuse_qkv,
        fuse_gate_up=args.fuse_gate_up,
    )
    print("[INFO] VLASH policy server ready")
    print(json.dumps(service.metadata(), indent=2, ensure_ascii=False))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    print(f"[INFO] Serving HTTP on {args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received; shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
