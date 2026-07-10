#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path("/data/cqy_workspace/flexible_lerobot")
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_TASK = "Put the eraser into the small box"


def ensure_offline_defaults() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def encode_array(value: np.ndarray) -> dict[str, Any]:
    value = np.ascontiguousarray(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "data_b64": base64.b64encode(value.tobytes()).decode("ascii"),
    }


def image_to_hwc_uint8(value: Any) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape={image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(image)) <= 1.5 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8, copy=False)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape={image.shape}")
    return np.ascontiguousarray(image)


def state_to_float32(value: Any) -> np.ndarray:
    state = np.asarray(value, dtype=np.float32)
    if state.shape != (6,):
        raise ValueError(f"Expected SO101 state shape=(6,), got shape={state.shape}")
    return np.ascontiguousarray(state)


def get_json(url: str, timeout_s: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test a running train_all VLASH HTTP policy server.")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8005")))
    parser.add_argument("--task", default=os.getenv("TASK", DEFAULT_TASK))
    parser.add_argument("--repo-id", default="desk_cleanup_v1/eraser_cup_multi_task")
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "datasets/desk_cleanup_v1/eraser_cup_multi_task"))
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()

    ensure_offline_defaults()

    base_url = f"http://{args.host}:{args.port}"
    metadata = get_json(f"{base_url}/metadata", timeout_s=10.0)

    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        video_backend=args.video_backend,
    )
    sample = dataset[args.sample_index]
    payload = {
        "observation.images.top": encode_array(image_to_hwc_uint8(sample["observation.images.top"])),
        "observation.images.wrist": encode_array(image_to_hwc_uint8(sample["observation.images.wrist"])),
        "observation.state": encode_array(state_to_float32(sample["observation.state"])),
        "prompt": args.task,
    }
    result = post_json(f"{base_url}/infer", payload, timeout_s=args.timeout_s)
    if not result.get("ok", False):
        raise RuntimeError(f"Server returned a failed inference response: {result}")

    actions = np.asarray(result["actions"], dtype=np.float32)
    finite = bool(np.isfinite(actions).all())
    if not finite:
        raise RuntimeError("Server returned NaN or Inf action values.")
    if actions.ndim != 2 or actions.shape[1] != 6:
        raise RuntimeError(f"Expected action chunk shape=(horizon, 6), got {actions.shape}")

    summary = {
        "server": base_url,
        "metadata": metadata,
        "task": args.task,
        "sample_index": args.sample_index,
        "action_shape": list(actions.shape),
        "finite": finite,
        "first_action": actions[0].astype(float).tolist(),
        "server_timing": result.get("server_timing", {}),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
