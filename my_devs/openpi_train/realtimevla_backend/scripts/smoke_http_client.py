from __future__ import annotations

import argparse
import pickle

import cv2
import numpy as np
import requests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send one synthetic SO101 request to a RealtimeVLA backend.")
    parser.add_argument("--url", default="http://127.0.0.1:18080/infer")
    parser.add_argument("--timeout-s", type=float, default=5.0)
    return parser


def _jpeg(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("failed to encode synthetic image")
    return buf.tobytes()


def main() -> int:
    args = build_parser().parse_args()
    top = np.zeros((480, 640, 3), dtype=np.uint8)
    wrist = np.zeros((480, 640, 3), dtype=np.uint8)
    top[..., 0] = 32
    wrist[..., 1] = 32
    payload = {
        "images": {
            "top": _jpeg(top),
            "wrist": _jpeg(wrist),
        },
        "state": np.zeros((6,), dtype=np.float32),
        "prompt": "Put the eraser into the small box",
    }
    response = requests.post(
        args.url,
        data=pickle.dumps(payload),
        headers={"Content-Type": "application/octet-stream"},
        timeout=args.timeout_s,
    )
    response.raise_for_status()
    result = pickle.loads(response.content)
    actions = np.asarray(result["action_list"], dtype=np.float32)
    print(f"status=ok action_shape={actions.shape} infer_time={result.get('infer_time', 0.0):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

