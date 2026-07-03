from __future__ import annotations

import argparse
import json
import os

import numpy as np

from openpi_so101 import dataset_v3
from openpi_so101 import patches
from openpi_so101 import runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare v3 adapter samples with converted LeRobot v2.1 pilot samples.")
    parser.add_argument("--num-samples", type=int, default=int(os.environ.get("OPENPI_SO101_ALIGN_SAMPLES", "9")))
    parser.add_argument("--samples-per-episode", type=int, default=3)
    parser.add_argument("--action-horizon", type=int, default=int(os.environ.get("OPENPI_SO101_ACTION_HORIZON", "50")))
    parser.add_argument("--image-mean-threshold", type=float, default=8.0)
    parser.add_argument("--image-max-threshold", type=float, default=80.0)
    return parser


def _to_hwc_uint8(image) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255)
    return arr.astype(np.uint8)


def _sample_indices(v21_meta, *, max_samples: int, samples_per_episode: int) -> list[int]:
    selected: list[int] = []
    start = 0
    for episode in v21_meta.episodes.values():
        length = int(episode["length"])
        candidates = [0, length // 2, length - 1]
        for local_index in candidates[:samples_per_episode]:
            idx = start + local_index
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= max_samples:
                return selected
        start += length
    return selected


def main() -> None:
    runtime.bootstrap()
    args = build_parser().parse_args()
    repo_id = os.environ["OPENPI_SO101_REPO_ID"]
    v3 = dataset_v3.SO101LeRobotV3Dataset(
        repo_id,
        runtime.dataset_root(),
        action_horizon=args.action_horizon,
        max_frames=None,
        decode_images=True,
    )
    v21 = patches.LeRobotDatasetV21Compat(
        repo_id,
        runtime.converted_dataset_root(),
        action_horizon=args.action_horizon,
        action_sequence_keys=("action",),
        max_frames=None,
    )

    indices = [idx for idx in _sample_indices(v21.meta, max_samples=args.num_samples, samples_per_episode=args.samples_per_episode) if idx < len(v3) and idx < len(v21)]
    summary = {
        "samples": len(indices),
        "indices": indices,
        "image": {},
        "max_action_abs_diff": 0.0,
        "max_state_abs_diff": 0.0,
    }
    for idx in indices:
        a = v3[idx]
        b = v21[idx]
        task = str(b["task"])
        if str(a["prompt"]) != task:
            raise AssertionError(f"prompt mismatch at {idx}: {a['prompt']!r} != {task!r}")

        state_diff = float(np.max(np.abs(np.asarray(a["observation.state"]) - np.asarray(b["observation.state"]))))
        action_diff = float(np.max(np.abs(np.asarray(a["action"]) - np.asarray(b["action"]))))
        summary["max_state_abs_diff"] = max(summary["max_state_abs_diff"], state_diff)
        summary["max_action_abs_diff"] = max(summary["max_action_abs_diff"], action_diff)
        if state_diff > 1e-6:
            raise AssertionError(f"state mismatch at {idx}: {state_diff}")
        if action_diff > 1e-6:
            raise AssertionError(f"action chunk mismatch at {idx}: {action_diff}")

        for key in ("observation.images.top", "observation.images.wrist"):
            img_a = _to_hwc_uint8(a[key]).astype(np.int16)
            img_b = _to_hwc_uint8(b[key]).astype(np.int16)
            if img_a.shape != img_b.shape:
                raise AssertionError(f"{key} shape mismatch at {idx}: {img_a.shape} != {img_b.shape}")
            diff = np.abs(img_a - img_b)
            item = summary["image"].setdefault(key, {"max_mean_abs_diff": 0.0, "max_pixel_abs_diff": 0.0})
            item["max_mean_abs_diff"] = max(item["max_mean_abs_diff"], float(diff.mean()))
            item["max_pixel_abs_diff"] = max(item["max_pixel_abs_diff"], float(diff.max()))

    for key, item in summary["image"].items():
        if item["max_mean_abs_diff"] > args.image_mean_threshold:
            raise AssertionError(f"{key} mean image diff too high: {item}")
        if item["max_pixel_abs_diff"] > args.image_max_threshold:
            raise AssertionError(f"{key} max image diff too high: {item}")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
