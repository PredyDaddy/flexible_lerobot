from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path("/data/cqy_workspace/flexible_lerobot/my_devs/openpi_train")
OPENPI_ROOT = ROOT / "openpi-main"
for path in (ROOT, OPENPI_ROOT, OPENPI_ROOT / "src", OPENPI_ROOT / "packages/openpi-client/src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from openpi_so101 import dataset_v3  # noqa: E402
from openpi_so101 import patches  # noqa: E402


DEFAULT_SOURCE_ROOT = Path("/data/cqy_workspace/flexible_lerobot/datasets/desk_cleanup_v1/eraser_cup_multi_task")
DEFAULT_V21_ROOT = ROOT / "easy_use/data/lerobot_v21/desk_cleanup_v1/eraser_cup_multi_task"
DEFAULT_REPO_ID = "desk_cleanup_v1/eraser_cup_multi_task"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a converted LeRobot v2.1 SO101 dataset.")
    parser.add_argument("--root", type=Path, default=DEFAULT_V21_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--expected-episodes", type=int, default=157)
    parser.add_argument("--expected-frames", type=int, default=53235)
    parser.add_argument("--expected-tasks", type=int, default=3)
    parser.add_argument("--expected-fps", type=int, default=30)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--alignment-samples", type=int, default=24)
    parser.add_argument("--image-mean-threshold", type=float, default=8.0)
    parser.add_argument("--image-max-threshold", type=float, default=96.0)
    parser.add_argument("--skip-v3-alignment", action="store_true")
    return parser


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise AssertionError(f"Missing required file: {path}")
    return json.loads(path.read_text())


def _to_hwc_uint8(image) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255)
    return arr.astype(np.uint8)


def _sample_indices(total_frames: int, total_episodes: int, max_samples: int) -> list[int]:
    if total_frames <= 0:
        return []
    candidates = {0, total_frames - 1, total_frames // 2}
    if total_episodes > 1:
        stride = max(total_frames // max_samples, 1)
        candidates.update(range(0, total_frames, stride))
    return sorted(idx for idx in candidates if 0 <= idx < total_frames)[:max_samples]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_metadata(args) -> dict:
    root = args.root.resolve()
    info = _load_json(root / "meta/info.json")
    _assert(info.get("codebase_version") == "v2.1", f"codebase_version must be v2.1, got {info.get('codebase_version')}")
    _assert(int(info.get("fps")) == args.expected_fps, f"fps mismatch: {info.get('fps')} != {args.expected_fps}")
    _assert(int(info.get("total_episodes")) == args.expected_episodes, "total_episodes mismatch")
    _assert(int(info.get("total_frames")) == args.expected_frames, "total_frames mismatch")
    _assert(int(info.get("total_tasks")) == args.expected_tasks, "total_tasks mismatch")
    _assert(int(info.get("total_videos")) == args.expected_episodes * 2, "total_videos mismatch")

    features = info.get("features", {})
    required = {
        "action",
        "observation.state",
        "observation.images.top",
        "observation.images.wrist",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    missing = sorted(required - set(features))
    _assert(not missing, f"Missing features: {missing}")
    _assert(features["action"]["shape"] == [6], "action shape must be [6]")
    _assert(features["observation.state"]["shape"] == [6], "observation.state shape must be [6]")
    for key in ("observation.images.top", "observation.images.wrist"):
        _assert(features[key]["dtype"] == "video", f"{key} must be video")
        _assert(features[key]["shape"] == [3, 480, 640], f"{key} shape must be [3, 480, 640]")

    meta_files = ["tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"]
    for filename in meta_files:
        _assert((root / "meta" / filename).exists(), f"Missing meta/{filename}")

    parquet_files = sorted((root / "data").glob("chunk-*/*.parquet"))
    top_videos = sorted((root / "videos").glob("chunk-*/observation.images.top/*.mp4"))
    wrist_videos = sorted((root / "videos").glob("chunk-*/observation.images.wrist/*.mp4"))
    _assert(len(parquet_files) == args.expected_episodes, f"parquet episode count mismatch: {len(parquet_files)}")
    _assert(len(top_videos) == args.expected_episodes, f"top video count mismatch: {len(top_videos)}")
    _assert(len(wrist_videos) == args.expected_episodes, f"wrist video count mismatch: {len(wrist_videos)}")
    return info


def validate_official_loader(args) -> dict:
    from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    root = args.root.resolve()
    meta = LeRobotDatasetMetadata(args.repo_id, root, CODEBASE_VERSION)
    _assert(meta.fps == args.expected_fps, "official metadata fps mismatch")
    _assert(meta.info["total_episodes"] == args.expected_episodes, "official metadata total_episodes mismatch")
    _assert(meta.info["total_frames"] == args.expected_frames, "official metadata total_frames mismatch")
    _assert(len(meta.tasks) == args.expected_tasks, "official metadata task count mismatch")

    dataset = patches.LeRobotDatasetV21Compat(
        args.repo_id,
        root,
        action_horizon=args.action_horizon,
        action_sequence_keys=("action",),
        max_frames=None,
    )
    _assert(len(dataset) == args.expected_frames, f"dataset length mismatch: {len(dataset)}")
    indices = _sample_indices(args.expected_frames, args.expected_episodes, args.alignment_samples)
    for idx in indices[: min(6, len(indices))]:
        item = dataset[idx]
        _assert(item["observation.state"].shape[-1] == 6, f"state shape mismatch at {idx}")
        _assert(item["action"].shape == (args.action_horizon, 6), f"action chunk shape mismatch at {idx}")
        _assert(str(item["task"]), f"empty task at {idx}")
        for key in ("observation.images.top", "observation.images.wrist"):
            image = _to_hwc_uint8(item[key])
            _assert(image.shape == (480, 640, 3), f"{key} image shape mismatch at {idx}: {image.shape}")
    return {"sampled_indices": indices}


def validate_v3_alignment(args) -> dict:
    v3 = dataset_v3.SO101LeRobotV3Dataset(
        args.repo_id,
        args.source_root.resolve(),
        action_horizon=args.action_horizon,
        max_frames=None,
        decode_images=True,
    )
    v21 = patches.LeRobotDatasetV21Compat(
        args.repo_id,
        args.root.resolve(),
        action_horizon=args.action_horizon,
        action_sequence_keys=("action",),
        max_frames=None,
    )
    indices = _sample_indices(len(v21), args.expected_episodes, args.alignment_samples)
    summary = {
        "samples": len(indices),
        "indices": indices,
        "max_state_abs_diff": 0.0,
        "max_action_abs_diff": 0.0,
        "image": {
            "observation.images.top": {"max_mean_abs_diff": 0.0, "max_pixel_abs_diff": 0.0},
            "observation.images.wrist": {"max_mean_abs_diff": 0.0, "max_pixel_abs_diff": 0.0},
        },
    }
    for idx in indices:
        a = v3[idx]
        b = v21[idx]
        _assert(str(a["prompt"]) == str(b["task"]), f"prompt mismatch at {idx}")
        state_diff = float(np.max(np.abs(np.asarray(a["observation.state"]) - np.asarray(b["observation.state"]))))
        action_diff = float(np.max(np.abs(np.asarray(a["action"]) - np.asarray(b["action"]))))
        summary["max_state_abs_diff"] = max(summary["max_state_abs_diff"], state_diff)
        summary["max_action_abs_diff"] = max(summary["max_action_abs_diff"], action_diff)
        _assert(state_diff <= 1e-6, f"state mismatch at {idx}: {state_diff}")
        _assert(action_diff <= 1e-6, f"action mismatch at {idx}: {action_diff}")
        for key in ("observation.images.top", "observation.images.wrist"):
            img_a = _to_hwc_uint8(a[key]).astype(np.int16)
            img_b = _to_hwc_uint8(b[key]).astype(np.int16)
            _assert(img_a.shape == img_b.shape, f"{key} shape mismatch at {idx}: {img_a.shape} != {img_b.shape}")
            diff = np.abs(img_a - img_b)
            item = summary["image"][key]
            item["max_mean_abs_diff"] = max(item["max_mean_abs_diff"], float(diff.mean()))
            item["max_pixel_abs_diff"] = max(item["max_pixel_abs_diff"], float(diff.max()))

    for key, item in summary["image"].items():
        _assert(item["max_mean_abs_diff"] <= args.image_mean_threshold, f"{key} mean image diff too high: {item}")
        _assert(item["max_pixel_abs_diff"] <= args.image_max_threshold, f"{key} max image diff too high: {item}")
    return summary


def main() -> None:
    args = build_parser().parse_args()
    args.root = args.root.expanduser().resolve()
    args.source_root = args.source_root.expanduser().resolve()
    os.environ.setdefault("OPENPI_SO101_V21_ROOT", str(args.root))
    result = {
        "root": str(args.root),
        "metadata": validate_metadata(args),
        "official_loader": validate_official_loader(args),
    }
    if not args.skip_v3_alignment:
        result["v3_alignment"] = validate_v3_alignment(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
