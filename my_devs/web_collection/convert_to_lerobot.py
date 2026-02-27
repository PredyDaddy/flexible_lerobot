from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
import yaml

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _normalize_instruction(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip()


def _pick_unique_root(parent: Path, name: str) -> Path:
    """Pick a non-existing dataset root path under parent.

    LeRobotDataset.create uses mkdir(exist_ok=False), so we must ensure uniqueness.
    """
    parent.mkdir(parents=True, exist_ok=True)
    base = parent / name
    if not base.exists():
        return base
    for i in range(1, 1000):
        cand = parent / f"{name}_{i:03d}"
        if not cand.exists():
            return cand
    raise RuntimeError(f"Could not find a free output directory for {name} under {parent}")


def _sanitize_repo_id(repo_id: str) -> str:
    # Local folder name: keep slashes (HF-style) but remove weird characters.
    repo_id = repo_id.strip()
    if not repo_id:
        raise ValueError("repo_id is empty")
    # Hyphen must be first/last in the character class to avoid being interpreted as a range.
    repo_id = re.sub(r"[^a-zA-Z0-9_./\\-]+", "_", repo_id)
    repo_id = repo_id.strip("/ ")
    return repo_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert a web_collection HDF5 episode to LeRobot v3 dataset format.")
    p.add_argument("--input_hdf5", type=str, required=True, help="Path to episode .hdf5")
    p.add_argument("--output_dir", type=str, default="/tmp/web_collection_lerobot", help="Parent output directory.")
    p.add_argument("--repo_id", type=str, required=True, help="LeRobot dataset repo_id (used in metadata).")
    p.add_argument("--robot_type", type=str, default=None, help="Robot type metadata (optional).")
    p.add_argument("--fps", type=int, default=None, help="Override fps (otherwise read from HDF5 attrs).")
    p.add_argument("--use_videos", action="store_true", help="Store visual keys as videos (mp4).")
    p.add_argument("--swap_rb", action="store_true", help="Swap red/blue channels for all RGB images.")
    p.add_argument("--instruction", type=str, default=None, help="Override instruction/task text.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_hdf5).expanduser().resolve()
    if not in_path.is_file():
        raise FileNotFoundError(in_path)

    repo_id = _sanitize_repo_id(args.repo_id)
    out_parent = Path(args.output_dir).expanduser().resolve()
    out_root = _pick_unique_root(out_parent, repo_id.replace("/", "__"))

    with h5py.File(in_path, "r") as f:
        if f.attrs.get("schema", "") != "web_collection_hdf5_v1":
            # Be permissive: allow converting older HDF5 if keys match.
            pass

        fps = int(args.fps if args.fps is not None else f.attrs.get("fps", 30))
        instruction = _normalize_instruction(args.instruction)
        if not instruction:
            instruction = _normalize_instruction(str(f.attrs.get("instruction", "") or ""))
        if not instruction:
            raise ValueError(
                "instruction 不能为空。请通过 --instruction 指定，或确保 HDF5 attrs['instruction'] 为非空字符串。"
            )

        obs = f["observations"]
        g_img = obs["images"]
        cam_names = sorted(list(g_img.keys()))
        if not cam_names:
            raise ValueError("No observations/images found in HDF5.")

        qpos = obs["qpos"]
        action = f["action"]
        base_action = f.get("base_action", None)

        n = int(qpos.shape[0])
        if int(action.shape[0]) != n:
            raise ValueError(f"Length mismatch: qpos={qpos.shape[0]} action={action.shape[0]}")
        for cam in cam_names:
            if int(g_img[cam].shape[0]) != n:
                raise ValueError(f"Length mismatch: images[{cam}]={g_img[cam].shape[0]} expected={n}")

        state_dim = int(qpos.shape[1])
        action_dim = int(action.shape[1])
        h, w, c = map(int, g_img[cam_names[0]].shape[1:])
        if c not in (1, 3, 4):
            raise ValueError(f"Unexpected image channels: {c}")

        # Features.
        img_dtype = "video" if args.use_videos else "image"
        features: dict = {
            "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": None},
            "action": {"dtype": "float32", "shape": (action_dim,), "names": None},
            "language_instruction": {"dtype": "string", "shape": (1,), "names": None},
        }
        for cam in cam_names:
            hh, ww, cc = map(int, g_img[cam].shape[1:])
            if cc == 4:
                cc = 3
            features[f"observation.images.{cam}"] = {
                "dtype": img_dtype,
                "shape": (hh, ww, cc),
                "names": ["height", "width", "channels"],
            }

        if base_action is not None:
            features["observation.environment_state"] = {
                "dtype": "float32",
                "shape": (2,),
                "names": {"axes": ["base_lin_x", "base_ang_z"]},
            }

        # Create dataset.
        ds = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=str(out_root),
            robot_type=args.robot_type,
            use_videos=args.use_videos,
        )

        # One episode per input HDF5.
        for i in range(n):
            frame = {
                "task": instruction,
                "language_instruction": instruction,
                "observation.state": np.asarray(qpos[i], dtype=np.float32),
                "action": np.asarray(action[i], dtype=np.float32),
            }
            if base_action is not None:
                frame["observation.environment_state"] = np.asarray(base_action[i], dtype=np.float32)

            for cam in cam_names:
                img = np.asarray(g_img[cam][i])
                if img.ndim == 2:
                    img = np.repeat(img[:, :, None], 3, axis=2)
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                if args.swap_rb and img.shape[2] == 3:
                    img = img[:, :, [2, 1, 0]]
                frame[f"observation.images.{cam}"] = img

            ds.add_frame(frame)

        ds.save_episode()
        ds.finalize()

    print(f"[ok] wrote LeRobot dataset to: {out_root}")


if __name__ == "__main__":
    main()
