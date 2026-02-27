from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import h5py
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _normalize_instruction(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip()


def _pick_unique_root(parent: Path, name: str) -> Path:
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
    repo_id = repo_id.strip()
    if not repo_id:
        raise ValueError("repo_id is empty")
    repo_id = re.sub(r"[^a-zA-Z0-9_./\\-]+", "_", repo_id)
    repo_id = repo_id.strip("/ ")
    return repo_id


def _list_hdf5_episodes(dataset_dir: Path) -> list[Path]:
    eps = sorted(dataset_dir.glob("episode_*.hdf5"))
    if not eps:
        raise FileNotFoundError(f"No episode_*.hdf5 found in: {dataset_dir}")
    return eps


def _infer_features_from_episode(
    episode_path: Path, *, use_videos: bool, include_base: bool, cam_names: list[str] | None = None
) -> tuple[dict, list[str]]:
    img_dtype = "video" if use_videos else "image"
    with h5py.File(episode_path, "r") as f:
        obs = f["observations"]
        g_img = obs["images"]
        cams = sorted(list(g_img.keys()))
        if cam_names is not None and cams != cam_names:
            raise ValueError(f"Camera keys mismatch. expected={cam_names} got={cams} in {episode_path}")

        qpos = obs["qpos"]
        action = f["action"]
        state_dim = int(qpos.shape[1])
        action_dim = int(action.shape[1])

        features: dict = {
            "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": None},
            "action": {"dtype": "float32", "shape": (action_dim,), "names": None},
            "language_instruction": {"dtype": "string", "shape": (1,), "names": None},
        }

        for cam in cams:
            h, w, c = map(int, g_img[cam].shape[1:])
            if c == 4:
                c = 3
            features[f"observation.images.{cam}"] = {
                "dtype": img_dtype,
                "shape": (h, w, c),
                "names": ["height", "width", "channels"],
            }

        if include_base:
            features["observation.environment_state"] = {
                "dtype": "float32",
                "shape": (2,),
                "names": {"axes": ["base_lin_x", "base_ang_z"]},
            }

        return features, cams


def _convert_one_episode(
    ds: LeRobotDataset,
    episode_path: Path,
    *,
    swap_rb: bool,
    instruction_override: str | None,
    expected_cam_names: list[str],
    include_base: bool,
) -> None:
    with h5py.File(episode_path, "r") as f:
        obs = f["observations"]
        g_img = obs["images"]
        cam_names = sorted(list(g_img.keys()))
        if cam_names != expected_cam_names:
            raise ValueError(f"Camera keys mismatch. expected={expected_cam_names} got={cam_names} in {episode_path}")

        qpos = obs["qpos"]
        action = f["action"]
        base_action = f.get("base_action", None)

        if include_base and base_action is None:
            raise ValueError(f"Expected base_action but not found in {episode_path}")

        n = int(qpos.shape[0])
        if int(action.shape[0]) != n:
            raise ValueError(f"Length mismatch: qpos={qpos.shape[0]} action={action.shape[0]} in {episode_path}")

        # Per-episode instruction.
        instruction = _normalize_instruction(instruction_override)
        if not instruction:
            instruction = _normalize_instruction(str(f.attrs.get("instruction", "") or ""))
        if not instruction:
            raise ValueError(
                f"Episode {episode_path.name} 缺少非空 instruction。"
                "请传 --instruction 覆盖，或先在 HDF5 attrs['instruction'] 中补齐。"
            )

        t0: float | None = None
        for i in range(n):
            if i == 0:
                t0 = time.time()
                print(f"[frames] {episode_path.name}: {n} 帧，开始转换...", flush=True)

            frame = {
                "task": instruction,
                "language_instruction": instruction,
                "observation.state": np.asarray(qpos[i], dtype=np.float32),
                "action": np.asarray(action[i], dtype=np.float32),
            }
            if include_base and base_action is not None:
                frame["observation.environment_state"] = np.asarray(base_action[i], dtype=np.float32)

            for cam in cam_names:
                img = np.asarray(g_img[cam][i])
                if img.ndim == 2:
                    img = np.repeat(img[:, :, None], 3, axis=2)
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                if swap_rb and img.shape[2] == 3:
                    img = img[:, :, [2, 1, 0]]
                frame[f"observation.images.{cam}"] = img

            ds.add_frame(frame)

            if t0 is not None and (i + 1) % 200 == 0:
                dt = time.time() - t0
                print(f"[frames] {episode_path.name}: {i + 1}/{n} ({(i + 1) / max(dt, 1e-6):.1f} fps)", flush=True)

        ds.save_episode()
        print(f"[episode] 已保存：{episode_path.name}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert a directory of episode_*.hdf5 to one LeRobot v3 dataset.")
    p.add_argument("--input_dataset_dir", type=str, required=True, help="Directory containing episode_*.hdf5 files.")
    p.add_argument("--output_dir", type=str, default="/tmp/web_collection_lerobot", help="Parent output directory.")
    p.add_argument("--repo_id", type=str, required=True, help="LeRobot dataset repo_id (used in metadata).")
    p.add_argument("--robot_type", type=str, default=None, help="Robot type metadata (optional).")
    p.add_argument("--fps", type=int, default=None, help="Override fps (otherwise read from first episode attrs).")
    p.add_argument("--use_videos", action="store_true", help="Store visual keys as videos (mp4).")
    p.add_argument("--swap_rb", action="store_true", help="Swap red/blue channels for all RGB images.")
    p.add_argument("--instruction", type=str, default=None, help="Override instruction/task text for ALL episodes.")
    p.add_argument("--no_base", action="store_true", help="Do not include base_action even if present.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dataset_dir).expanduser().resolve()
    if not in_dir.is_dir():
        raise NotADirectoryError(in_dir)

    repo_id = _sanitize_repo_id(args.repo_id)
    out_parent = Path(args.output_dir).expanduser().resolve()
    # Safety: don't write outputs inside the input dataset folder.
    # This avoids "touching" the original dataset directory even though we only read HDF5 files.
    if out_parent == in_dir or out_parent.is_relative_to(in_dir):
        raise ValueError(
            f"output_dir 不能放在 input_dataset_dir 里面。\n"
            f"input_dataset_dir={in_dir}\n"
            f"output_dir={out_parent}\n"
            f"建议把 output_dir 设到另一个目录（例如 /tmp/web_collection_lerobot 或 datasets 的同级目录）。"
        )
    out_root = _pick_unique_root(out_parent, repo_id.replace("/", "__"))

    episodes = _list_hdf5_episodes(in_dir)

    # Determine fps from first episode (or override).
    with h5py.File(episodes[0], "r") as f0:
        fps = int(args.fps if args.fps is not None else f0.attrs.get("fps", 30))
        include_base = ("base_action" in f0) and (not args.no_base)

    print(f"[start] 输入目录: {in_dir}", flush=True)
    print(f"[start] episode 数量: {len(episodes)}", flush=True)
    print(f"[start] 输出目录: {out_root}", flush=True)
    print(f"[start] repo_id: {repo_id} fps={fps} use_videos={args.use_videos} include_base={include_base}", flush=True)

    features, cam_names = _infer_features_from_episode(
        episodes[0], use_videos=args.use_videos, include_base=include_base
    )
    print(f"[start] cameras: {cam_names}", flush=True)

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=str(out_root),
        robot_type=args.robot_type,
        use_videos=args.use_videos,
    )

    for idx, ep in enumerate(episodes, start=1):
        print(f"[episode] {idx}/{len(episodes)}: {ep}", flush=True)
        _convert_one_episode(
            ds,
            ep,
            swap_rb=args.swap_rb,
            instruction_override=args.instruction,
            expected_cam_names=cam_names,
            include_base=include_base,
        )

    ds.finalize()
    print(f"[ok] wrote LeRobot dataset to: {out_root}")


if __name__ == "__main__":
    main()
