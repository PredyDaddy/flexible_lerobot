#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path("/data/cqy_workspace/flexible_lerobot")
VLASH_ROOT = REPO_ROOT / "my_devs/vla_engineering/vlash-main"
for path in (VLASH_ROOT, REPO_ROOT / "src"):
    if path.as_posix() not in sys.path:
        sys.path.insert(0, path.as_posix())

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from vlash.policies.factory import get_policy_class


def ensure_offline_defaults() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a VLASH PI0.5 checkpoint and run one action-chunk prediction.")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "datasets/desk_cleanup_v1/eraser_cup_multi_task"))
    parser.add_argument("--repo-id", default="desk_cleanup_v1/eraser_cup_multi_task")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()

    ensure_offline_defaults()

    policy_path = Path(args.policy_path)
    required = ["config.json", "model.safetensors", "train_config.json"]
    missing = [name for name in required if not (policy_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint files in {policy_path}: {missing}")

    cfg = PreTrainedConfig.from_pretrained(
        policy_path,
        cli_overrides=[
            f"--device={args.device}",
            f"--num_inference_steps={args.num_inference_steps}",
            "--compile_model=false",
            "--fuse_qkv=false",
            "--fuse_gate_up=false",
        ],
    )
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(policy_path, config=cfg, dataset_stats=None)
    policy.eval()

    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        video_backend=args.video_backend,
    )
    sample = dataset[args.sample_index]
    batch = {}
    for key, value in sample.items():
        if key.startswith("observation.") and hasattr(value, "unsqueeze"):
            batch[key] = value.unsqueeze(0).to(args.device)
        elif key == "task":
            batch[key] = [value]

    if "task" not in batch:
        batch["task"] = ["Put the eraser into the small box"]

    with torch.inference_mode():
        actions = policy.predict_action_chunk(batch)

    summary = {
        "policy_path": str(policy_path),
        "policy_type": cfg.type,
        "device": args.device,
        "num_inference_steps": args.num_inference_steps,
        "action_shape": list(actions.shape),
        "action_dtype": str(actions.dtype),
        "finite": bool(torch.isfinite(actions).all().item()),
        "first_action": actions[0, 0].detach().float().cpu().tolist(),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
