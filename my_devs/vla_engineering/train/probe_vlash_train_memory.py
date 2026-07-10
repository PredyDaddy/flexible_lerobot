#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path("/data/cqy_workspace/flexible_lerobot")
VLASH_ROOT = REPO_ROOT / "my_devs/vla_engineering/vlash-main"
for path in (VLASH_ROOT, REPO_ROOT / "src"):
    path_str = path.as_posix()
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from accelerate import Accelerator

from vlash.configs.train_config import VLASHTrainConfig
from vlash.lora.apply import apply_lora
from vlash.policies.factory import make_policy
from vlash.train import make_optimizer_and_scheduler, make_vlash_dataset, update_policy


def ensure_offline_defaults() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("WANDB_DISABLED", "true")


def cuda_mem() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_gib": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gib": torch.cuda.memory_reserved() / 1024**3,
        "max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "max_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe one-batch VLASH training memory for a config.")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--skip-backward", action="store_true")
    args, cli_overrides = parser.parse_known_args()

    ensure_offline_defaults()
    cfg = VLASHTrainConfig.from_pretrained(args.config_path, cli_args=cli_overrides)
    cfg.output_dir = REPO_ROOT / f"my_devs/vla_engineering/train/tmp/memory_probe_output_{os.getpid()}"
    cfg.validate()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    total_start = time.perf_counter()
    accelerator = Accelerator(step_scheduler_with_optimizer=False)
    device = accelerator.device

    dataset_start = time.perf_counter()
    dataset = make_vlash_dataset(cfg)
    dataset_s = time.perf_counter() - dataset_start
    after_dataset = cuda_mem()

    policy_start = time.perf_counter()
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    policy_s = time.perf_counter() - policy_start
    after_policy = cuda_mem()

    lora_start = time.perf_counter()
    apply_lora(cfg.lora, policy, verbose=False)
    lora_s = time.perf_counter() - lora_start
    after_lora = cuda_mem()

    optimizer_start = time.perf_counter()
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    optimizer_s = time.perf_counter() - optimizer_start
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(policy, optimizer, dataloader, lr_scheduler)
    batch_start = time.perf_counter()
    batch = next(iter(dataloader))
    batch_s = time.perf_counter() - batch_start
    after_batch = cuda_mem()

    policy.train()
    result = {
        "config_path": str(Path(args.config_path).resolve()),
        "device": str(device),
        "batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "lora_enabled": cfg.lora.enable,
        "qlora_enabled": cfg.lora.use_qlora,
        "elapsed_dataset_s": dataset_s,
        "elapsed_policy_s": policy_s,
        "elapsed_lora_s": lora_s,
        "elapsed_optimizer_s": optimizer_s,
        "elapsed_batch_s": batch_s,
        "after_dataset": after_dataset,
        "after_policy": after_policy,
        "after_lora": after_lora,
        "after_batch": after_batch,
    }

    if args.skip_backward:
        forward_start = time.perf_counter()
        with torch.inference_mode():
            loss, _ = policy.forward(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result["elapsed_forward_s"] = time.perf_counter() - forward_start
        result["loss"] = float(loss.detach().cpu())
        result["after_forward"] = cuda_mem()
    else:
        from lerobot.utils.logging_utils import AverageMeter, MetricsTracker

        tracker = MetricsTracker(
            cfg.batch_size,
            dataset.num_frames,
            dataset.num_episodes,
            {
                "loss": AverageMeter("loss", ":.3f"),
                "grad_norm": AverageMeter("grdn", ":.3f"),
                "lr": AverageMeter("lr", ":0.1e"),
                "update_s": AverageMeter("updt_s", ":.3f"),
                "dataloading_s": AverageMeter("data_s", ":.3f"),
            },
            accelerator=accelerator,
        )
        train_step_start = time.perf_counter()
        tracker, output_dict = update_policy(
            tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result["elapsed_train_step_s"] = time.perf_counter() - train_step_start
        result["loss"] = float(tracker.loss.avg)
        result["output_keys"] = sorted(output_dict.keys()) if output_dict else []
        result["after_backward_step"] = cuda_mem()

    result["elapsed_total_s"] = time.perf_counter() - total_start
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
