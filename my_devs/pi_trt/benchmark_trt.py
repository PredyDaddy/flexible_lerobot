#!/usr/bin/env python

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Failed to locate repo root from: {start}")


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from my_devs.groot_trt.trt_utils import TrtSession
from my_devs.pi_trt.common import (
    DEFAULT_POLICY_PATH,
    build_prefix_from_image_embeddings,
    compute_action_chunk,
    compute_denoise_step,
    compute_prefix_cache,
    load_policy,
    load_preprocessor,
    prepare_batch,
    prepare_policy_for_fp16,
    preprocess_observation,
    resolve_policy_dir,
    save_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark PI0.5 Torch FP16 vs TensorRT FP16.")
    parser.add_argument("--policy-path", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--tensorrt-py-dir", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task", default="Put the block in the bin")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    return parser


def load_sessions(engine_dir: Path, tensorrt_py_dir: str | None) -> dict[str, TrtSession]:
    required_paths = {
        "vision": engine_dir / "vision_encoder_fp16.engine",
        "prefix": engine_dir / "prefix_cache_fp16.engine",
        "denoise": engine_dir / "denoise_step_fp16.engine",
    }
    missing_paths = [path for path in required_paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError("Missing TensorRT engine files:\n" + "\n".join(f"  - {path}" for path in missing_paths))
    return {name: TrtSession.load(path, trt_py_dir=tensorrt_py_dir) for name, path in required_paths.items()}


def summarize_ms(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms:
        raise ValueError("samples_ms must not be empty")
    samples_np = np.asarray(samples_ms, dtype=np.float64)
    return {
        "avg_ms": float(samples_np.mean()),
        "p50_ms": float(np.percentile(samples_np, 50)),
        "p95_ms": float(np.percentile(samples_np, 95)),
        "min_ms": float(samples_np.min()),
        "max_ms": float(samples_np.max()),
        "stdev_ms": float(statistics.pstdev(samples_ms)) if len(samples_ms) > 1 else 0.0,
    }


def benchmark_cuda_callable(
    fn: Callable[[], Any],
    *,
    warmup_iters: int,
    iters: int,
) -> dict[str, Any]:
    if warmup_iters < 0:
        raise ValueError(f"warmup_iters must be >= 0, got {warmup_iters}")
    if iters <= 0:
        raise ValueError(f"iters must be > 0, got {iters}")

    for _ in range(warmup_iters):
        _ = fn()
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    last_output = None
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        last_output = fn()
        end.record()
        torch.cuda.synchronize()
        samples_ms.append(float(start.elapsed_time(end)))

    return {
        "timing": summarize_ms(samples_ms),
        "last_output_type": type(last_output).__name__,
    }


def compute_action_chunk_trt(
    sessions: dict[str, TrtSession],
    prefix_pad_masks: torch.Tensor,
    kv_cache: torch.Tensor,
    noise: torch.Tensor,
    *,
    num_steps: int,
) -> torch.Tensor:
    x_t = noise.to(torch.float16).contiguous()
    dt = -1.0 / float(num_steps)
    for step in range(num_steps):
        time_value = 1.0 + step * dt
        timestep = torch.full((1,), time_value, dtype=torch.float32, device=x_t.device)
        velocity = sessions["denoise"].run(
            {
                "x_t": x_t,
                "timestep": timestep,
                "prefix_pad_masks": prefix_pad_masks,
                "kv_cache": kv_cache,
            }
        )["velocity"].to(torch.float16)
        x_t = (x_t + dt * velocity).to(torch.float16)
    return x_t


def add_speedup(report: dict[str, Any]) -> None:
    torch_avg = float(report["torch"]["avg_ms"])
    trt_avg = float(report["trt"]["avg_ms"])
    report["speedup_vs_torch"] = torch_avg / trt_avg if trt_avg > 0 else float("inf")


@torch.inference_mode()
def main() -> None:
    args = build_parser().parse_args()
    policy_path = resolve_policy_dir(args.policy_path)
    engine_dir = Path(args.engine_dir).expanduser()
    if not engine_dir.is_dir():
        raise FileNotFoundError(f"--engine-dir does not exist: {engine_dir}")

    _, policy = load_policy(policy_path, device="cuda")
    prepare_policy_for_fp16(policy)
    preprocessor = load_preprocessor(policy_path)
    batch = preprocess_observation(policy, preprocessor, seed=args.seed, task=args.task)
    prepared = prepare_batch(policy, batch)
    sessions = load_sessions(engine_dir, args.tensorrt_py_dir)

    image_embeddings_torch = policy.model.paligemma_with_expert.embed_image(prepared.image_tensor).to(torch.float16)
    (
        prefix_embs_torch,
        prefix_pad_masks_torch,
        _prefix_att_masks_torch,
        prefix_attention_mask_4d_torch,
        prefix_position_ids_torch,
    ) = build_prefix_from_image_embeddings(
        policy,
        image_embeddings_torch,
        prepared.image_masks,
        prepared.tokens,
        prepared.masks,
    )
    kv_cache_torch = compute_prefix_cache(policy, prefix_embs_torch, prefix_attention_mask_4d_torch, prefix_position_ids_torch)

    prefix_embs_trt = prefix_embs_torch.clone()
    prefix_pad_masks_trt = prefix_pad_masks_torch.clone()
    prefix_attention_mask_4d_trt = prefix_attention_mask_4d_torch.clone()
    prefix_position_ids_trt = prefix_position_ids_torch.clone()
    kv_cache_trt = sessions["prefix"].run(
        {
            "prefix_embs": prefix_embs_trt,
            "prefix_attention_mask_4d": prefix_attention_mask_4d_trt,
            "prefix_position_ids": prefix_position_ids_trt,
        }
    )["kv_cache"].to(torch.float16)

    torch.manual_seed(args.seed + 100)
    torch.cuda.manual_seed_all(args.seed + 100)
    x_t = torch.randn(
        (1, int(policy.config.chunk_size), int(policy.config.max_action_dim)),
        dtype=torch.float16,
        device="cuda",
    )
    timestep = torch.full((1,), 1.0, dtype=torch.float32, device="cuda")
    noise = x_t.clone()

    benchmark_results: dict[str, dict[str, Any]] = {}

    benchmark_results["vision_encoder"] = {
        "torch": benchmark_cuda_callable(
            lambda: policy.model.paligemma_with_expert.embed_image(prepared.image_tensor).to(torch.float16),
            warmup_iters=args.warmup_iters,
            iters=args.iters,
        )["timing"],
        "trt": benchmark_cuda_callable(
            lambda: sessions["vision"].run({"image_tensor": prepared.image_tensor})["image_embeddings"],
            warmup_iters=args.warmup_iters,
            iters=args.iters,
        )["timing"],
    }
    add_speedup(benchmark_results["vision_encoder"])

    benchmark_results["prefix_cache"] = {
        "torch": benchmark_cuda_callable(
            lambda: compute_prefix_cache(policy, prefix_embs_torch, prefix_attention_mask_4d_torch, prefix_position_ids_torch),
            warmup_iters=args.warmup_iters,
            iters=args.iters,
        )["timing"],
        "trt": benchmark_cuda_callable(
            lambda: sessions["prefix"].run(
                {
                    "prefix_embs": prefix_embs_trt,
                    "prefix_attention_mask_4d": prefix_attention_mask_4d_trt,
                    "prefix_position_ids": prefix_position_ids_trt,
                }
            )["kv_cache"],
            warmup_iters=args.warmup_iters,
            iters=args.iters,
        )["timing"],
    }
    add_speedup(benchmark_results["prefix_cache"])

    benchmark_results["denoise_step"] = {
        "torch": benchmark_cuda_callable(
            lambda: compute_denoise_step(policy, x_t, timestep, prefix_pad_masks_torch, kv_cache_torch),
            warmup_iters=args.warmup_iters,
            iters=args.iters,
        )["timing"],
        "trt": benchmark_cuda_callable(
            lambda: sessions["denoise"].run(
                {
                    "x_t": x_t,
                    "timestep": timestep,
                    "prefix_pad_masks": prefix_pad_masks_trt,
                    "kv_cache": kv_cache_trt,
                }
            )["velocity"],
            warmup_iters=args.warmup_iters,
            iters=args.iters,
        )["timing"],
    }
    add_speedup(benchmark_results["denoise_step"])

    benchmark_results["action_chunk_pipeline"] = {
        "torch": benchmark_cuda_callable(
            lambda: compute_action_chunk(policy, prepared, noise=noise.clone()),
            warmup_iters=args.warmup_iters,
            iters=args.iters,
        )["timing"],
        "trt": benchmark_cuda_callable(
            lambda: compute_action_chunk_trt(
                sessions,
                prefix_pad_masks_trt,
                kv_cache_trt,
                noise.clone(),
                num_steps=int(policy.config.num_inference_steps),
            ),
            warmup_iters=args.warmup_iters,
            iters=args.iters,
        )["timing"],
    }
    add_speedup(benchmark_results["action_chunk_pipeline"])

    report = {
        "policy_path": str(policy_path),
        "engine_dir": str(engine_dir),
        "tensorrt_version": sessions["vision"].trt.__version__,
        "seed": int(args.seed),
        "task": args.task,
        "warmup_iters": int(args.warmup_iters),
        "iters": int(args.iters),
        "results": benchmark_results,
    }

    for stage_name, stage_report in report["results"].items():
        print(
            f"[BENCH] {stage_name}: "
            f"torch_avg={stage_report['torch']['avg_ms']:.3f}ms, "
            f"trt_avg={stage_report['trt']['avg_ms']:.3f}ms, "
            f"speedup={stage_report['speedup_vs_torch']:.2f}x"
        )

    json_out = Path(args.json_out).expanduser() if args.json_out else engine_dir / "benchmark_metrics_torch_trt.json"
    save_json(json_out, report)
    print(f"[OK] Benchmark report saved to: {json_out}")


if __name__ == "__main__":
    main()
