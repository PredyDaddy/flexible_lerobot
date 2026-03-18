#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    numpy_metrics,
    prepare_batch,
    prepare_policy_for_fp16,
    preprocess_observation,
    resolve_policy_dir,
    save_json,
    summarize_metrics,
    tensor_to_numpy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare PI0.5 Torch FP16 reference vs TensorRT.")
    parser.add_argument("--policy-path", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--tensorrt-py-dir", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task", default="Put the block in the bin")
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
    image_embeddings_trt = sessions["vision"].run({"image_tensor": prepared.image_tensor})["image_embeddings"].to(torch.float16)

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
    (
        prefix_embs_trt,
        prefix_pad_masks_trt,
        _prefix_att_masks_trt,
        prefix_attention_mask_4d_trt,
        prefix_position_ids_trt,
    ) = build_prefix_from_image_embeddings(
        policy,
        image_embeddings_trt,
        prepared.image_masks,
        prepared.tokens,
        prepared.masks,
    )

    kv_cache_torch = compute_prefix_cache(policy, prefix_embs_torch, prefix_attention_mask_4d_torch, prefix_position_ids_torch)
    kv_cache_trt = sessions["prefix"].run(
        {
            "prefix_embs": prefix_embs_trt,
            "prefix_attention_mask_4d": prefix_attention_mask_4d_trt,
            "prefix_position_ids": prefix_position_ids_trt,
        }
    )["kv_cache"].to(torch.float16)

    x_t = torch.randn(
        (1, int(policy.config.chunk_size), int(policy.config.max_action_dim)),
        dtype=torch.float16,
        device="cuda",
    )
    timestep = torch.full((1,), 1.0, dtype=torch.float32, device="cuda")
    velocity_torch = compute_denoise_step(policy, x_t, timestep, prefix_pad_masks_torch, kv_cache_torch)
    velocity_trt = sessions["denoise"].run(
        {
            "x_t": x_t,
            "timestep": timestep,
            "prefix_pad_masks": prefix_pad_masks_trt,
            "kv_cache": kv_cache_trt,
        }
    )["velocity"].to(torch.float16)

    torch.manual_seed(args.seed + 100)
    torch.cuda.manual_seed_all(args.seed + 100)
    noise = torch.randn(
        (1, int(policy.config.chunk_size), int(policy.config.max_action_dim)),
        dtype=torch.float16,
        device="cuda",
    )
    chunk_torch = compute_action_chunk(policy, prepared, noise=noise.clone())

    x_t_trt = noise.clone()
    dt = -1.0 / float(policy.config.num_inference_steps)
    for step in range(int(policy.config.num_inference_steps)):
        time_value = 1.0 + step * dt
        timestep_step = torch.full((1,), time_value, dtype=torch.float32, device="cuda")
        velocity_step = sessions["denoise"].run(
            {
                "x_t": x_t_trt,
                "timestep": timestep_step,
                "prefix_pad_masks": prefix_pad_masks_trt,
                "kv_cache": kv_cache_trt,
            }
        )["velocity"].to(torch.float16)
        x_t_trt = (x_t_trt + dt * velocity_step).to(torch.float16)
    chunk_trt = x_t_trt

    report = {
        "policy_path": str(policy_path),
        "engine_dir": str(engine_dir),
        "tensorrt_version": sessions["vision"].trt.__version__,
        "engines": {name: session.engine_path.as_posix() for name, session in sessions.items()},
        "results": {
            "vision_encoder": numpy_metrics(tensor_to_numpy(image_embeddings_torch), tensor_to_numpy(image_embeddings_trt)),
            "prefix_cache": numpy_metrics(tensor_to_numpy(kv_cache_torch), tensor_to_numpy(kv_cache_trt)),
            "denoise_step": numpy_metrics(tensor_to_numpy(velocity_torch), tensor_to_numpy(velocity_trt)),
            "action_chunk_pipeline": numpy_metrics(tensor_to_numpy(chunk_torch), tensor_to_numpy(chunk_trt)),
        },
    }
    report["summary"] = summarize_metrics(report["results"])

    for key, value in report["results"].items():
        print(
            f"[RESULT] {key}: cos={value['cosine']:.8f}, rmse={value['rmse']:.8f}, "
            f"mean_abs={value['mean_abs']:.8f}, max_abs={value['max_abs']:.8f}"
        )

    json_out = Path(args.json_out).expanduser() if args.json_out else engine_dir / "compare_metrics_torch_trt.json"
    save_json(json_out, report)
    print(f"[OK] Comparison report saved to: {json_out}")


if __name__ == "__main__":
    main()
