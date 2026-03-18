#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import onnxruntime as ort
import torch


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Failed to locate repo root from: {start}")


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

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


def create_session(onnx_path: Path) -> ort.InferenceSession:
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=["CPUExecutionProvider"])


def run_ort(session: ort.InferenceSession, inputs: dict[str, Any]) -> dict[str, Any]:
    return dict(zip([output.name for output in session.get_outputs()], session.run(None, inputs), strict=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare PI0.5 Torch FP16 reference vs ONNX.")
    parser.add_argument("--policy-path", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task", default="Put the block in the bin")
    return parser


@torch.inference_mode()
def main() -> None:
    args = build_parser().parse_args()
    policy_path = resolve_policy_dir(args.policy_path)
    onnx_dir = Path(args.onnx_dir).expanduser()
    required_paths = {
        "vision": onnx_dir / "vision_encoder_fp16.onnx",
        "prefix": onnx_dir / "prefix_cache_fp16.onnx",
        "denoise": onnx_dir / "denoise_step_fp16.onnx",
    }
    missing_paths = [path for path in required_paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError("Missing ONNX files:\n" + "\n".join(f"  - {path}" for path in missing_paths))

    _, policy = load_policy(policy_path, device="cuda")
    prepare_policy_for_fp16(policy)
    preprocessor = load_preprocessor(policy_path)
    batch = preprocess_observation(policy, preprocessor, seed=args.seed, task=args.task)
    prepared = prepare_batch(policy, batch)

    sessions = {name: create_session(path) for name, path in required_paths.items()}

    image_embeddings_torch = policy.model.paligemma_with_expert.embed_image(prepared.image_tensor).to(torch.float16)
    image_embeddings_onnx = torch.from_numpy(
        run_ort(
            sessions["vision"],
            {"image_tensor": tensor_to_numpy(prepared.image_tensor)},
        )["image_embeddings"]
    ).to(torch.float16)

    prefix_embs_torch, prefix_pad_masks_torch, _prefix_att_masks_torch, prefix_attention_mask_4d_torch, prefix_position_ids_torch = (
        build_prefix_from_image_embeddings(
            policy,
            image_embeddings_torch,
            prepared.image_masks,
            prepared.tokens,
            prepared.masks,
        )
    )
    prefix_embs_onnx, prefix_pad_masks_onnx, _prefix_att_masks_onnx, prefix_attention_mask_4d_onnx, prefix_position_ids_onnx = (
        build_prefix_from_image_embeddings(
            policy,
            image_embeddings_onnx.to(device=prepared.image_tensor.device),
            prepared.image_masks,
            prepared.tokens,
            prepared.masks,
        )
    )

    kv_cache_torch = compute_prefix_cache(policy, prefix_embs_torch, prefix_attention_mask_4d_torch, prefix_position_ids_torch)
    kv_cache_onnx = torch.from_numpy(
        run_ort(
            sessions["prefix"],
            {
                "prefix_embs": tensor_to_numpy(prefix_embs_onnx),
                "prefix_attention_mask_4d": tensor_to_numpy(prefix_attention_mask_4d_onnx),
                "prefix_position_ids": tensor_to_numpy(prefix_position_ids_onnx),
            },
        )["kv_cache"]
    ).to(torch.float16)

    x_t = torch.randn(
        (1, int(policy.config.chunk_size), int(policy.config.max_action_dim)),
        dtype=torch.float16,
        device="cuda",
    )
    timestep = torch.full((1,), 1.0, dtype=torch.float32, device="cuda")
    velocity_torch = compute_denoise_step(policy, x_t, timestep, prefix_pad_masks_torch, kv_cache_torch)
    velocity_onnx = torch.from_numpy(
        run_ort(
            sessions["denoise"],
            {
                "x_t": tensor_to_numpy(x_t),
                "timestep": tensor_to_numpy(timestep),
                "prefix_pad_masks": tensor_to_numpy(prefix_pad_masks_onnx),
                "kv_cache": tensor_to_numpy(kv_cache_onnx),
            },
        )["velocity"]
    ).to(torch.float16)

    torch.manual_seed(args.seed + 100)
    noise = torch.randn(
        (1, int(policy.config.chunk_size), int(policy.config.max_action_dim)),
        dtype=torch.float16,
        device="cuda",
    )
    chunk_torch = compute_action_chunk(policy, prepared, noise=noise.clone())

    x_t_onnx = noise.detach().cpu().numpy()
    dt = -1.0 / float(policy.config.num_inference_steps)
    for step in range(int(policy.config.num_inference_steps)):
        time_value = 1.0 + step * dt
        timestep_np = torch.full((1,), time_value, dtype=torch.float32).numpy()
        velocity_step = run_ort(
            sessions["denoise"],
            {
                "x_t": x_t_onnx,
                "timestep": timestep_np,
                "prefix_pad_masks": tensor_to_numpy(prefix_pad_masks_onnx),
                "kv_cache": tensor_to_numpy(kv_cache_onnx),
            },
        )["velocity"]
        x_t_onnx = (x_t_onnx + dt * velocity_step).astype("float16")
    chunk_onnx = x_t_onnx

    report = {
        "policy_path": str(policy_path),
        "onnx_dir": str(onnx_dir),
        "providers": ort.get_available_providers(),
        "results": {
            "vision_encoder": numpy_metrics(tensor_to_numpy(image_embeddings_torch), tensor_to_numpy(image_embeddings_onnx)),
            "prefix_cache": numpy_metrics(tensor_to_numpy(kv_cache_torch), tensor_to_numpy(kv_cache_onnx)),
            "denoise_step": numpy_metrics(tensor_to_numpy(velocity_torch), tensor_to_numpy(velocity_onnx)),
            "action_chunk_pipeline": numpy_metrics(tensor_to_numpy(chunk_torch), chunk_onnx),
        },
    }
    report["summary"] = summarize_metrics(report["results"])

    for key, value in report["results"].items():
        print(
            f"[RESULT] {key}: cos={value['cosine']:.8f}, rmse={value['rmse']:.8f}, "
            f"mean_abs={value['mean_abs']:.8f}, max_abs={value['max_abs']:.8f}"
        )
    json_out = Path(args.json_out).expanduser() if args.json_out else onnx_dir / "compare_metrics_torch_onnx.json"
    save_json(json_out, report)
    print(f"[OK] Comparison report saved to: {json_out}")


if __name__ == "__main__":
    main()
