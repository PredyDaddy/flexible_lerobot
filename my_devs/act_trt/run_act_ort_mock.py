from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from lerobot.policies.utils import prepare_observation_for_inference

from my_devs.act_trt.common import (
    apply_act_policy_runtime_overrides,
    load_model_spec,
    load_act_policy,
    load_pre_post_processors,
    make_mock_observation,
    save_json,
    tensor_to_numpy,
)
from my_devs.act_trt.ort_policy import ActOrtPolicyAdapter


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.clone()
        elif isinstance(value, list):
            cloned[key] = [item.clone() if isinstance(item, torch.Tensor) else item for item in value]
        else:
            cloned[key] = value
    return cloned


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, list):
            moved[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in value]
        else:
            moved[key] = value
    return moved


def preprocess_observation(
    observation: dict[str, np.ndarray],
    *,
    preprocessor,
    device: torch.device,
    task: str,
    robot_type: str,
) -> dict[str, Any]:
    prepared = prepare_observation_for_inference(dict(observation), device=device, task=task, robot_type=robot_type)
    return move_batch_to_device(preprocessor(prepared), device)


def compare_arrays(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float]:
    lhs_f = np.asarray(lhs, dtype=np.float32)
    rhs_f = np.asarray(rhs, dtype=np.float32)
    diff = np.abs(lhs_f - rhs_f)
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ACT ONNXRuntime inference on synthetic observations.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--export-metadata", type=str, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--ort-provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--case", choices=["random", "zeros", "ones", "linspace"], default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-n-action-steps", type=int, default=None)
    parser.add_argument("--policy-temporal-ensemble-coeff", type=float, default=None)
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--robot-type", type=str, default="")
    parser.add_argument("--compare-torch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    onnx_path = Path(args.onnx).expanduser().resolve()
    export_metadata = Path(args.export_metadata).expanduser().resolve() if args.export_metadata else None
    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else onnx_path.parent / "mock_infer_report_ort.json"
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    torch_policy = None
    if args.compare_torch:
        torch_policy = load_act_policy(checkpoint, device=args.device)
        apply_act_policy_runtime_overrides(
            policy=torch_policy,
            policy_n_action_steps=args.policy_n_action_steps,
            policy_temporal_ensemble_coeff=args.policy_temporal_ensemble_coeff,
        )
        torch_policy.to(device)
        torch_policy.eval()
        policy_cfg = torch_policy.config
    else:
        torch_policy = load_act_policy(checkpoint, device=args.device)
        apply_act_policy_runtime_overrides(
            policy=torch_policy,
            policy_n_action_steps=args.policy_n_action_steps,
            policy_temporal_ensemble_coeff=args.policy_temporal_ensemble_coeff,
        )
        policy_cfg = torch_policy.config

    ort_providers = ["CPUExecutionProvider"]
    if args.ort_provider == "cuda":
        ort_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ort_policy = ActOrtPolicyAdapter(
        checkpoint=checkpoint,
        onnx_path=onnx_path,
        config=policy_cfg,
        export_metadata_path=export_metadata,
        providers=ort_providers,
    )

    preprocessor, ort_postprocessor = load_pre_post_processors(checkpoint)
    spec = load_model_spec(checkpoint)

    torch_postprocessor = None
    if args.compare_torch:
        _, torch_postprocessor = load_pre_post_processors(checkpoint)

    preprocessor.reset()
    ort_postprocessor.reset()
    ort_policy.reset()
    if torch_policy is not None:
        torch_policy.reset()
    if torch_postprocessor is not None:
        torch_postprocessor.reset()

    step_reports: list[dict[str, Any]] = []
    prepare_times: list[float] = []
    model_times: list[float] = []
    post_times: list[float] = []
    total_times: list[float] = []
    compare_norm_metrics: list[dict[str, float]] = []
    compare_real_metrics: list[dict[str, float]] = []

    for step in range(args.steps):
        loop_start = time.perf_counter()
        step_seed = args.seed * 1000 + step if args.case == "random" else args.seed
        observation = make_mock_observation(spec=spec, seed=step_seed, case=args.case)

        t0 = time.perf_counter()
        batch = preprocess_observation(
            observation,
            preprocessor=preprocessor,
            device=device,
            task=args.task,
            robot_type=args.robot_type,
        )
        t1 = time.perf_counter()
        action_norm = ort_policy.select_action(clone_batch(batch))
        t2 = time.perf_counter()
        action_real = ort_postprocessor(action_norm.clone())
        t3 = time.perf_counter()

        prepare_times.append(t1 - t0)
        model_times.append(t2 - t1)
        post_times.append(t3 - t2)
        total_times.append(t3 - loop_start)

        step_payload: dict[str, Any] = {
            "step": step,
            "ort_action_norm": tensor_to_numpy(action_norm).reshape(-1).tolist(),
            "ort_action_real": tensor_to_numpy(action_real).reshape(-1).tolist(),
            "timing_s": {
                "prepare": t1 - t0,
                "model": t2 - t1,
                "post": t3 - t2,
                "total": t3 - loop_start,
            },
        }

        if torch_policy is not None and torch_postprocessor is not None:
            with torch.inference_mode():
                torch_action_norm = torch_policy.select_action(clone_batch(batch))
                torch_action_real = torch_postprocessor(torch_action_norm.clone())
            norm_metrics = compare_arrays(tensor_to_numpy(torch_action_norm), tensor_to_numpy(action_norm))
            real_metrics = compare_arrays(tensor_to_numpy(torch_action_real), tensor_to_numpy(action_real))
            compare_norm_metrics.append(norm_metrics)
            compare_real_metrics.append(real_metrics)
            step_payload["torch_vs_ort_action_norm"] = norm_metrics
            step_payload["torch_vs_ort_action_real"] = real_metrics

        step_reports.append(step_payload)

        if args.fps > 0:
            remaining = 1.0 / args.fps - (time.perf_counter() - loop_start)
            if remaining > 0:
                time.sleep(remaining)

    summary: dict[str, Any] = {
        "avg_prepare_s": float(np.mean(prepare_times)),
        "avg_model_s": float(np.mean(model_times)),
        "avg_post_s": float(np.mean(post_times)),
        "avg_total_s": float(np.mean(total_times)),
        "max_total_s": float(np.max(total_times)),
    }
    if compare_norm_metrics:
        summary["torch_vs_ort_action_norm_max_abs_diff"] = max(item["max_abs_diff"] for item in compare_norm_metrics)
        summary["torch_vs_ort_action_real_max_abs_diff"] = max(item["max_abs_diff"] for item in compare_real_metrics)

    payload = {
        "checkpoint": str(checkpoint),
        "onnx": str(onnx_path),
        "export_metadata": str(export_metadata) if export_metadata is not None else None,
        "device": args.device,
        "ort_provider": args.ort_provider,
        "case": args.case,
        "steps": int(args.steps),
        "fps": float(args.fps),
        "compare_torch": bool(args.compare_torch),
        "policy_config": {
            "chunk_size": int(policy_cfg.chunk_size),
            "n_action_steps": int(policy_cfg.n_action_steps),
            "temporal_ensemble_coeff": policy_cfg.temporal_ensemble_coeff,
        },
        "summary": summary,
        "steps_detail": step_reports,
    }
    save_json(report_path, payload)
    print(report_path)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
