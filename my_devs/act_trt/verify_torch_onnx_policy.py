from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from lerobot.policies.utils import prepare_observation_for_inference

from my_devs.act_trt.common import (
    apply_act_policy_runtime_overrides,
    build_case_catalog,
    load_model_spec,
    load_act_policy,
    load_pre_post_processors,
    make_mock_observation,
    save_json,
    tensor_to_numpy,
)
from my_devs.act_trt.ort_policy import ActOrtPolicyAdapter


def compare_arrays(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float]:
    lhs_f = np.asarray(lhs, dtype=np.float32)
    rhs_f = np.asarray(rhs, dtype=np.float32)
    diff = np.abs(lhs_f - rhs_f)
    denom = np.maximum(np.abs(lhs_f), 1e-8)
    rel = diff / denom
    flat_lhs = lhs_f.reshape(-1)
    flat_rhs = rhs_f.reshape(-1)
    cosine = float(np.dot(flat_lhs, flat_rhs) / (np.linalg.norm(flat_lhs) * np.linalg.norm(flat_rhs) + 1e-12))
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_rel_diff": float(rel.max()),
        "cosine_similarity": cosine,
    }


def merge_summary(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    return {
        "max_abs_diff": max(item["max_abs_diff"] for item in metrics_list),
        "mean_abs_diff": max(item["mean_abs_diff"] for item in metrics_list),
        "max_rel_diff": max(item["max_rel_diff"] for item in metrics_list),
        "min_cosine_similarity": min(item["cosine_similarity"] for item in metrics_list),
    }


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Torch ACT policy vs ONNXRuntime ACT adapter consistency.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--export-metadata", type=str, default=None)
    parser.add_argument("--report", type=str, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--ort-provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--random-cases", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--policy-n-action-steps", type=int, default=None)
    parser.add_argument("--policy-temporal-ensemble-coeff", type=float, default=None)
    parser.add_argument("--threshold-max-abs-diff-norm", type=float, default=1e-4)
    parser.add_argument("--threshold-max-abs-diff-real", type=float, default=1e-3)
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--robot-type", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    onnx_path = Path(args.onnx).expanduser().resolve()
    export_metadata = Path(args.export_metadata).expanduser().resolve() if args.export_metadata else None
    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else onnx_path.parent / "consistency_report_torch_onnx_policy.json"
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    torch_policy = load_act_policy(checkpoint, device=args.device)
    apply_act_policy_runtime_overrides(
        policy=torch_policy,
        policy_n_action_steps=args.policy_n_action_steps,
        policy_temporal_ensemble_coeff=args.policy_temporal_ensemble_coeff,
    )
    torch_policy.to(device)
    torch_policy.eval()

    ort_providers = ["CPUExecutionProvider"]
    if args.ort_provider == "cuda":
        ort_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ort_policy = ActOrtPolicyAdapter(
        checkpoint=checkpoint,
        onnx_path=onnx_path,
        config=torch_policy.config,
        export_metadata_path=export_metadata,
        providers=ort_providers,
    )

    preprocessor, torch_postprocessor = load_pre_post_processors(checkpoint)
    _, ort_postprocessor = load_pre_post_processors(checkpoint)
    spec = load_model_spec(checkpoint)

    chunk_metrics_all: list[dict[str, float]] = []
    select_norm_metrics_all: list[dict[str, float]] = []
    select_real_metrics_all: list[dict[str, float]] = []
    case_reports: list[dict[str, Any]] = []

    for case_info in build_case_catalog(args.random_cases):
        preprocessor.reset()
        torch_postprocessor.reset()
        ort_postprocessor.reset()
        torch_policy.reset()
        ort_policy.reset()

        base_seed = case_info["seed"]
        chunk_observation = make_mock_observation(spec=spec, seed=base_seed, case=case_info["case"])
        chunk_batch = preprocess_observation(
            chunk_observation,
            preprocessor=preprocessor,
            device=device,
            task=args.task,
            robot_type=args.robot_type,
        )

        with torch.inference_mode():
            torch_chunk = tensor_to_numpy(torch_policy.predict_action_chunk(clone_batch(chunk_batch)))
            ort_chunk = tensor_to_numpy(ort_policy.predict_action_chunk(clone_batch(chunk_batch)))
        chunk_metrics = compare_arrays(torch_chunk, ort_chunk)
        chunk_metrics_all.append(chunk_metrics)

        preprocessor.reset()
        torch_postprocessor.reset()
        ort_postprocessor.reset()
        torch_policy.reset()
        ort_policy.reset()

        step_reports: list[dict[str, Any]] = []
        case_select_norm_metrics: list[dict[str, float]] = []
        case_select_real_metrics: list[dict[str, float]] = []
        for step_index in range(args.sequence_length):
            step_seed = None if base_seed is None else base_seed * 1000 + step_index
            observation = make_mock_observation(spec=spec, seed=step_seed, case=case_info["case"])
            batch = preprocess_observation(
                observation,
                preprocessor=preprocessor,
                device=device,
                task=args.task,
                robot_type=args.robot_type,
            )

            with torch.inference_mode():
                torch_action_norm = torch_policy.select_action(clone_batch(batch))
                ort_action_norm = ort_policy.select_action(clone_batch(batch))
                torch_action_real = torch_postprocessor(torch_action_norm.clone())
                ort_action_real = ort_postprocessor(ort_action_norm.clone())

            torch_action_norm_np = tensor_to_numpy(torch_action_norm)
            ort_action_norm_np = tensor_to_numpy(ort_action_norm)
            torch_action_real_np = tensor_to_numpy(torch_action_real)
            ort_action_real_np = tensor_to_numpy(ort_action_real)

            select_norm_metrics = compare_arrays(torch_action_norm_np, ort_action_norm_np)
            select_real_metrics = compare_arrays(torch_action_real_np, ort_action_real_np)
            case_select_norm_metrics.append(select_norm_metrics)
            case_select_real_metrics.append(select_real_metrics)
            select_norm_metrics_all.append(select_norm_metrics)
            select_real_metrics_all.append(select_real_metrics)

            step_reports.append(
                {
                    "step": step_index,
                    "torch_vs_onnx_action_norm": select_norm_metrics,
                    "torch_vs_onnx_action_real": select_real_metrics,
                }
            )

        case_reports.append(
            {
                "case": case_info["name"],
                "seed": case_info["seed"],
                "kind": case_info["case"],
                "predict_action_chunk_torch_vs_onnx": chunk_metrics,
                "select_action_norm_summary": merge_summary(case_select_norm_metrics),
                "select_action_real_summary": merge_summary(case_select_real_metrics),
                "steps": step_reports,
            }
        )

    summary = {
        "predict_action_chunk_torch_vs_onnx": merge_summary(chunk_metrics_all),
        "select_action_norm_torch_vs_onnx": merge_summary(select_norm_metrics_all),
        "select_action_real_torch_vs_onnx": merge_summary(select_real_metrics_all),
    }
    passed = (
        summary["predict_action_chunk_torch_vs_onnx"]["max_abs_diff"] <= args.threshold_max_abs_diff_norm
        and summary["select_action_norm_torch_vs_onnx"]["max_abs_diff"] <= args.threshold_max_abs_diff_norm
        and summary["select_action_real_torch_vs_onnx"]["max_abs_diff"] <= args.threshold_max_abs_diff_real
    )

    payload = {
        "checkpoint": str(checkpoint),
        "onnx": str(onnx_path),
        "export_metadata": str(export_metadata) if export_metadata is not None else None,
        "device": args.device,
        "ort_provider": args.ort_provider,
        "sequence_length": int(args.sequence_length),
        "threshold_max_abs_diff_norm": float(args.threshold_max_abs_diff_norm),
        "threshold_max_abs_diff_real": float(args.threshold_max_abs_diff_real),
        "policy_config": {
            "chunk_size": int(torch_policy.config.chunk_size),
            "n_action_steps": int(torch_policy.config.n_action_steps),
            "temporal_ensemble_coeff": torch_policy.config.temporal_ensemble_coeff,
            "image_features": list(getattr(torch_policy.config, "image_features", []) or []),
        },
        "summary": summary,
        "passed": bool(passed),
        "cases": case_reports,
    }
    save_json(report_path, payload)
    print(report_path)
    print(payload["summary"])
    print(f"passed={passed}")
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
