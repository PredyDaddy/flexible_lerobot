from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import onnxruntime as ort
import torch

from my_devs.act_trt.common import (
    build_case_catalog,
    load_act_policy,
    load_model_spec,
    make_case_inputs,
    save_json,
    tensor_to_numpy,
    torch_core_forward,
)
from my_devs.act_trt.trt_runtime import TensorRTRunner


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Torch vs ONNX vs TensorRT consistency.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--engine", type=str, required=True)
    parser.add_argument("--report", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--random-cases", type=int, default=3)
    parser.add_argument("--threshold-max-abs-diff", type=float, default=1e-4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    onnx_path = Path(args.onnx).expanduser().resolve()
    engine_path = Path(args.engine).expanduser().resolve()
    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else engine_path.parent / f"consistency_report_{engine_path.stem}.json"
    )

    spec = load_model_spec(checkpoint)
    policy = load_act_policy(checkpoint, device="cpu")
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    trt_runner = TensorRTRunner(engine_path=engine_path, device=args.device)

    case_reports: list[dict[str, Any]] = []
    torch_onnx_metrics: list[dict[str, float]] = []
    torch_trt_metrics: list[dict[str, float]] = []
    onnx_trt_metrics: list[dict[str, float]] = []

    for case_info in build_case_catalog(args.random_cases):
        obs_state, img0, img1 = make_case_inputs(spec=spec, seed=case_info["seed"], case=case_info["case"])

        with torch.inference_mode():
            torch_output = tensor_to_numpy(torch_core_forward(policy, obs_state, img0, img1))

        feed_dict = {
            "obs_state_norm": obs_state.numpy(),
            "img0_norm": img0.numpy(),
            "img1_norm": img1.numpy(),
        }
        onnx_output = np.asarray(ort_session.run(["actions_norm"], feed_dict)[0], dtype=np.float32)
        trt_output = tensor_to_numpy(trt_runner.infer(feed_dict)["actions_norm"])

        torch_onnx = compare_arrays(torch_output, onnx_output)
        torch_trt = compare_arrays(torch_output, trt_output)
        onnx_trt = compare_arrays(onnx_output, trt_output)
        torch_onnx_metrics.append(torch_onnx)
        torch_trt_metrics.append(torch_trt)
        onnx_trt_metrics.append(onnx_trt)

        case_reports.append(
            {
                "case": case_info["name"],
                "seed": case_info["seed"],
                "kind": case_info["case"],
                "torch_vs_onnx": torch_onnx,
                "torch_vs_trt": torch_trt,
                "onnx_vs_trt": onnx_trt,
            }
        )

    summary = {
        "torch_vs_onnx": merge_summary(torch_onnx_metrics),
        "torch_vs_trt": merge_summary(torch_trt_metrics),
        "onnx_vs_trt": merge_summary(onnx_trt_metrics),
    }
    passed = (
        summary["torch_vs_onnx"]["max_abs_diff"] <= args.threshold_max_abs_diff
        and summary["torch_vs_trt"]["max_abs_diff"] <= args.threshold_max_abs_diff
        and summary["onnx_vs_trt"]["max_abs_diff"] <= args.threshold_max_abs_diff
    )

    payload = {
        "checkpoint": str(checkpoint),
        "onnx": str(onnx_path),
        "engine": str(engine_path),
        "device": args.device,
        "threshold_max_abs_diff": float(args.threshold_max_abs_diff),
        "spec": {
            "visual_keys": spec.visual_keys,
            "state_dim": spec.state_dim,
            "image_height": spec.image_height,
            "image_width": spec.image_width,
            "action_dim": spec.action_dim,
            "chunk_size": spec.chunk_size,
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
