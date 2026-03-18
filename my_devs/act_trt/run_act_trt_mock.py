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
    apply_act_runtime_overrides,
    load_act_config,
    load_act_policy,
    load_model_spec,
    load_pre_post_processors,
    make_mock_observation,
    save_json,
    tensor_to_numpy,
)
from my_devs.act_trt.trt_policy import ActTrtPolicyAdapter
from my_devs.act_trt.trt_runtime import TensorRTRunner

MOCK_EXECUTION_MODULE = (
    __spec__.name if __spec__ is not None and __spec__.name else "my_devs.act_trt.run_act_trt_mock"
)


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
    prepared = prepare_observation_for_inference(
        dict(observation),
        device=device,
        task=task,
        robot_type=robot_type,
    )
    return move_batch_to_device(preprocessor(prepared), device)


def compare_arrays(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float]:
    lhs_f = np.asarray(lhs, dtype=np.float32)
    rhs_f = np.asarray(rhs, dtype=np.float32)
    diff = np.abs(lhs_f - rhs_f)
    denom = np.maximum(np.abs(lhs_f), 1e-8)
    rel = diff / denom
    flat_lhs = lhs_f.reshape(-1)
    flat_rhs = rhs_f.reshape(-1)
    lhs_norm = float(np.linalg.norm(flat_lhs))
    rhs_norm = float(np.linalg.norm(flat_rhs))
    if lhs_norm <= 1e-12 and rhs_norm <= 1e-12:
        cosine = 1.0
    elif lhs_norm <= 1e-12 or rhs_norm <= 1e-12:
        cosine = 0.0
    else:
        cosine = float(np.dot(flat_lhs, flat_rhs) / (lhs_norm * rhs_norm))
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_rel_diff": float(rel.max()),
        "cosine_similarity": cosine,
    }


def merge_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    return {
        "max_abs_diff": max(item["max_abs_diff"] for item in metrics_list),
        "mean_abs_diff": max(item["mean_abs_diff"] for item in metrics_list),
        "max_rel_diff": max(item["max_rel_diff"] for item in metrics_list),
        "min_cosine_similarity": min(item["cosine_similarity"] for item in metrics_list),
    }


def _build_trt_adapter(
    *,
    checkpoint: Path,
    engine_path: Path,
    policy_cfg,
    export_metadata: Path | None,
    device: str,
):
    return ActTrtPolicyAdapter(
        checkpoint=checkpoint,
        engine_path=engine_path,
        config=policy_cfg,
        export_metadata_path=export_metadata,
        device=device,
    )


def _try_describe_engine(engine_path: Path, device: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    try:
        runner = TensorRTRunner(engine_path=engine_path, device=device)
        return [vars(item) for item in runner.describe()], None
    except Exception as exc:  # pragma: no cover - best effort reporting only
        return None, f"{type(exc).__name__}: {exc}"


def _resolve_entrypoint_path(path_value: str | Path | None) -> str | None:
    if path_value is None:
        return None
    path_text = str(path_value).strip()
    if not path_text:
        return None
    try:
        return str(Path(path_text).expanduser().resolve())
    except (OSError, RuntimeError):  # pragma: no cover - defensive only
        return path_text


def _build_entrypoint_trace(
    *,
    cli_entrypoint_path: str | Path | None,
    cli_entrypoint_module: str | None,
) -> dict[str, Any]:
    executed_module_path = str(Path(__file__).resolve())
    resolved_cli_path = _resolve_entrypoint_path(cli_entrypoint_path or (sys.argv[0] if sys.argv else None))
    resolved_cli_module = cli_entrypoint_module or MOCK_EXECUTION_MODULE
    alias_wrapper_used = (
        resolved_cli_module != MOCK_EXECUTION_MODULE
        or (resolved_cli_path is not None and resolved_cli_path != executed_module_path)
    )
    cli_file_name = Path(resolved_cli_path).name if resolved_cli_path is not None else None
    return {
        "executed_module": {
            "module_name": MOCK_EXECUTION_MODULE,
            "file_path": executed_module_path,
            "file_name": Path(executed_module_path).name,
        },
        "cli_entrypoint": {
            "module_name": resolved_cli_module,
            "file_path": resolved_cli_path,
            "file_name": cli_file_name,
            "argv": list(sys.argv),
            "is_alias_wrapper": alias_wrapper_used,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ACT TensorRT mock inference on synthetic observations.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--engine", type=str, required=True)
    parser.add_argument("--export-metadata", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--case", choices=["random", "zeros", "ones", "linspace"], default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-n-action-steps", type=int, default=None)
    parser.add_argument("--policy-temporal-ensemble-coeff", type=float, default=None)
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--robot-type", type=str, default="")
    parser.add_argument("--compare-torch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--report", type=str, default=None)
    return parser.parse_args()


def main(
    *,
    cli_entrypoint_path: str | Path | None = None,
    cli_entrypoint_module: str | None = None,
) -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    engine_path = Path(args.engine).expanduser().resolve()
    export_metadata = Path(args.export_metadata).expanduser().resolve() if args.export_metadata else None
    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else engine_path.parent / "mock_infer_report_trt.json"
    )

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"TensorRT mock inference requires a CUDA device, got {device}")
    if not torch.cuda.is_available():
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
        policy_cfg = load_act_config(checkpoint)
        apply_act_runtime_overrides(
            policy_cfg=policy_cfg,
            policy_n_action_steps=args.policy_n_action_steps,
            policy_temporal_ensemble_coeff=args.policy_temporal_ensemble_coeff,
        )

    trt_policy = _build_trt_adapter(
        checkpoint=checkpoint,
        engine_path=engine_path,
        policy_cfg=policy_cfg,
        export_metadata=export_metadata,
        device=args.device,
    )

    preprocessor, trt_postprocessor = load_pre_post_processors(checkpoint)
    spec = load_model_spec(checkpoint)
    engine_io, engine_io_error = _try_describe_engine(engine_path=engine_path, device=args.device)

    torch_postprocessor = None
    if args.compare_torch:
        _, torch_postprocessor = load_pre_post_processors(checkpoint)

    preprocessor.reset()
    trt_postprocessor.reset()
    trt_policy.reset()
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
        action_norm = trt_policy.select_action(clone_batch(batch))
        t2 = time.perf_counter()
        action_real = trt_postprocessor(action_norm.clone())
        t3 = time.perf_counter()

        prepare_times.append(t1 - t0)
        model_times.append(t2 - t1)
        post_times.append(t3 - t2)
        total_times.append(t3 - loop_start)

        step_payload: dict[str, Any] = {
            "step": step,
            "case": args.case,
            "seed": step_seed,
            "trt_action_norm": tensor_to_numpy(action_norm).reshape(-1).tolist(),
            "trt_action_real": tensor_to_numpy(action_real).reshape(-1).tolist(),
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
            step_payload["torch_vs_trt_action_norm"] = norm_metrics
            step_payload["torch_vs_trt_action_real"] = real_metrics

        step_reports.append(step_payload)

        if args.fps > 0:
            remaining = 1.0 / args.fps - (time.perf_counter() - loop_start)
            if remaining > 0:
                time.sleep(remaining)

    summary: dict[str, Any] = {
        "steps_executed": len(step_reports),
        "avg_prepare_s": float(np.mean(prepare_times)),
        "avg_model_s": float(np.mean(model_times)),
        "avg_post_s": float(np.mean(post_times)),
        "avg_total_s": float(np.mean(total_times)),
        "max_total_s": float(np.max(total_times)),
    }
    if compare_norm_metrics:
        summary["torch_vs_trt_action_norm"] = merge_metrics(compare_norm_metrics)
        summary["torch_vs_trt_action_real"] = merge_metrics(compare_real_metrics)

    payload = {
        **_build_entrypoint_trace(
            cli_entrypoint_path=cli_entrypoint_path,
            cli_entrypoint_module=cli_entrypoint_module,
        ),
        "execution_mode": "mock_only",
        "robot_connection_attempted": False,
        "robot_connected": False,
        "checkpoint": str(checkpoint),
        "engine": str(engine_path),
        "export_metadata": str(export_metadata) if export_metadata is not None else None,
        "device": args.device,
        "case": args.case,
        "steps": int(args.steps),
        "fps": float(args.fps),
        "compare_torch": bool(args.compare_torch),
        "spec": {
            "visual_keys": spec.visual_keys,
            "state_dim": int(spec.state_dim),
            "image_height": int(spec.image_height),
            "image_width": int(spec.image_width),
            "action_dim": int(spec.action_dim),
            "chunk_size": int(spec.chunk_size),
            "n_action_steps": int(spec.n_action_steps),
        },
        "policy_config": {
            "chunk_size": int(policy_cfg.chunk_size),
            "n_action_steps": int(policy_cfg.n_action_steps),
            "temporal_ensemble_coeff": policy_cfg.temporal_ensemble_coeff,
        },
        "engine_io": engine_io,
        "engine_io_error": engine_io_error,
        "summary": summary,
        "steps_detail": step_reports,
    }
    save_json(report_path, payload)
    print(report_path)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
