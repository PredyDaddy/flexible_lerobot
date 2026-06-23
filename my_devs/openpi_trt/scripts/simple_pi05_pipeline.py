#!/usr/bin/env python

"""Simple PI0.5 split TensorRT pipeline: export, convert, and validate."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
OPENPI_TRT_DIR = SCRIPT_DIR.parent
REPO_ROOT = OPENPI_TRT_DIR.parents[1]
for path in (OPENPI_TRT_DIR, REPO_ROOT):
    if path.as_posix() not in sys.path:
        sys.path.insert(0, path.as_posix())

from runtime.simple_pi05_split import SimplePI05SplitTRTRuntime, SimplePI05TRTProfile  # noqa: E402
from scripts.pi05_onnx_common import (  # noqa: E402
    DEFAULT_POLICY_PATH,
    DEFAULT_TASK,
    PI05DenoiseStepONNXWrapper,
    PI05PrefixCacheONNXWrapper,
    configure_runtime,
    denoise_step_input_names,
    load_policy,
    make_denoise_step_inputs,
    make_export_inputs,
    make_policy_batch,
    make_prefix_cache_inputs,
    patch_transformers_for_onnx_export,
    prefix_cache_tensor_names,
    tensor_stats,
)


ARTIFACT_DIR = Path("my_devs/openpi_trt/artifacts")
PREFIX_ONNX = ARTIFACT_DIR / "pi05_so101_prefix_cache_b1_fp32.onnx"
DENOISE_ONNX = ARTIFACT_DIR / "pi05_so101_denoise_step_b1_fp32.onnx"
PREFIX_ENGINE = ARTIFACT_DIR / "pi05_so101_prefix_cache_b1_fp32.engine"
DENOISE_FP32_ENGINE = ARTIFACT_DIR / "pi05_so101_denoise_step_b1_fp32.engine"
DENOISE_FP16_CONSTRAINED_ENGINE = ARTIFACT_DIR / "pi05_so101_denoise_step_b1_fp16_constrained.engine"


def log(message: str) -> None:
    print(message, flush=True)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple PI0.5 export -> TensorRT convert -> inference validate.")
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--profile", choices=["fp32", "fp16_constrained"], default="fp16_constrained")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument("--opset", type=int, default=19)
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--report", type=Path, default=ARTIFACT_DIR / "simple_pipeline_report.json")
    return parser


def export_onnx(policy, batch, args: argparse.Namespace) -> dict[str, Path]:
    patch_transformers_for_onnx_export()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            "prefix_cache",
            PREFIX_ONNX,
            PI05PrefixCacheONNXWrapper(policy).eval(),
            make_prefix_cache_inputs(policy, batch),
            ["image_0", "image_1", "img_mask_0", "img_mask_1", "tokens", "masks"],
            prefix_cache_tensor_names(),
        ),
        (
            "denoise_step",
            DENOISE_ONNX,
            PI05DenoiseStepONNXWrapper(policy).eval(),
            make_denoise_step_inputs(policy, batch, seed=args.noise_seed),
            denoise_step_input_names(),
            ["v_t"],
        ),
    ]
    outputs = {}
    for name, output, wrapper, inputs, input_names, output_names in jobs:
        outputs[name] = output
        if output.is_file() and not args.force_export:
            log(f"[EXPORT] Reuse {name} ONNX: {output}")
            continue
        log(f"[EXPORT] Exporting {name} ONNX -> {output}")
        for input_name, tensor in zip(input_names, inputs, strict=True):
            log(f"[EXPORT]   {input_name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}")
        t0 = time.perf_counter()
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                inputs,
                str(output),
                input_names=input_names,
                output_names=output_names,
                opset_version=args.opset,
                do_constant_folding=True,
            )
        log(f"[EXPORT] Done {name} in {time.perf_counter() - t0:.2f}s, size={output.stat().st_size} bytes")
    return outputs


def build_engine(onnx_path: Path, engine_path: Path, precision: str, workspace_gb: float, force: bool) -> Path:
    import tensorrt as trt

    onnx_path = onnx_path.expanduser().resolve()
    engine_path = engine_path.expanduser()
    if engine_path.is_file() and not force:
        log(f"[CONVERT] Reuse {precision} engine: {engine_path}")
        return engine_path
    if not onnx_path.is_file():
        raise FileNotFoundError(f"Missing ONNX for conversion: {onnx_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    old_cwd = Path.cwd()
    try:
        os.chdir(onnx_path.parent)
        log(f"[CONVERT] Parsing ONNX: {onnx_path}")
        parsed = parser.parse(onnx_path.read_bytes())
    finally:
        os.chdir(old_cwd)
    if not parsed:
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1024**3)))
    if precision == "fp16_constrained":
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        _configure_fp16_precision_constraints(network)
    elif precision != "fp32":
        raise ValueError(f"Unsupported precision: {precision}")

    log(f"[CONVERT] Building {precision} engine -> {engine_path}")
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build_serialized_network returned None")
    engine_path.write_bytes(bytes(serialized))
    log(f"[CONVERT] Done {precision} in {time.perf_counter() - t0:.2f}s, size={engine_path.stat().st_size} bytes")
    return engine_path


def _set_layer_fp32(layer) -> None:
    import tensorrt as trt

    layer.precision = trt.DataType.FLOAT
    for i in range(layer.num_outputs):
        layer.set_output_type(i, trt.DataType.FLOAT)


def _configure_fp16_precision_constraints(network) -> None:
    import tensorrt as trt

    sensitive_markers = (
        "layernorm",
        "layer_norm",
        "input_layernorm",
        "post_attention_layernorm",
        "softmax",
        "reduce",
        "sqrt",
        "pow",
        "norm",
    )
    sensitive_types = {
        trt.LayerType.REDUCE,
        trt.LayerType.SOFTMAX,
        trt.LayerType.UNARY,
        trt.LayerType.ELEMENTWISE,
    }
    for layer_idx in range(network.num_layers):
        layer = network.get_layer(layer_idx)
        name = layer.name.lower()
        if layer.type in sensitive_types and any(marker in name for marker in sensitive_markers):
            _set_layer_fp32(layer)


def convert_engines(args: argparse.Namespace) -> dict[str, Path]:
    engines = {}
    engines["prefix_fp32"] = build_engine(PREFIX_ONNX, PREFIX_ENGINE, "fp32", args.workspace_gb, args.force_convert)
    if args.profile == "fp32":
        engines["denoise"] = build_engine(DENOISE_ONNX, DENOISE_FP32_ENGINE, "fp32", args.workspace_gb, args.force_convert)
    else:
        engines["denoise"] = build_engine(
            DENOISE_ONNX,
            DENOISE_FP16_CONSTRAINED_ENGINE,
            "fp16_constrained",
            args.workspace_gb,
            args.force_convert,
        )
    return engines


@torch.no_grad()
def validate_inference(policy, batch, args: argparse.Namespace) -> dict:
    inputs = make_export_inputs(policy, batch, seed=args.noise_seed)
    image_0, image_1, img_mask_0, img_mask_1, tokens, masks, noise = inputs
    images = [image_0, image_1]
    img_masks = [img_mask_0, img_mask_1]

    log("[INFER] Running Torch sample_actions baseline")
    t0 = time.perf_counter()
    torch_actions = policy.model.sample_actions(images, img_masks, tokens, masks, noise=noise)
    torch_dt = time.perf_counter() - t0

    if args.profile == "fp32":
        denoise_engine = DENOISE_FP32_ENGINE
    else:
        denoise_engine = DENOISE_FP16_CONSTRAINED_ENGINE
    profile = SimplePI05TRTProfile(
        name=args.profile,
        prefix_engine_path=PREFIX_ENGINE,
        denoise_engine_path=denoise_engine,
    )
    log(f"[INFER] Loading simple TensorRT runtime profile={args.profile}")
    runtime = SimplePI05SplitTRTRuntime(profile)
    log(f"[INFER] Runtime: {runtime.describe()}")
    t0 = time.perf_counter()
    trt_actions = runtime.sample_actions(policy, images, img_masks, tokens, masks, noise)
    trt_dt = time.perf_counter() - t0

    action_dim = policy.config.output_features["action"].shape[0]
    torch_np = torch_actions[:, :, :action_dim].detach().cpu().numpy()
    trt_np = trt_actions[:, :, :action_dim].detach().cpu().numpy()
    stats = tensor_stats(torch_np, trt_np)
    passed = bool(np.allclose(torch_np, trt_np, rtol=2e-2, atol=1e-1))
    log(f"[INFER] Torch latency: {torch_dt * 1000:.2f} ms")
    log(f"[INFER] TRT latency: {trt_dt * 1000:.2f} ms")
    log(f"[INFER] passed_allclose={passed}")
    log(f"[INFER] mean_abs_diff={stats['mean_abs_diff']:.8f}")
    log(f"[INFER] max_abs_diff={stats['max_abs_diff']:.8f}")
    log(f"[INFER] cosine_similarity={stats['cosine_similarity']:.8f}")
    return {
        "passed_allclose": passed,
        "torch_latency_ms": torch_dt * 1000,
        "trt_latency_ms": trt_dt * 1000,
        "action_chunk_stats": stats,
        "runtime": runtime.describe(),
    }


def main() -> None:
    configure_runtime()
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for simple PI0.5 TensorRT pipeline.")

    policy_path = args.policy_path.expanduser().resolve()
    log("[PIPELINE] Simple PI0.5 split TensorRT pipeline")
    log(f"[PIPELINE] policy_path={policy_path}")
    log(f"[PIPELINE] profile={args.profile}")
    log(f"[PIPELINE] report={args.report}")

    log("[PIPELINE] Loading policy and deterministic validation batch")
    t0 = time.perf_counter()
    policy = load_policy(policy_path, device="cuda", model_dtype="float32")
    batch = make_policy_batch(policy_path, task=args.task, seed=args.seed)
    log(f"[PIPELINE] Policy/batch ready in {time.perf_counter() - t0:.2f}s")

    report = {"policy_path": str(policy_path), "profile": args.profile}
    if not args.convert_only and not args.validate_only:
        report["onnx"] = {k: str(v) for k, v in export_onnx(policy, batch, args).items()}
    if not args.export_only and not args.validate_only:
        report["engines"] = {k: str(v) for k, v in convert_engines(args).items()}
    if not args.export_only and not args.convert_only:
        report["inference"] = validate_inference(policy, batch, args)

    write_json(args.report.expanduser(), report)
    log(f"[PIPELINE] Report written: {args.report}")
    if report.get("inference", {}).get("passed_allclose") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
