#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorrt as trt


def _check_cuda_available() -> None:
    # Avoid TensorRT builder hard-aborting when no CUDA device is present.
    from cuda import cudart  # type: ignore

    err, count = cudart.cudaGetDeviceCount()
    if err != cudart.cudaError_t.cudaSuccess or count is None or int(count) <= 0:
        raise RuntimeError(f"No CUDA device available (cudaGetDeviceCount -> {err}, {count}).")


def _parse_major_minor(version: str) -> tuple[int, int]:
    parts = version.split(".")
    if len(parts) < 2:
        return (0, 0)
    return int(parts[0]), int(parts[1])


def _is_trt_before_10() -> bool:
    major, _minor = _parse_major_minor(trt.__version__)
    return major < 10


def _to_plan_bytes(serialized_plan) -> bytes:
    # TensorRT may return an object exposing the buffer protocol, or a context-manager buffer.
    try:
        return bytes(memoryview(serialized_plan))
    except TypeError:
        with serialized_plan as buffer:
            return bytes(buffer)


def _save_timing_cache(config: trt.IBuilderConfig, timing_cache_path: Path) -> None:
    timing_cache = config.get_timing_cache()
    serialized = timing_cache.serialize()
    try:
        data = memoryview(serialized)
        timing_cache_path.write_bytes(data.tobytes())
    except TypeError:
        with serialized as buffer:
            timing_cache_path.write_bytes(bytes(buffer))


def _default_engine_path(onnx_path: Path, precision: str) -> Path:
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "cache" / "models"
    return models_dir / f"{onnx_path.stem}_{precision}.plan"


def _default_timing_cache_path(onnx_path: Path, precision: str) -> Path:
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "cache" / "models"
    return models_dir / f"{onnx_path.stem}_{precision}.tcache"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a TensorRT engine (.plan) from an ONNX model.")
    p.add_argument("--onnx", type=Path, required=True, help="Path to ONNX file.")
    p.add_argument("--engine", type=Path, default=None, help="Output engine path (.plan).")
    p.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--workspace-bytes", type=int, default=(1 << 30))
    p.add_argument("--opt-level", type=int, default=3)
    p.add_argument("--timing-cache", type=Path, default=None, help="Timing cache path (.tcache).")
    p.add_argument("--disable-timing-cache", action="store_true")
    p.add_argument("--dynamic-batch", action="store_true")
    p.add_argument("--input-name", type=str, default="input")
    p.add_argument("--min-batch", type=int, default=1)
    p.add_argument("--opt-batch", type=int, default=8)
    p.add_argument("--max-batch", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.onnx.is_file():
        raise FileNotFoundError(f"ONNX file not found: {args.onnx}")

    _check_cuda_available()

    engine_path = args.engine if args.engine is not None else _default_engine_path(args.onnx, args.precision)
    timing_cache_path = (
        args.timing_cache if args.timing_cache is not None else _default_timing_cache_path(args.onnx, args.precision)
    )

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")

    builder = trt.Builder(logger)
    network_flags = 0
    if _is_trt_before_10():
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    parser = trt.OnnxParser(network, logger)
    onnx_bytes = args.onnx.read_bytes()
    if not parser.parse(onnx_bytes):
        for i in range(parser.num_errors):
            print(str(parser.get_error(i)))
        raise RuntimeError(f"ONNX parse failed: {args.onnx}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_bytes))
    config.builder_optimization_level = int(args.opt_level)

    if args.precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("[WARN] platform_has_fast_fp16 is False; building without FP16 flag.")

    if args.dynamic_batch:
        profile = builder.create_optimization_profile()
        min_shape = (int(args.min_batch), 3, 224, 224)
        opt_shape = (int(args.opt_batch), 3, 224, 224)
        max_shape = (int(args.max_batch), 3, 224, 224)
        set_shape_result = profile.set_shape(args.input_name, min_shape, opt_shape, max_shape)
        if set_shape_result is False:
            raise RuntimeError(
                f"Failed to set profile shape for {args.input_name}:"
                f" min={min_shape}, opt={opt_shape}, max={max_shape}"
            )
        add_profile_result = config.add_optimization_profile(profile)
        if add_profile_result is False:
            raise RuntimeError("Failed to add optimization profile")
        if isinstance(add_profile_result, int) and add_profile_result < 0:
            raise RuntimeError("Failed to add optimization profile")

    if not args.disable_timing_cache:
        timing_cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_bytes = timing_cache_path.read_bytes() if timing_cache_path.exists() else b""
        timing_cache = config.create_timing_cache(cache_bytes)
        config.set_timing_cache(timing_cache, ignore_mismatch=True)

    serialized_plan = builder.build_serialized_network(network, config)
    if serialized_plan is None:
        raise RuntimeError("build_serialized_network returned None")

    plan_bytes = _to_plan_bytes(serialized_plan)

    if not args.disable_timing_cache:
        _save_timing_cache(config, timing_cache_path)

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(plan_bytes)

    # Validate deserialization path.
    trt.init_libnvinfer_plugins(logger, "")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan_bytes)
    if engine is None:
        raise RuntimeError("Engine deserialization failed after build")

    summary = {
        "onnx": str(args.onnx),
        "engine": str(engine_path),
        "precision": args.precision,
        "workspace_bytes": int(args.workspace_bytes),
        "dynamic_batch": bool(args.dynamic_batch),
        "timing_cache": None if args.disable_timing_cache else str(timing_cache_path),
        "engine_size_bytes": int(len(plan_bytes)),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

