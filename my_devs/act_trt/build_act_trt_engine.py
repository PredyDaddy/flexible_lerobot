from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tensorrt as trt

from my_devs.act_trt.common import load_json, save_json
from my_devs.act_trt.trt_runtime import TensorRTRunner


def _serialize_to_bytes(serialized: Any) -> bytes:
    try:
        return bytes(memoryview(serialized))
    except TypeError:
        with serialized as buffer:
            return bytes(buffer)


def _save_timing_cache(config: trt.IBuilderConfig, path: Path) -> None:
    timing_cache = config.get_timing_cache()
    serialized = timing_cache.serialize()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_serialize_to_bytes(serialized))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build static ACT TensorRT engine from ONNX.")
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--metadata", type=str, default=None)
    parser.add_argument("--engine", type=str, default=None)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument("--opt-level", type=int, default=3)
    parser.add_argument("--timing-cache", type=str, default=None)
    parser.add_argument("--report", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    onnx_path = Path(args.onnx).expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    metadata_path = Path(args.metadata).expanduser().resolve() if args.metadata else onnx_path.parent / "export_metadata.json"
    metadata = load_json(metadata_path)

    engine_path = Path(args.engine).expanduser().resolve() if args.engine else onnx_path.parent / f"act_single_{args.precision}.plan"
    timing_cache_path = (
        Path(args.timing_cache).expanduser().resolve()
        if args.timing_cache
        else onnx_path.parent / f"act_single_{args.precision}.tcache"
    )
    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else onnx_path.parent / f"trt_build_summary_{args.precision}.json"
    )

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(logger) as builder, builder.create_network(network_flags) as network, trt.OnnxParser(network, logger) as parser:
        ok = parser.parse_from_file(str(onnx_path))
        if not ok:
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errors))

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1 << 30)))
        config.builder_optimization_level = int(args.opt_level)
        if hasattr(trt.BuilderFlag, "TF32"):
            config.clear_flag(trt.BuilderFlag.TF32)

        if args.precision == "fp16":
            if not builder.platform_has_fast_fp16:
                raise RuntimeError("platform_has_fast_fp16 is False; cannot build fp16 engine")
            config.set_flag(trt.BuilderFlag.FP16)

        cache_bytes = timing_cache_path.read_bytes() if timing_cache_path.exists() else b""
        timing_cache = config.create_timing_cache(cache_bytes)
        config.set_timing_cache(timing_cache, ignore_mismatch=True)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("build_serialized_network returned None")
        plan_bytes = _serialize_to_bytes(serialized)

        engine_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.write_bytes(plan_bytes)
        _save_timing_cache(config, timing_cache_path)

    runner = TensorRTRunner(engine_path=engine_path, device=args.device)
    summary = {
        "onnx_path": str(onnx_path),
        "metadata_path": str(metadata_path),
        "engine_path": str(engine_path),
        "precision": args.precision,
        "workspace_gb": float(args.workspace_gb),
        "opt_level": int(args.opt_level),
        "tf32_disabled": True,
        "timing_cache": str(timing_cache_path),
        "engine_size_bytes": int(engine_path.stat().st_size),
        "metadata_shapes": metadata.get("shapes", {}),
        "engine_tensors": [tensor.__dict__ for tensor in runner.describe()],
    }
    save_json(report_path, summary)
    print(engine_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
