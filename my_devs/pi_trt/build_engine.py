#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Failed to locate repo root from: {start}")


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

try:
    # When `my_devs/` is importable as a package (e.g. `python -m my_devs.pi_trt.build_engine`).
    from my_devs.groot_trt.trt_utils import import_tensorrt  # type: ignore
except Exception:  # pragma: no cover - fallback for `python my_devs/pi_trt/build_engine.py`.
    from my_devs.groot_trt.trt_utils import import_tensorrt  # type: ignore


PI05_STATIC_INPUT_SHAPES: dict[str, dict[str, tuple[int, ...]]] = {
    "vision_encoder_fp16.onnx": {
        "image_tensor": (2, 3, 224, 224),
    },
    "prefix_cache_fp16.onnx": {
        "prefix_embs": (1, 712, 2048),
        "prefix_attention_mask_4d": (1, 1, 712, 712),
        "prefix_position_ids": (1, 712),
    },
    "denoise_step_fp16.onnx": {
        "x_t": (1, 50, 32),
        "timestep": (1,),
        "prefix_pad_masks": (1, 712),
        "kv_cache": (18, 2, 1, 1, 712, 256),
    },
}


@dataclass
class BuildConfig:
    workspace_gb: float = 8.0
    strongly_typed: bool = True
    verbose: bool = False


def _parse_onnx(parser: Any, onnx_path: Path) -> None:
    # PI0.5 ONNX exports use external data sidecars, so parsing by file path is safer than parsing raw bytes.
    if hasattr(parser, "parse_from_file"):
        ok = bool(parser.parse_from_file(str(onnx_path)))
    else:  # pragma: no cover - older TensorRT fallback.
        ok = bool(parser.parse(onnx_path.read_bytes()))
    if ok:
        return

    errors = [parser.get_error(i) for i in range(parser.num_errors)]
    joined = "\n".join(str(err) for err in errors)
    raise RuntimeError(f"Failed to parse ONNX: {onnx_path}\n{joined}")


def _resolve_dynamic_shape(onnx_name: str, input_name: str, template_shape: tuple[int, ...]) -> tuple[int, ...]:
    if all(int(dim) >= 0 for dim in template_shape):
        return tuple(int(dim) for dim in template_shape)

    shape_map = PI05_STATIC_INPUT_SHAPES.get(onnx_name, {})
    if input_name not in shape_map:
        raise ValueError(
            "Encountered a dynamic TensorRT input without a known static fallback shape.\n"
            f"  - onnx={onnx_name}\n"
            f"  - input={input_name}\n"
            f"  - template={template_shape}\n"
        )
    return shape_map[input_name]


def _collect_io_metadata(network: Any, kind: str) -> list[dict[str, Any]]:
    if kind == "input":
        count = network.num_inputs
        getter = network.get_input
    elif kind == "output":
        count = network.num_outputs
        getter = network.get_output
    else:  # pragma: no cover - internal misuse guard.
        raise ValueError(f"Unsupported io kind: {kind}")

    metadata: list[dict[str, Any]] = []
    for index in range(count):
        tensor = getter(index)
        metadata.append(
            {
                "name": tensor.name,
                "dtype": str(tensor.dtype),
                "shape": [int(dim) for dim in tensor.shape],
            }
        )
    return metadata


def _maybe_add_profile(trt: Any, builder: Any, network: Any, config: Any, onnx_path: Path) -> list[dict[str, Any]]:
    dynamic_inputs: list[dict[str, Any]] = []
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        template_shape = tuple(int(dim) for dim in tensor.shape)
        if all(dim >= 0 for dim in template_shape):
            continue
        resolved_shape = _resolve_dynamic_shape(onnx_path.name, tensor.name, template_shape)
        dynamic_inputs.append(
            {
                "name": tensor.name,
                "template_shape": list(template_shape),
                "resolved_shape": list(resolved_shape),
            }
        )

    if not dynamic_inputs:
        return dynamic_inputs

    profile = builder.create_optimization_profile()
    for item in dynamic_inputs:
        resolved_shape = tuple(int(dim) for dim in item["resolved_shape"])
        profile.set_shape(item["name"], resolved_shape, resolved_shape, resolved_shape)
    config.add_optimization_profile(profile)
    return dynamic_inputs


def _build_engine(
    trt: Any,
    onnx_path: Path,
    engine_path: Path,
    build_config: BuildConfig,
) -> dict[str, Any]:
    logger = trt.Logger(trt.Logger.INFO if build_config.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if build_config.strongly_typed:
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)

    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    _parse_onnx(parser, onnx_path)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(build_config.workspace_gb * 1024**3))
    dynamic_inputs = _maybe_add_profile(trt, builder, network, config, onnx_path)

    start = time.time()
    serialized = builder.build_serialized_network(network, config)
    end = time.time()
    if serialized is None:
        raise RuntimeError(f"TensorRT build failed (serialized network is None): {onnx_path}")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)

    return {
        "onnx": onnx_path.as_posix(),
        "engine": engine_path.as_posix(),
        "build_seconds": end - start,
        "workspace_gb": build_config.workspace_gb,
        "dynamic_inputs": dynamic_inputs,
        "inputs": _collect_io_metadata(network, "input"),
        "outputs": _collect_io_metadata(network, "output"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build TensorRT engines (Python API) for PI0.5 ONNX exports.")
    parser.add_argument("--onnx-dir", required=True, help="Directory containing the PI0.5 ONNX exports.")
    parser.add_argument(
        "--engine-out-dir",
        default=None,
        help="Engine output directory. Default: sibling folder next to --onnx-dir (pi05_engine_api_trt1013).",
    )
    parser.add_argument(
        "--tensorrt-py-dir",
        default=None,
        help="Optional repo-local TensorRT install dir (the folder that contains `tensorrt/` and `tensorrt_libs/`).",
    )
    parser.add_argument("--workspace-gb", type=float, default=8.0)
    parser.add_argument("--verbose", action="store_true", help="Enable TensorRT INFO logs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    onnx_dir = Path(args.onnx_dir).expanduser()
    if not onnx_dir.is_dir():
        raise FileNotFoundError(f"--onnx-dir does not exist: {onnx_dir}")

    engine_dir = (
        Path(args.engine_out_dir).expanduser() if args.engine_out_dir else onnx_dir.parent / "pi05_engine_api_trt1013"
    )
    engine_dir.mkdir(parents=True, exist_ok=True)

    planned: list[tuple[Path, Path]] = []
    for onnx_name in PI05_STATIC_INPUT_SHAPES:
        onnx_path = onnx_dir / onnx_name
        if onnx_path.is_file():
            planned.append((onnx_path, engine_dir / f"{onnx_path.stem}.engine"))

    if not planned:
        expected = "\n".join(f"  - {onnx_dir / name}" for name in PI05_STATIC_INPUT_SHAPES)
        raise FileNotFoundError("No PI0.5 ONNX files found to build.\nExpected any of:\n" + expected)

    trt = import_tensorrt(args.tensorrt_py_dir)
    build_config = BuildConfig(
        workspace_gb=float(args.workspace_gb),
        strongly_typed=True,
        verbose=bool(args.verbose),
    )

    report: dict[str, Any] = {
        "tensorrt_version": trt.__version__,
        "onnx_dir": onnx_dir.as_posix(),
        "engine_dir": engine_dir.as_posix(),
        "build_config": asdict(build_config),
        "missing_onnx": [
            (onnx_dir / name).as_posix() for name in PI05_STATIC_INPUT_SHAPES if not (onnx_dir / name).is_file()
        ],
        "engines": [],
    }

    for onnx_path, engine_path in planned:
        print(f"[BUILD] {onnx_path.name} -> {engine_path.name}")
        engine_info = _build_engine(trt, onnx_path, engine_path, build_config)
        report["engines"].append(engine_info)
        print(f"[OK] Built {engine_path.name} ({engine_info['build_seconds']:.2f}s)")

    json_out = engine_dir / "build_report.json"
    json_out.write_text(json.dumps(report, indent=2))
    print(f"[OK] Build report saved to: {json_out}")


if __name__ == "__main__":
    main()
