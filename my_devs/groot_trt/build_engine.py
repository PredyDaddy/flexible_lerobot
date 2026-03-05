#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    # When `my_devs/` is importable as a package (e.g. `python -m my_devs.groot_trt.build_engine`).
    from my_devs.groot_trt.trt_utils import import_tensorrt  # type: ignore
except Exception:  # pragma: no cover - fallback for `python my_devs/groot_trt/build_engine.py`.
    from trt_utils import import_tensorrt  # type: ignore


@dataclass
class BuildProfile:
    max_batch: int = 2
    vit_opt_batch: int = 2
    opt_batch: int = 1
    min_seq_len: int = 80
    opt_seq_len: int = 568
    max_seq_len: int = 600
    workspace_gb: float = 8.0
    strongly_typed: bool = True
    verbose: bool = False


def _set_shape_for_profile(
    input_name: str,
    template_shape: tuple[int, ...],
    batch: int,
    seq_len: int,
) -> tuple[int, ...]:
    # TensorRT parses dynamic axes as -1 dims.
    shape = list(int(v) for v in template_shape)
    for idx, dim in enumerate(shape):
        if dim != -1:
            continue
        if idx == 0:
            shape[idx] = int(batch)
            continue
        if input_name in {"inputs_embeds", "attention_mask", "backbone_features", "vl_embs"} and idx == 1:
            shape[idx] = int(seq_len)
            continue
        raise ValueError(
            f"Unhandled dynamic dim for input={input_name!r}, template_shape={template_shape}, idx={idx}."
        )
    return tuple(shape)


def _build_engine(
    trt: Any,
    onnx_path: Path,
    engine_path: Path,
    profile: BuildProfile,
) -> dict[str, Any]:
    logger = trt.Logger(trt.Logger.INFO if profile.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if profile.strongly_typed:
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [parser.get_error(i) for i in range(parser.num_errors)]
        joined = "\n".join(str(err) for err in errors)
        raise RuntimeError(f"Failed to parse ONNX: {onnx_path}\n{joined}")

    config = builder.create_builder_config()
    workspace = int(profile.workspace_gb * 1024**3)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)

    opt_profile = builder.create_optimization_profile()
    # Heuristic: ViT uses views as batch, so we want opt batch to cover 2-view. Other modules typically run b=1.
    if "vit_" in onnx_path.name:
        opt_batch = min(profile.vit_opt_batch, profile.max_batch)
    else:
        opt_batch = min(profile.opt_batch, profile.max_batch)

    for i in range(network.num_inputs):
        inp = network.get_input(i)
        name = inp.name
        template = tuple(int(v) for v in inp.shape)
        min_shape = _set_shape_for_profile(name, template, batch=1, seq_len=profile.min_seq_len)
        opt_shape = _set_shape_for_profile(name, template, batch=opt_batch, seq_len=profile.opt_seq_len)
        max_shape = _set_shape_for_profile(name, template, batch=profile.max_batch, seq_len=profile.max_seq_len)
        try:
            # NOTE: TensorRT 10 returns `None` here (raises on invalid shapes).
            opt_profile.set_shape(name, min_shape, opt_shape, max_shape)
        except Exception as exc:
            raise RuntimeError(
                "Failed to set optimization profile shape.\n"
                f"  - onnx={onnx_path.name}\n"
                f"  - input={name}\n"
                f"  - min={min_shape}\n"
                f"  - opt={opt_shape}\n"
                f"  - max={max_shape}\n"
                f"  - template={template}\n"
            ) from exc

    config.add_optimization_profile(opt_profile)

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
        "inputs": [
            {
                "name": network.get_input(i).name,
                "dtype": str(network.get_input(i).dtype),
                "shape": list(int(v) for v in network.get_input(i).shape),
            }
            for i in range(network.num_inputs)
        ],
        "outputs": [
            {
                "name": network.get_output(i).name,
                "dtype": str(network.get_output(i).dtype),
                "shape": list(int(v) for v in network.get_output(i).shape),
            }
            for i in range(network.num_outputs)
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build TensorRT engines (Python API) for LeRobot GROOT ONNX exports.")
    parser.add_argument(
        "--onnx-dir",
        required=True,
        help="Directory containing `eagle2/` and `action_head/` ONNX exports.",
    )
    parser.add_argument(
        "--engine-out-dir",
        default=None,
        help="Engine output directory. Default: sibling folder next to --onnx-dir (gr00t_engine_api_trt1013).",
    )
    parser.add_argument(
        "--tensorrt-py-dir",
        default=None,
        help="Optional repo-local TensorRT install dir (the folder that contains `tensorrt/` and `tensorrt_libs/`).",
    )
    parser.add_argument("--max-batch", type=int, default=2)
    parser.add_argument("--vit-opt-batch", type=int, default=2)
    parser.add_argument("--opt-batch", type=int, default=1)
    parser.add_argument("--min-seq-len", type=int, default=80)
    parser.add_argument("--opt-seq-len", type=int, default=568)
    parser.add_argument("--max-seq-len", type=int, default=600)
    parser.add_argument("--workspace-gb", type=float, default=8.0)
    parser.add_argument("--verbose", action="store_true", help="Enable TensorRT INFO logs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    onnx_dir = Path(args.onnx_dir).expanduser()
    if not onnx_dir.is_dir():
        raise FileNotFoundError(f"--onnx-dir does not exist: {onnx_dir}")

    engine_dir = Path(args.engine_out_dir).expanduser() if args.engine_out_dir else onnx_dir.parent / "gr00t_engine_api_trt1013"
    engine_dir.mkdir(parents=True, exist_ok=True)

    trt = import_tensorrt(args.tensorrt_py_dir)
    build_profile = BuildProfile(
        max_batch=int(args.max_batch),
        vit_opt_batch=int(args.vit_opt_batch),
        opt_batch=int(args.opt_batch),
        min_seq_len=int(args.min_seq_len),
        opt_seq_len=int(args.opt_seq_len),
        max_seq_len=int(args.max_seq_len),
        workspace_gb=float(args.workspace_gb),
        strongly_typed=True,
        verbose=bool(args.verbose),
    )

    planned: list[tuple[Path, Path]] = []
    # Backbone.
    vit_onnx = onnx_dir / "eagle2" / "vit_fp16.onnx"
    llm_onnx = onnx_dir / "eagle2" / "llm_fp16.onnx"
    if vit_onnx.is_file():
        planned.append((vit_onnx, engine_dir / "vit_fp16.engine"))
    if llm_onnx.is_file():
        planned.append((llm_onnx, engine_dir / "llm_fp16.engine"))

    # Action head.
    planned.extend(
        [
            (onnx_dir / "action_head" / "vlln_vl_self_attention.onnx", engine_dir / "vlln_vl_self_attention.engine"),
            (onnx_dir / "action_head" / "state_encoder.onnx", engine_dir / "state_encoder.engine"),
            (onnx_dir / "action_head" / "action_encoder.onnx", engine_dir / "action_encoder.engine"),
            (onnx_dir / "action_head" / "DiT_fp16.onnx", engine_dir / "DiT_fp16.engine"),
            (onnx_dir / "action_head" / "action_decoder.onnx", engine_dir / "action_decoder.engine"),
        ]
    )

    missing = [onnx for onnx, _ in planned if not onnx.is_file()]
    if missing:
        raise FileNotFoundError("Missing required ONNX files:\n" + "\n".join(f"  - {p}" for p in missing))

    report: dict[str, Any] = {
        "tensorrt_version": trt.__version__,
        "onnx_dir": onnx_dir.as_posix(),
        "engine_dir": engine_dir.as_posix(),
        "build_profile": asdict(build_profile),
        "engines": [],
    }
    for onnx_path, engine_path in planned:
        print(f"[BUILD] {onnx_path.name} -> {engine_path.name}")
        engine_info = _build_engine(trt, onnx_path, engine_path, build_profile)
        report["engines"].append(engine_info)
        print(f"[OK] Built {engine_path.name} ({engine_info['build_seconds']:.2f}s)")

    json_out = engine_dir / "build_report.json"
    json_out.write_text(json.dumps(report, indent=2))
    print(f"[OK] Build report saved to: {json_out}")


if __name__ == "__main__":
    main()
