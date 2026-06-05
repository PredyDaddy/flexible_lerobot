#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
from pathlib import Path

import tensorrt as trt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a TensorRT engine from a LeRobot PI0.5 ONNX file.")
    parser.add_argument("--onnx-path", type=Path, required=True)
    parser.add_argument("--engine-path", type=Path, required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16"], required=True)
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    return parser


def _set_layer_fp32(layer: trt.ILayer) -> None:
    layer.precision = trt.DataType.FLOAT
    for i in range(layer.num_outputs):
        layer.set_output_type(i, trt.DataType.FLOAT)


def _configure_fp16_precision_constraints(network: trt.INetworkDefinition) -> None:
    """Keep numerically sensitive denoise_step subgraphs in FP32.

    The denoise_step ONNX graph contains many decomposed LayerNorm blocks and
    softmax/reduction nodes. A pure FP16 build can drift badly on this graph,
    so we keep the sensitive normalization/attention reduction layers in FP32
    while still allowing the rest of the graph to use FP16 tactics.
    """

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


def main() -> None:
    args = build_parser().parse_args()
    onnx_path = args.onnx_path.expanduser().resolve()
    engine_path = args.engine_path.expanduser()
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    print(f"[INFO] Parsing ONNX: {onnx_path}")
    old_cwd = Path.cwd()
    try:
        # TensorRT resolves ONNX external data files relative to the process
        # working directory when parsing from bytes.
        os.chdir(onnx_path.parent)
        with onnx_path.open("rb") as f:
            parsed = parser.parse(f.read())
    finally:
        os.chdir(old_cwd)
    if not parsed:
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("Failed to parse ONNX:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    workspace_bytes = int(args.workspace_gb * (1024**3))
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)

    if args.precision == "fp16":
        if not builder.platform_has_fast_fp16:
            print("[WARN] Platform does not report fast FP16 support; building FP16 engine anyway.")
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    print("[INFO] Network inputs:")
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        print(f"[INFO]   {tensor.name}: shape={tensor.shape} dtype={tensor.dtype}")
    print("[INFO] Network outputs:")
    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        print(f"[INFO]   {tensor.name}: shape={tensor.shape} dtype={tensor.dtype}")

    if args.precision == "fp16":
        print("[INFO] Applying FP32 precision constraints to sensitive normalization/attention layers.")
        _configure_fp16_precision_constraints(network)

    print(f"[INFO] Building TensorRT {args.precision.upper()} engine: {engine_path}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build_serialized_network returned None")

    engine_path.write_bytes(bytes(serialized))
    print(f"[INFO] Engine written: {engine_path} ({engine_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
