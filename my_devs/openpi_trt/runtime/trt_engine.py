#!/usr/bin/env python

"""Minimal TensorRT runtime backed by torch CUDA tensors."""

from __future__ import annotations

from pathlib import Path

import tensorrt as trt
import torch


def torch_dtype_from_trt(dtype):
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.bool: torch.bool,
        trt.uint8: torch.uint8,
    }
    if dtype not in mapping:
        raise TypeError(f"Unsupported TensorRT dtype: {dtype}")
    return mapping[dtype]


class TorchTensorRTEngine:
    """Run TensorRT engines using caller-owned torch CUDA input/output tensors."""

    def __init__(self, engine_path: str | Path):
        self.engine_path = Path(engine_path)
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, "")

        runtime = trt.Runtime(self.logger)
        with self.engine_path.open("rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create TensorRT execution context: {self.engine_path}")

        self.input_names: list[str] = []
        self.output_names: list[str] = []
        self.tensor_dtypes: dict[str, torch.dtype] = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.tensor_dtypes[name] = torch_dtype_from_trt(self.engine.get_tensor_dtype(name))
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def set_input_shapes(self, inputs: dict[str, torch.Tensor]) -> None:
        for name in self.input_names:
            tensor = inputs[name]
            self.context.set_input_shape(name, tuple(tensor.shape))

    def __call__(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        self.set_input_shapes(inputs)
        stream = torch.cuda.current_stream()
        references: list[torch.Tensor] = []

        for name in self.input_names:
            tensor = inputs[name]
            expected_dtype = self.tensor_dtypes[name]
            if tensor.dtype != expected_dtype:
                raise TypeError(f"{name}: expected {expected_dtype}, got {tensor.dtype}")
            if not tensor.is_cuda:
                raise ValueError(f"{name}: expected CUDA tensor, got {tensor.device}")
            tensor = tensor.contiguous()
            self.context.set_tensor_address(name, tensor.data_ptr())
            references.append(tensor)

        outputs: dict[str, torch.Tensor] = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self.tensor_dtypes[name]
            output = torch.empty(shape, dtype=dtype, device=references[0].device)
            self.context.set_tensor_address(name, output.data_ptr())
            outputs[name] = output
            references.append(output)

        ok = self.context.execute_async_v3(stream.cuda_stream)
        if not ok:
            raise RuntimeError(f"TensorRT execute_async_v3 failed for {self.engine_path}")
        stream.synchronize()
        return outputs

    def describe(self) -> dict:
        tensors = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            tensors[name] = {
                "shape": list(self.engine.get_tensor_shape(name)),
                "dtype": str(self.tensor_dtypes[name]),
                "mode": str(self.engine.get_tensor_mode(name)),
            }
        return {
            "engine_path": str(self.engine_path),
            "input_names": self.input_names,
            "output_names": self.output_names,
            "tensors": tensors,
        }

