#!/usr/bin/env python

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


DEFAULT_TENSORRT_PY_DIR = Path("/data/cqy_workspace/third_party/tensorrt_10_13_0_35")


def _prepend_env_path(var_name: str, path: Path) -> None:
    value = os.environ.get(var_name, "")
    parts = [part for part in value.split(":") if part]
    path_str = path.as_posix()
    if path_str in parts:
        return
    parts.insert(0, path_str)
    os.environ[var_name] = ":".join(parts)


def import_tensorrt(trt_py_dir: str | Path | None = None) -> Any:
    """Import TensorRT with a repo-local fallback install.

    Why this exists:
    - We develop in `lerobot_flex`, but the root filesystem is usually tight on space.
    - `pip install tensorrt==...` into the conda env may fail.
    - A workable workaround is: `pip --target /data/... tensorrt==...` then add that dir to `PYTHONPATH`
      and `LD_LIBRARY_PATH` at runtime.

    This helper makes scripts under `my_devs/` runnable without needing users to remember env exports.
    """

    try:
        import tensorrt as trt  # type: ignore

        return trt
    except Exception:
        pass

    py_dir = Path(trt_py_dir) if trt_py_dir is not None else Path(os.environ.get("TENSORRT_PY_DIR", ""))
    if not py_dir:
        py_dir = DEFAULT_TENSORRT_PY_DIR
    py_dir = py_dir.expanduser().resolve()

    if py_dir.is_dir() and py_dir.as_posix() not in sys.path:
        sys.path.insert(0, py_dir.as_posix())

    libs_dir = py_dir / "tensorrt_libs"
    if libs_dir.is_dir():
        _prepend_env_path("LD_LIBRARY_PATH", libs_dir)

    try:
        import tensorrt as trt  # type: ignore

        return trt
    except Exception as exc:
        raise RuntimeError(
            "Failed to import TensorRT.\n"
            "Tried repo-local install dir:\n"
            f"  - {py_dir}\n\n"
            "Fix options:\n"
            "1) Install into a shared /data location:\n"
            "   `python -m pip install --target /data/<you>/third_party/tensorrt_10_13_0_35 tensorrt==10.13.0.35`\n"
            "2) Then run with env:\n"
            "   `env TENSORRT_PY_DIR=/data/<you>/third_party/tensorrt_10_13_0_35 ...`\n"
            "3) Or export paths explicitly:\n"
            "   `PYTHONPATH=<dir>:$PYTHONPATH LD_LIBRARY_PATH=<dir>/tensorrt_libs:$LD_LIBRARY_PATH`\n"
        ) from exc


def torch_dtype_from_trt(trt: Any, dtype: Any) -> torch.dtype:
    if dtype == trt.DataType.HALF:
        return torch.float16
    if dtype == trt.DataType.FLOAT:
        return torch.float32
    if dtype == trt.DataType.INT8:
        return torch.int8
    if dtype == trt.DataType.INT32:
        return torch.int32
    if dtype == trt.DataType.INT64:
        return torch.int64
    if dtype == trt.DataType.BOOL:
        return torch.bool
    raise TypeError(f"Unsupported TensorRT dtype: {dtype!r}")


@dataclass
class TrtSession:
    trt: Any
    engine_path: Path
    engine: Any
    context: Any
    input_names: list[str]
    output_names: list[str]

    @classmethod
    def load(cls, engine_path: str | Path, trt_py_dir: str | Path | None = None) -> "TrtSession":
        trt = import_tensorrt(trt_py_dir)
        engine_path = Path(engine_path).expanduser()
        if not engine_path.is_file():
            raise FileNotFoundError(engine_path)
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine_bytes = engine_path.read_bytes()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError(f"Failed to create execution context: {engine_path}")

        input_names: list[str] = []
        output_names: list[str] = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)

        return cls(
            trt=trt,
            engine_path=engine_path,
            engine=engine,
            context=context,
            input_names=input_names,
            output_names=output_names,
        )

    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Validate inputs.
        for name in self.input_names:
            if name not in inputs:
                raise KeyError(f"Missing required input tensor: {name} (engine={self.engine_path.name})")
        for name, tensor in inputs.items():
            if tensor.device.type != "cuda":
                raise ValueError(f"TensorRT inputs must be CUDA tensors. Got {name} on {tensor.device}.")
            if not tensor.is_contiguous():
                inputs[name] = tensor.contiguous()

        # Dynamic shapes must be set before allocating outputs.
        for name in self.input_names:
            shape = tuple(int(v) for v in inputs[name].shape)
            if not self.context.set_input_shape(name, shape):
                raise RuntimeError(f"Failed to set input shape: {name}={shape} (engine={self.engine_path.name})")

        outputs: dict[str, torch.Tensor] = {}
        for name in self.output_names:
            shape = tuple(int(v) for v in self.context.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = torch_dtype_from_trt(self.trt, dtype)
            outputs[name] = torch.empty(shape, device="cuda", dtype=torch_dtype)

        # Bind tensor addresses.
        for name in self.input_names:
            self.context.set_tensor_address(name, int(inputs[name].data_ptr()))
        for name in self.output_names:
            self.context.set_tensor_address(name, int(outputs[name].data_ptr()))

        stream = torch.cuda.current_stream().cuda_stream
        if not self.context.execute_async_v3(stream):
            raise RuntimeError(f"TensorRT execute_async_v3 failed: {self.engine_path}")
        return outputs

