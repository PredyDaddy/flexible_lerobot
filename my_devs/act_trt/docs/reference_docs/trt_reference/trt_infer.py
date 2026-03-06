#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import ctypes
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorrt as trt
from PIL import Image

from cuda import cuda, cudart  # type: ignore


def _check_cuda_available() -> None:
    err, count = cudart.cudaGetDeviceCount()
    if err != cudart.cudaError_t.cudaSuccess or count is None or int(count) <= 0:
        raise RuntimeError(f"No CUDA device available (cudaGetDeviceCount -> {err}, {count}).")


def _check_cuda_error(err) -> None:
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA driver error: {err}")
        return
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA runtime error: {err}")
        return
    raise RuntimeError(f"Unknown CUDA error type: {type(err)}")


def cuda_call(call):
    err = call[0]
    rest = call[1:]
    _check_cuda_error(err)
    if len(rest) == 1:
        return rest[0]
    return rest


@dataclass
class HostDeviceMem:
    size: int
    dtype: np.dtype

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"size must be > 0, got {self.size}")

        nbytes = int(self.size) * int(self.dtype.itemsize)
        host_ptr = cuda_call(cudart.cudaMallocHost(nbytes))
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(self.dtype))
        self.host = np.ctypeslib.as_array(ctypes.cast(host_ptr, pointer_type), (self.size,))
        self.device = int(cuda_call(cudart.cudaMalloc(nbytes)))
        self.nbytes = nbytes

    def free(self) -> None:
        if getattr(self, "device", 0):
            cuda_call(cudart.cudaFree(self.device))
            self.device = 0
        if getattr(self, "host", None) is not None:
            cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))
            self.host = None


@dataclass
class AllocatedBuffers:
    names: list[str]
    input_names: list[str]
    output_names: list[str]
    buffers: dict[str, HostDeviceMem]
    stream: int


def trt_dtype_to_numpy_dtype(trt_dtype: trt.DataType) -> np.dtype | None:
    try:
        return np.dtype(trt.nptype(trt_dtype))
    except TypeError:
        return None


def _shape_for_allocation(engine: trt.ICudaEngine, name: str, profile_index: int | None) -> tuple[int, ...]:
    static_shape = tuple(engine.get_tensor_shape(name))
    if np.all(np.asarray(static_shape) >= 0):
        return static_shape

    if profile_index is not None:
        try:
            _min_shape, _opt_shape, max_shape = engine.get_tensor_profile_shape(name, profile_index)
            shape = tuple(max_shape)
            if np.all(np.asarray(shape) >= 0):
                return shape
        except Exception:
            pass

    return tuple(1 if dim < 0 else int(dim) for dim in static_shape)


def allocate_io_buffers(engine: trt.ICudaEngine, profile_index: int | None = None) -> AllocatedBuffers:
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    input_names: list[str] = []
    output_names: list[str] = []
    buffers: dict[str, HostDeviceMem] = {}

    for name in names:
        shape = _shape_for_allocation(engine, name, profile_index)
        numel = int(trt.volume(shape))
        trt_dtype = engine.get_tensor_dtype(name)
        np_dtype = trt_dtype_to_numpy_dtype(trt_dtype)

        if np_dtype is None:
            byte_count = int(numel * trt_dtype.itemsize)
            buffers[name] = HostDeviceMem(size=byte_count, dtype=np.dtype(np.uint8))
        else:
            buffers[name] = HostDeviceMem(size=numel, dtype=np_dtype)

        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        elif mode == trt.TensorIOMode.OUTPUT:
            output_names.append(name)

    stream = int(cuda_call(cudart.cudaStreamCreate()))
    return AllocatedBuffers(
        names=names,
        input_names=input_names,
        output_names=output_names,
        buffers=buffers,
        stream=stream,
    )


def free_allocated_buffers(allocated: AllocatedBuffers | None) -> None:
    if allocated is None:
        return
    for mem in allocated.buffers.values():
        mem.free()
    if allocated.stream:
        cuda_call(cudart.cudaStreamDestroy(allocated.stream))


def copy_array_to_pinned_host(target: HostDeviceMem, array: np.ndarray) -> None:
    src = np.ascontiguousarray(array)
    if src.nbytes > target.nbytes:
        raise ValueError(f"Input is larger than buffer capacity: {src.nbytes} > {target.nbytes} bytes")

    if target.dtype == np.uint8:
        target.host[: src.nbytes] = np.frombuffer(src.tobytes(), dtype=np.uint8)
    else:
        casted = src.reshape(-1).astype(target.dtype, copy=False)
        target.host[: casted.size] = casted


def memcpy_h2d_async(device_ptr: int, host_mem: HostDeviceMem, stream: int, nbytes: int) -> None:
    cuda_call(
        cudart.cudaMemcpyAsync(
            device_ptr,
            host_mem.host,
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream,
        )
    )


def memcpy_d2h_async(host_mem: HostDeviceMem, device_ptr: int, stream: int, nbytes: int) -> None:
    cuda_call(
        cudart.cudaMemcpyAsync(
            host_mem.host,
            device_ptr,
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            stream,
        )
    )


def stream_synchronize(stream: int) -> None:
    cuda_call(cudart.cudaStreamSynchronize(stream))


def ensure_buffer_capacity(
    allocated: AllocatedBuffers,
    name: str,
    required_nbytes: int,
    dtype: np.dtype,
) -> HostDeviceMem:
    if required_nbytes <= 0:
        raise ValueError(f"required_nbytes must be > 0, got {required_nbytes}")

    mem = allocated.buffers[name]
    if mem.dtype == dtype and mem.nbytes >= required_nbytes:
        return mem

    mem.free()
    required_size = int(np.ceil(required_nbytes / dtype.itemsize))
    new_mem = HostDeviceMem(size=required_size, dtype=dtype)
    allocated.buffers[name] = new_mem
    return new_mem


def preprocess_image(image_path: Path) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size: {w}x{h} for {image_path}")

    if w < h:
        new_w = 256
        new_h = int(round(h * 256.0 / w))
    else:
        new_h = 256
        new_w = int(round(w * 256.0 / h))

    img = img.resize((new_w, new_h), resample=Image.BILINEAR)
    left = (new_w - 224) // 2
    top = (new_h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    batch = np.ascontiguousarray(arr[None, ...], dtype=np.float32)
    return batch


def compute_topk(row: np.ndarray, k: int) -> list[int]:
    if k <= 0:
        raise ValueError(f"Invalid topk: {k}")
    k = min(k, int(row.shape[0]))
    idx = np.argpartition(-row, k - 1)[:k]
    idx = idx[np.argsort(-row[idx])]
    return [int(i) for i in idx]


def postprocess_logits(logits: np.ndarray, topk: int) -> tuple[int, list[int]]:
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 2:
        raise ValueError(f"Expected logits shape [N,1000], got {logits.shape}")
    row = logits[0]
    top1 = int(np.argmax(row))
    top5 = compute_topk(row, topk)
    return top1, top5


def summarize_latency_ms(latency_ms: list[float]) -> tuple[dict[str, float], float]:
    arr = np.asarray(latency_ms, dtype=np.float64)
    mean = float(arr.mean()) if arr.size else float("nan")
    stats = {
        "mean": mean,
        "std": float(arr.std(ddof=0)) if arr.size else float("nan"),
        "min": float(arr.min()) if arr.size else float("nan"),
        "max": float(arr.max()) if arr.size else float("nan"),
        "p50": float(np.percentile(arr, 50)) if arr.size else float("nan"),
        "p95": float(np.percentile(arr, 95)) if arr.size else float("nan"),
        "p99": float(np.percentile(arr, 99)) if arr.size else float("nan"),
    }
    qps = float(1000.0 / mean) if mean > 0 else 0.0
    return stats, qps


def _infer_once(
    engine: trt.ICudaEngine,
    context: trt.IExecutionContext,
    allocated: AllocatedBuffers,
    batch: np.ndarray,
    output_name: str,
) -> np.ndarray:
    if batch.ndim != 4:
        raise ValueError(f"Expected input shape [N,3,224,224], got {batch.shape}")
    if len(allocated.input_names) != 1:
        raise RuntimeError(f"Expected exactly 1 input, got {allocated.input_names}")

    input_name = allocated.input_names[0]
    if engine.is_shape_inference_io(input_name):
        raise NotImplementedError("Shape-tensor input is not implemented in this simple runtime.")

    input_dtype = trt_dtype_to_numpy_dtype(engine.get_tensor_dtype(input_name))
    if input_dtype is None:
        raise RuntimeError(f"Unsupported input dtype for tensor '{input_name}'")

    input_array = np.ascontiguousarray(batch, dtype=input_dtype)
    if not context.set_input_shape(input_name, tuple(input_array.shape)):
        raise RuntimeError(f"set_input_shape failed for '{input_name}' -> {tuple(input_array.shape)}")

    unresolved = context.infer_shapes()
    if unresolved:
        unresolved_names = [str(name) for name in unresolved]
        raise RuntimeError(f"infer_shapes unresolved tensors: {unresolved_names}")

    input_mem = ensure_buffer_capacity(
        allocated=allocated,
        name=input_name,
        required_nbytes=int(input_array.nbytes),
        dtype=input_dtype,
    )
    copy_array_to_pinned_host(input_mem, input_array)
    if engine.get_tensor_location(input_name) == trt.TensorLocation.DEVICE:
        memcpy_h2d_async(input_mem.device, input_mem, allocated.stream, int(input_array.nbytes))

    # Prepare outputs.
    output_meta: dict[str, tuple[tuple[int, ...], np.dtype | None, int, trt.DataType]] = {}
    for name in allocated.output_names:
        shape = tuple(context.get_tensor_shape(name))
        if any(dim < 0 for dim in shape):
            raise RuntimeError(f"Invalid output shape for {name}: {shape}")

        trt_dtype = engine.get_tensor_dtype(name)
        np_dtype = trt_dtype_to_numpy_dtype(trt_dtype)
        if np_dtype is None:
            host_dtype = np.dtype(np.uint8)
            required_nbytes = int(trt.volume(shape) * trt_dtype.itemsize)
        else:
            host_dtype = np_dtype
            required_nbytes = int(trt.volume(shape) * np_dtype.itemsize)

        ensure_buffer_capacity(
            allocated=allocated,
            name=name,
            required_nbytes=required_nbytes,
            dtype=host_dtype,
        )
        output_meta[name] = (shape, np_dtype, required_nbytes, trt_dtype)

    for name in allocated.names:
        mem = allocated.buffers[name]
        if engine.get_tensor_location(name) == trt.TensorLocation.HOST:
            ptr = int(mem.host.ctypes.data)
        else:
            ptr = int(mem.device)
        if not context.set_tensor_address(name, ptr):
            raise RuntimeError(f"set_tensor_address failed for tensor '{name}'")

    if not context.execute_async_v3(stream_handle=allocated.stream):
        raise RuntimeError("execute_async_v3 failed")

    for name in allocated.output_names:
        _shape, np_dtype, nbytes, _trt_dtype = output_meta[name]
        out_mem = allocated.buffers[name]
        if engine.get_tensor_location(name) != trt.TensorLocation.HOST:
            memcpy_d2h_async(out_mem, out_mem.device, allocated.stream, nbytes)

    stream_synchronize(allocated.stream)

    if output_name not in output_meta:
        raise RuntimeError(f"Output '{output_name}' not found in engine outputs: {allocated.output_names}")

    shape, np_dtype, nbytes, trt_dtype = output_meta[output_name]
    out_mem = allocated.buffers[output_name]
    if np_dtype is None:
        # Byte-level fallback: attempt to interpret bytes using TRT dtype.
        raw = np.array(out_mem.host[:nbytes], copy=True)
        if trt_dtype == trt.DataType.FLOAT:
            data = raw.view(np.float32)
        elif trt_dtype == trt.DataType.HALF:
            data = raw.view(np.float16).astype(np.float32)
        else:
            raise RuntimeError(f"Unsupported output dtype for byte-fallback: {trt_dtype}")
        return np.asarray(data.reshape(shape), dtype=np.float32)

    numel = int(trt.volume(shape))
    view = out_mem.host[:numel].view(np_dtype).reshape(shape)
    return np.asarray(view, dtype=np.float32)


def benchmark_trt(
    engine: trt.ICudaEngine,
    context: trt.IExecutionContext,
    allocated: AllocatedBuffers,
    batch: np.ndarray,
    output_name: str,
    warmup: int,
    repeat: int,
) -> dict:
    for _ in range(int(warmup)):
        _ = _infer_once(engine, context, allocated, batch, output_name=output_name)

    latency_ms: list[float] = []
    for _ in range(int(repeat)):
        t0 = time.perf_counter()
        _ = _infer_once(engine, context, allocated, batch, output_name=output_name)
        t1 = time.perf_counter()
        latency_ms.append((t1 - t0) * 1000.0)

    stats, qps = summarize_latency_ms(latency_ms)
    return {"warmup": int(warmup), "repeat": int(repeat), "latency_ms": stats, "qps": qps}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TensorRT inference (name-based API).")
    p.add_argument("--engine", type=Path, required=True, help="Path to TensorRT engine (.plan).")
    p.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50", "resnet101"])
    p.add_argument("--weights", type=str, default="imagenet", choices=["imagenet", "random"])
    p.add_argument("--precision", type=str, required=True, choices=["fp16", "fp32"])
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--profile-index", type=int, default=0)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=50)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.engine.is_file():
        raise FileNotFoundError(f"Engine file not found: {args.engine}")
    if not args.image.is_file():
        raise FileNotFoundError(f"Image file not found: {args.image}")

    _check_cuda_available()

    script_dir = Path(__file__).resolve().parent
    reports_dir = script_dir / "cache" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    out_path = args.out
    if out_path is None:
        out_path = reports_dir / f"trt_{args.model}_{args.weights}_{args.precision}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    batch = preprocess_image(args.image)

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize engine: {args.engine}")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TensorRT execution context")

    if not (0 <= int(args.profile_index) < int(engine.num_optimization_profiles)):
        raise ValueError(
            f"Invalid profile index {args.profile_index}. Available profiles: {engine.num_optimization_profiles}"
        )

    allocated = allocate_io_buffers(engine, profile_index=int(args.profile_index))
    try:
        if engine.num_optimization_profiles > 1 or int(args.profile_index) != 0:
            if not context.set_optimization_profile_async(int(args.profile_index), allocated.stream):
                raise RuntimeError(f"Failed to set optimization profile {args.profile_index}")
            stream_synchronize(allocated.stream)

        if len(allocated.output_names) < 1:
            raise RuntimeError("Engine has no output tensors")
        output_name = "logits" if "logits" in allocated.output_names else allocated.output_names[0]

        logits = _infer_once(engine, context, allocated, batch, output_name=output_name)
        top1, top5 = postprocess_logits(logits, topk=int(args.topk))

        bench = benchmark_trt(
            engine=engine,
            context=context,
            allocated=allocated,
            batch=batch,
            output_name=output_name,
            warmup=int(args.warmup),
            repeat=int(args.repeat),
        )

        report = {
            "backend": "tensorrt",
            "model": args.model,
            "weights": args.weights,
            "precision": args.precision,
            "image": str(args.image),
            "input_shape": list(batch.shape),
            "output_shape": list(logits.shape),
            "logits": np.asarray(logits, dtype=np.float32).reshape(-1).tolist(),
            "top1": int(top1),
            "top5": [int(i) for i in top5],
            "benchmark": bench,
        }

        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(str(out_path))
    finally:
        free_allocated_buffers(allocated)


if __name__ == "__main__":
    main()

