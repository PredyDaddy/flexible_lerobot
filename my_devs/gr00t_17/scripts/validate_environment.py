#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the local GR00T CUDA environment.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    import flash_attn
    import torchcodec
    from flash_attn import flash_attn_func

    if not flash_attn.__version__.startswith("2.7.4"):
        raise RuntimeError(f"Unexpected flash-attn version: {flash_attn.__version__}")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.empty(1, device=device)
    torch.cuda.reset_peak_memory_stats(device)

    left = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
    right = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
    product = left @ right
    if not torch.isfinite(product).all():
        raise RuntimeError("BF16 CUDA matmul produced non-finite values")

    attention_checks = []
    for head_dim, causal in ((64, False), (128, True)):
        query = torch.randn((1, 128, 4, head_dim), device=device, dtype=torch.bfloat16, requires_grad=True)
        key = torch.randn((1, 128, 4, head_dim), device=device, dtype=torch.bfloat16, requires_grad=True)
        value = torch.randn((1, 128, 4, head_dim), device=device, dtype=torch.bfloat16, requires_grad=True)
        attention = flash_attn_func(query, key, value, dropout_p=0.0, causal=causal)
        attention.float().square().mean().backward()
        backward_finite = query.grad is not None and bool(torch.isfinite(query.grad).all())
        if not backward_finite:
            raise RuntimeError(f"flash-attn backward failed: head_dim={head_dim}, causal={causal}")
        attention_checks.append(
            {
                "head_dim": head_dim,
                "causal": causal,
                "shape": list(attention.shape),
                "backward_finite": backward_finite,
            }
        )

    decoder = torchcodec.decoders.VideoDecoder(
        str(args.video.resolve(strict=True)),
        device="cpu",
        dimension_order="NHWC",
        num_ffmpeg_threads=0,
    )
    indices = np.asarray([0, len(decoder) // 2, len(decoder) - 1])
    frames = decoder.get_frames_at(indices=indices).data.numpy()
    if frames.shape != (3, 480, 640, 3) or frames.dtype != np.uint8:
        raise RuntimeError(f"Unexpected TorchCodec output: shape={frames.shape}, dtype={frames.dtype}")

    report = {
        "status": "passed",
        "python_torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "flash_attn": flash_attn.__version__,
        "flash_attention_checks": attention_checks,
        "torchcodec": torchcodec.__version__,
        "video_frames": len(decoder),
        "decoded_shape": list(frames.shape),
        "decoded_dtype": str(frames.dtype),
        "cuda_peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        "ld_preload": os.environ.get("LD_PRELOAD"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
