#!/usr/bin/env python

"""Compatibility helpers for VLASH's Gemma action-expert port.

VLASH expects a small adaRMS extension in Gemma RMSNorm:

- regular RMSNorm returns ``(hidden_states, None)``;
- adaptive RMSNorm accepts ``cond=...`` and returns ``(hidden_states, gate)``;
- adaptive norms store ``dense.{weight,bias}`` instead of a plain RMS weight.

Some local environments already carry this transformers patch, while stock
transformers wheels do not. Keeping the adapter here makes VLASH training and
inference independent of that environment-level patch.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class VLASHGemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: int | None = None):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.cond_dim = cond_dim

        if cond_dim is None:
            self.weight = nn.Parameter(torch.zeros(dim))
            self.dense = None
        else:
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            nn.init.zeros_(self.dense.weight)
            nn.init.zeros_(self.dense.bias)

    def _norm(self, x: Tensor) -> Tensor:
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps)

    def forward(self, x: Tensor, cond: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        dtype = x.dtype
        normed_inputs = self._norm(x)

        if cond is None or self.dense is None:
            normed_inputs = normed_inputs * (1.0 + self.weight.float())
            return normed_inputs.to(dtype), None

        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected cond dimension {self.cond_dim}, got {cond.shape[-1]}")

        modulation = self.dense(cond)
        if len(x.shape) == 3:
            modulation = modulation.unsqueeze(1)

        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
        normed_inputs = normed_inputs * (1 + scale.to(torch.float32)) + shift.to(torch.float32)
        return normed_inputs.to(dtype), gate.to(dtype)

    def extra_repr(self) -> str:
        if self.dense is None:
            return f"{tuple(self.weight.shape)}, eps={self.eps}"
        return f"({self.dim},), eps={self.eps}, adaptive=True, cond_dim={self.cond_dim}"


def _infer_norm_dim(norm: nn.Module) -> int:
    if hasattr(norm, "dim"):
        return int(norm.dim)
    if hasattr(norm, "weight"):
        return int(norm.weight.shape[0])
    if hasattr(norm, "dense"):
        return int(norm.dense.out_features // 3)
    raise TypeError(f"Cannot infer RMSNorm hidden dimension from {type(norm)}")


def _infer_norm_device(norm: nn.Module) -> torch.device:
    try:
        return next(norm.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _copy_regular_weight(new_norm: VLASHGemmaRMSNorm, old_norm: nn.Module) -> None:
    if new_norm.dense is not None or not hasattr(old_norm, "weight"):
        return
    with torch.no_grad():
        new_norm.weight.copy_(old_norm.weight.detach().to(new_norm.weight.device, dtype=new_norm.weight.dtype))


def _copy_adaptive_dense(new_norm: VLASHGemmaRMSNorm, old_norm: nn.Module) -> None:
    if new_norm.dense is None or not hasattr(old_norm, "dense") or old_norm.dense is None:
        return
    with torch.no_grad():
        new_norm.dense.weight.copy_(
            old_norm.dense.weight.detach().to(new_norm.dense.weight.device, dtype=new_norm.dense.weight.dtype)
        )
        new_norm.dense.bias.copy_(
            old_norm.dense.bias.detach().to(new_norm.dense.bias.device, dtype=new_norm.dense.bias.dtype)
        )


def make_vlash_rms_norm(old_norm: nn.Module, cond_dim: int | None) -> VLASHGemmaRMSNorm:
    dim = _infer_norm_dim(old_norm)
    eps = float(getattr(old_norm, "eps", 1e-6))
    device = _infer_norm_device(old_norm)

    new_norm = VLASHGemmaRMSNorm(dim, eps=eps, cond_dim=cond_dim).to(device=device)
    _copy_regular_weight(new_norm, old_norm)
    _copy_adaptive_dense(new_norm, old_norm)
    return new_norm


def ensure_gemma_rms_norm_compat(gemma_model: nn.Module, cond_dim: int | None = None) -> None:
    """Replace Gemma RMSNorm modules with VLASH-compatible RMSNorm modules."""

    for layer in gemma_model.layers:
        layer.input_layernorm = make_vlash_rms_norm(layer.input_layernorm, cond_dim)
        layer.post_attention_layernorm = make_vlash_rms_norm(layer.post_attention_layernorm, cond_dim)

    if hasattr(gemma_model, "norm"):
        gemma_model.norm = make_vlash_rms_norm(gemma_model.norm, cond_dim)
