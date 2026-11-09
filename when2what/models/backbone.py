"""Shared sequence backbones.

``HybridBackbone`` (default): bidirectional Mamba blocks with a self-attention
block in every ``attn_every``-th layer and a final LayerNorm. A bidirectional block
runs one Mamba forward and a second Mamba on the time-reversed sequence.

``TCNBackbone``: the causal dilated temporal convolutional network used in the
backbone-swap experiment.

Mamba requires the CUDA kernels of ``mamba-ssm``. ``use_reference_mamba()`` switches
``mamba-ssm`` to its pure PyTorch reference scan so that a model can run on CPU
(slow; intended for tests and small checks).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mamba(d_model: int, d_state: int, d_conv: int, expand: int) -> nn.Module:
    from mamba_ssm import Mamba
    return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)


def use_reference_mamba() -> None:
    """Run Mamba layers with the reference (non-fused) implementation, e.g. on CPU."""
    import mamba_ssm.modules.mamba_simple as ms
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
    ms.causal_conv1d_fn = None
    ms.selective_scan_fn = selective_scan_ref


class BiMambaBlock(nn.Module):
    """x + Mamba_fwd(LN(x)) + flip(Mamba_bwd(flip(LN(x))))."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd = _mamba(d_model, d_state, d_conv, expand)
        self.bwd = _mamba(d_model, d_state, d_conv, expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm(x)
        backward = torch.flip(self.bwd(torch.flip(xn, dims=[1])), dims=[1])
        return x + self.fwd(xn) + backward


class AttnBlock(nn.Module):
    """Pre-norm multi-head self-attention followed by a pre-norm MLP."""

    def __init__(self, d_model: int, n_heads: int = 8, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, mlp_ratio * d_model), nn.GELU(),
                                 nn.Linear(mlp_ratio * d_model, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm1(x)
        x = x + self.attn(xn, xn, xn, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class HybridBackbone(nn.Module):
    def __init__(self, d_model: int, n_layers: int = 4, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, attn_every: int = 2, n_heads: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList([
            AttnBlock(d_model, n_heads) if (i + 1) % attn_every == 0
            else BiMambaBlock(d_model, d_state, d_conv, expand)
            for i in range(n_layers)])
        self.norm_f = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm_f(x)


class CausalConv1d(nn.Module):
    """Dilated Conv1d padded on the left only: output t sees inputs t - (k - 1) d .. t."""

    def __init__(self, channels: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    """Pre-norm residual block with two causal dilated convolutions."""

    def __init__(self, d_model: int, kernel: int, dilation: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.c1 = CausalConv1d(d_model, kernel, dilation)
        self.c2 = CausalConv1d(d_model, kernel, dilation)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x).transpose(1, 2)
        h = self.c2(self.act(self.c1(h)))
        return x + h.transpose(1, 2)


class TCNBackbone(nn.Module):
    def __init__(self, d_model: int, kernel: int = 3, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        self.blocks = nn.ModuleList([TCNBlock(d_model, kernel, d) for d in dilations])
        self.norm_f = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm_f(x)
