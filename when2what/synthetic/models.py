"""Small convolutional backbone shared by the synthetic model types.

``SynthNet`` is a non-causal dilated residual 1-D CNN over the joint axis history + future (3000 samples at
100 Hz; the future part of the ECG channel is zero). Input channels: ECG, the known respiratory signal, the
timing map (future part only), an is-future indicator and the normalised time into the future. A Conv1d stem
(kernel 8, stride 4) reduces the axis to 25 Hz; eight residual blocks with dilations 1..128 give every output
the whole history; two transposed convolutions return to 100 Hz, the full-resolution timing map is concatenated,
and two Conv1d heads output the waveform and timing-map logits on the future part.

Model types (same network and parameter count):
  direct      zero map; loss MSE(wave, y)
  direct_mt   zero map; loss MSE(wave, y) + MapLoss(timing logits, true map)
  when2what   pass 1 with a zero map -> MapLoss; pass 2 with the TRUE map (teacher forcing) -> MSE(wave, y).
              Inference: pass 2 with the committed map (events picked from sigmoid(pass-1 logits), drawn as
              Gaussians) or, for the soft-map variant, with the raw predicted map.
MapLoss = mean((1 + 10 m) (sigmoid(logits) - m)^2).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .generator import N_FUT, N_HIST

C_WIDTH = 48
DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128)
MAP_POS_WEIGHT = 10.0
W_MAP = 1.0
MODEL_TYPES = ("direct", "direct_mt", "when2what")


class Block(nn.Module):
    def __init__(self, c: int, d: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, c)
        self.conv = nn.Conv1d(c, c, 5, dilation=d, padding=2 * d)
        self.proj = nn.Conv1d(c, c, 1)

    def forward(self, x):
        return x + self.proj(F.gelu(self.conv(self.norm(x))))


class SynthNet(nn.Module):
    def __init__(self, c: int = C_WIDTH):
        super().__init__()
        self.stem = nn.Conv1d(5, c, 8, stride=4, padding=2)
        self.blocks = nn.ModuleList([Block(c, d) for d in DILATIONS])
        self.norm = nn.GroupNorm(1, c)
        self.up1 = nn.ConvTranspose1d(c, 32, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose1d(32, 32, 4, stride=2, padding=1)
        self.fuse = nn.Conv1d(33, 32, 9, padding=4)
        self.head_wave = nn.Conv1d(32, 1, 9, padding=4)
        self.head_map = nn.Conv1d(32, 1, 9, padding=4)
        n = N_HIST + N_FUT
        is_future = torch.zeros(1, 1, n)
        is_future[:, :, N_HIST:] = 1.0
        pos = torch.zeros(1, 1, n)
        pos[:, :, N_HIST:] = torch.arange(N_FUT, dtype=torch.float32) / N_FUT
        self.register_buffer("const", torch.cat([is_future, pos], 1))

    def forward(self, hist, resp_h, resp_f, map_f):
        """hist, resp_h: (B, 2000); resp_f, map_f: (B, 1000) -> waveform and map logits on the future (B, 1000)."""
        b = hist.shape[0]
        ecg = torch.cat([hist, torch.zeros_like(resp_f)], 1)
        resp = torch.cat([resp_h, resp_f], 1)
        mp = torch.cat([torch.zeros_like(hist), map_f], 1)
        x = torch.cat([torch.stack([ecg, resp, mp], 1), self.const.expand(b, -1, -1)], 1)
        h = F.gelu(self.stem(x))
        for blk in self.blocks:
            h = blk(h)
        h = F.gelu(self.norm(h))
        h = F.gelu(self.up2(F.gelu(self.up1(h))))
        h = F.gelu(self.fuse(torch.cat([h, mp[:, None, :]], 1)))
        return self.head_wave(h)[:, 0, N_HIST:], self.head_map(h)[:, 0, N_HIST:]


def count(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def map_loss(logits, target):
    return ((1.0 + MAP_POS_WEIGHT * target) * (torch.sigmoid(logits) - target) ** 2).mean()


def losses(net: SynthNet, kind: str, batch):
    """-> (total, parts). batch = (hist, resp_h, resp_f, y, true_map)."""
    hist, resp_h, resp_f, y, tmap = batch
    zero = torch.zeros_like(tmap)
    if kind == "direct":
        wave, _ = net(hist, resp_h, resp_f, zero)
        lw = F.mse_loss(wave, y)
        return lw, {"wave": lw.item(), "map": 0.0}
    if kind == "direct_mt":
        wave, logits = net(hist, resp_h, resp_f, zero)
        lw, lm = F.mse_loss(wave, y), map_loss(logits, tmap)
        return lw + W_MAP * lm, {"wave": lw.item(), "map": lm.item()}
    if kind == "when2what":
        _, logits = net(hist, resp_h, resp_f, zero)
        lm = map_loss(logits, tmap)
        wave, _ = net(hist, resp_h, resp_f, tmap)
        lw = F.mse_loss(wave, y)
        return lw + W_MAP * lm, {"wave": lw.item(), "map": lm.item()}
    raise ValueError(kind)


@torch.no_grad()
def predict(net: SynthNet, kind: str, hist, resp_h, resp_f, committed_map=None):
    """-> {'wave'} (+ 'map' for direct_mt / when2what; + 'wave_committed' when a committed map is given)."""
    zero = torch.zeros_like(resp_f)
    if kind == "direct":
        return {"wave": net(hist, resp_h, resp_f, zero)[0]}
    if kind == "direct_mt":
        wave, logits = net(hist, resp_h, resp_f, zero)
        return {"wave": wave, "map": torch.sigmoid(logits)}
    if kind == "when2what":
        m = torch.sigmoid(net(hist, resp_h, resp_f, zero)[1])
        out = {"wave": net(hist, resp_h, resp_f, m)[0], "map": m}
        if committed_map is not None:
            out["wave_committed"] = net(hist, resp_h, resp_f, committed_map)[0]
        return out
    raise ValueError(kind)
