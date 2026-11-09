"""Training losses (Eqs. 4, 6 and 7 of the paper).

Eq. (6):  L = L_time(m_hat, m) + L_wave(y_hat, y) + lambda_h L_wave(x_hat, x),  lambda_h = 0.3

Eq. (4):  L_time = mean_t (1 + w m) (m_hat - m)^2 + lambda_F (1 - F_soft(m_hat, m)),
          F_soft = 2 sum_t m_hat m / (sum_t m_hat + sum_t m),  w = 50, lambda_F = 0.5
          (m_hat is clipped to [0, 1] inside F_soft).
Eq. (7):  L_wave = MSE + MSE of first differences + MSE of second differences.
The Split-MSE variant replaces the first term of L_wave by 5 x MSE inside |y| <= 1.6
(history z-score units) + 1 x MSE outside.

Model variants (Table 3 of the paper):
  when2what   timing loss; G receives the ground-truth timing map (teacher forcing)
  direct_mt   timing loss; G receives an all-zero map
  direct      no timing loss; G receives an all-zero map
"""
from __future__ import annotations

import contextlib
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from ..config import Config


def _difference_terms(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    d1p, d1t = pred[..., 1:] - pred[..., :-1], target[..., 1:] - target[..., :-1]
    d2p, d2t = d1p[..., 1:] - d1p[..., :-1], d1t[..., 1:] - d1t[..., :-1]
    return F.mse_loss(d1p, d1t), F.mse_loss(d2p, d2t)


def waveform_loss(pred: torch.Tensor, target: torch.Tensor, diff_weight: float = 1.0) -> torch.Tensor:
    d1, d2 = _difference_terms(pred, target)
    return F.mse_loss(pred, target) + diff_weight * (d1 + d2)


def split_mse(pred: torch.Tensor, target: torch.Tensor, bound: float = 1.6,
              w_inner: float = 5.0, w_outer: float = 1.0) -> torch.Tensor:
    """Band membership from the target; per-band means over the whole batch."""
    inner = (target >= -bound) & (target <= bound)
    se = (pred - target) ** 2
    l_in = se[inner].mean() if inner.any() else se.sum() * 0.0
    l_out = se[~inner].mean() if (~inner).any() else se.sum() * 0.0
    return w_inner * l_in + w_outer * l_out


def split_waveform_loss(pred: torch.Tensor, target: torch.Tensor, cfg: Config) -> torch.Tensor:
    d1, d2 = _difference_terms(pred, target)
    return split_mse(pred, target, cfg.split_bound, cfg.split_w_inner, cfg.split_w_outer) + cfg.diff_weight * (d1 + d2)


def soft_f1(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft overlap F1 of Eq. (4), one value per window."""
    p = pred.clamp(0.0, 1.0)
    return 2.0 * (p * true).sum(dim=1) / (p.sum(dim=1) + true.sum(dim=1) + eps)


def timing_loss(pred: torch.Tensor, true: torch.Tensor, pos_weight: float = 50.0,
                soft_f1_weight: float = 0.5) -> torch.Tensor:
    """L_time of Eq. (4): peak-weighted MSE plus the soft overlap F1 loss."""
    weighted_mse = ((1.0 + pos_weight * true) * (pred - true) ** 2).mean()
    return weighted_mse + soft_f1_weight * (1.0 - soft_f1(pred, true)).mean()


def autocast(cfg: Config, device: torch.device):
    if cfg.bf16 and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def compute_losses(model, batch: Dict[str, torch.Tensor], cfg: Config,
                   device: torch.device) -> Tuple[torch.Tensor, Dict[str, float]]:
    x = batch["x"].to(device, non_blocking=True)
    y = batch["y"].to(device, non_blocking=True)
    true_map = batch["timing"].to(device, non_blocking=True)
    with autocast(cfg, device):
        out = model(x, timing_map=true_map if cfg.conditioned else None)
        wave = out["wave"].float()
        l_wave = split_waveform_loss(wave, y, cfg) if cfg.wave_loss == "split_mse" else \
            waveform_loss(wave, y, cfg.diff_weight)
        l_hist = waveform_loss(model.reconstruct_history(x).float(), x, cfg.diff_weight)
        if cfg.uses_timing_loss:
            l_time = timing_loss(out["timing"].float(), true_map, cfg.peak_pos_weight, cfg.soft_f1_weight)
        else:
            l_time = torch.zeros((), device=device)
    total = cfg.w_timing * l_time + cfg.w_wave * l_wave + cfg.w_hist * l_hist
    return total, {"total": float(total.detach()), "timing": float(l_time.detach()),
                   "wave": float(l_wave.detach()), "hist": float(l_hist.detach())}
