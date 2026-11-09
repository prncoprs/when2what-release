"""Predict -> Commit -> Generate at inference time.

``Forecaster`` takes normalised histories (B, history_pts) and returns, per window, the
raw timing map m_hat (P), the committed events E_hat = C(m_hat) and the forecast
waveform y_hat (G). When2What feeds the committed map to G; Direct and Direct-MT feed
an all-zero map, as in training. ``generate_from_events`` renders G at any given
event times (timing-source swaps and perturbation analyses).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..config import Config
from ..data.labels import detect_r_peaks_safe
from ..data.pulse import upslope_anchors
from .commit import commit_events, committed_map


def history_events(x: np.ndarray, cfg: Config) -> np.ndarray:
    """Beat events of one normalised history, detected from the history alone."""
    x = np.asarray(x, np.float64)
    if cfg.signal == "ecg":
        return detect_r_peaks_safe(x, cfg.fs)
    return upslope_anchors(x, cfg.signal, cfg.fs)


class Forecaster:
    def __init__(self, model: torch.nn.Module, cfg: Config, device: torch.device,
                 float16_maps: bool = True):
        """``float16_maps``: round m_hat to float16 before the commit step; the paper's
        pipeline stored timing maps in float16 and committed events from the stored maps."""
        self.model = model.eval()
        self.cfg = cfg
        self.device = device
        self.float16_maps = float16_maps

    def _commit(self, m_hat: np.ndarray, hist_events: Sequence[int]) -> np.ndarray:
        cfg = self.cfg
        if self.float16_maps:
            m_hat = m_hat.astype(np.float16).astype(np.float32)
        return commit_events(m_hat, hist_events, cfg.fs, cfg.commit_window_s, cfg.commit_quantile,
                             cfg.commit_floor, cfg.commit_spacing_frac)

    def _maps(self, events: Sequence[np.ndarray]) -> torch.Tensor:
        cfg = self.cfg
        maps = np.stack([committed_map(e, cfg.horizon_pts, cfg.fs, cfg.peak_sigma_s) for e in events])
        return torch.as_tensor(maps, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, hist_events: Optional[List[np.ndarray]] = None) -> Dict[str, object]:
        x = x.to(self.device, dtype=torch.float32)
        x_np = x.cpu().numpy()
        if hist_events is None:
            hist_events = [history_events(row, self.cfg) for row in x_np]
        tokens = self.model.tokens(x)
        m_hat = self.model.predict_timing(x, tokens)
        m_np = m_hat.float().cpu().numpy()
        events = [self._commit(m, h) for m, h in zip(m_np, hist_events)]
        if self.cfg.conditioned:
            cond = self._maps(events)
        else:
            cond = torch.zeros_like(m_hat, dtype=torch.float32)
        wave = self.model.generate(x, cond, tokens).float().cpu().numpy()
        return {"timing_map": m_np, "events": events, "wave": wave, "history_events": hist_events}

    @torch.no_grad()
    def generate_from_events(self, x: torch.Tensor, events: Sequence[np.ndarray]) -> np.ndarray:
        x = x.to(self.device, dtype=torch.float32)
        return self.model.generate(x, self._maps(events)).float().cpu().numpy()

    @torch.no_grad()
    def generate_from_map(self, x: torch.Tensor, timing: torch.Tensor) -> np.ndarray:
        """G conditioned on an arbitrary map, e.g. the uncommitted m_hat (ablation of C)."""
        x = x.to(self.device, dtype=torch.float32)
        return self.model.generate(x, timing.to(self.device, dtype=torch.float32)).float().cpu().numpy()
