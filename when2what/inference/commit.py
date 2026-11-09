"""Commit step C (Section 3.4, Supplementary Section S2): E_hat = C(m_hat) => m_tilde.

1. Threshold: local 90th percentile of the positive map values within a sliding window
   (5 s for ECG and ABP, 10 s for PPG), evaluated on a 0.25-s grid and linearly
   interpolated, with a floor (0.15 for ECG and ABP, 0.10 for PPG).
2. Candidates: local maxima of m_hat above the threshold.
3. Refractory spacing: history RR intervals outside [0.25, 2.0] s are discarded and the
   median of the last five remaining ones is the history RR estimate (0.8 s when none
   remains). Candidates are kept greedily, strongest first, at least 0.5 x this estimate
   apart, the spacing being clamped to [0.125, 1.0] s.
4. Committed map m_tilde: a unit-height 10-ms Gaussian at every committed event (Eq. 5).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..data.labels import timing_map

SPACING_RANGE_S = (0.125, 1.0)


def pick_peaks(x: np.ndarray, fs: int, min_dist_s: float, height: float) -> np.ndarray:
    """Local maxima of x above ``height``, kept greedily by amplitude at least min_dist_s apart."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    ge_left = np.r_[True, x[1:] >= x[:-1]]
    ge_right = np.r_[x[:-1] >= x[1:], True]
    cand = np.nonzero(ge_left & ge_right & (x >= height))[0]
    if cand.size == 0:
        return np.zeros(0, dtype=np.int64)
    min_dist = max(1, int(round(min_dist_s * fs)))
    suppressed = np.zeros(n, dtype=bool)
    kept = []
    for c in cand[np.argsort(-x[cand])]:
        if suppressed[c]:
            continue
        kept.append(c)
        suppressed[max(0, c - min_dist):min(n, c + min_dist + 1)] = True
    return np.sort(np.asarray(kept, dtype=np.int64))


def local_threshold(x: np.ndarray, fs: int, window_s: float, quantile: float, floor: float,
                    grid_s: float = 0.25) -> np.ndarray:
    n = x.size
    step, half = int(grid_s * fs), int(window_s * fs / 2)
    centers = np.arange(0, n, step)
    thr = np.empty(centers.size)
    for k, c in enumerate(centers):
        seg = x[max(0, c - half):min(n, c + half)]
        pos = seg[seg > 0]
        thr[k] = max(floor, float(np.percentile(pos, quantile))) if pos.size else floor
    return np.interp(np.arange(n), centers, thr)


def history_rr(history_events: Sequence[int], fs: int, lo_s: float = 0.25, hi_s: float = 2.0,
               n_last: int = 5, default_s: float = 0.8) -> float:
    """Median of the last ``n_last`` history RR intervals inside [lo_s, hi_s] (samples)."""
    d = np.diff(np.sort(np.asarray(history_events, np.float64)))
    d = d[(d >= lo_s * fs) & (d <= hi_s * fs)][-n_last:]
    return float(np.median(d)) if d.size else default_s * fs


def commit_events(m_hat: np.ndarray, history_events: Sequence[int], fs: int = 500, window_s: float = 5.0,
                  quantile: float = 90.0, floor: float = 0.15, spacing_frac: float = 0.5) -> np.ndarray:
    """Committed beat times (sample indices into the horizon) from one raw timing map."""
    x = np.asarray(m_hat, np.float64)
    thr = local_threshold(x, fs, window_s, quantile, floor)
    spacing_s = float(np.clip(spacing_frac * history_rr(history_events, fs) / fs, *SPACING_RANGE_S))
    return pick_peaks(np.where(x >= thr, x, 0.0), fs, spacing_s, 1e-12)


def committed_map(events: Sequence[int], horizon_pts: int, fs: int = 500, sigma_s: float = 0.010) -> np.ndarray:
    return timing_map(np.asarray(events, np.int64), horizon_pts, fs, sigma_s)
