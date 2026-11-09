"""Event matching and per-bin event F1 (Supplementary Section S1, Eq. S1).

Predicted and reference events of one window are paired greedily one-to-one over the
whole 60-s horizon: predictions are taken in time order, and each is matched to the
closest unmatched reference event if it lies within the tolerance delta (50 ms; 25 and
100 ms in the sensitivity analysis). Matched pairs (TP) and unmatched references (FN)
are assigned to lead-time bins by the reference time, unmatched predictions (FP) by the
predicted time; F1 = 2 TP / (2 TP + FP + FN).
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

LEAD_TIME_BINS_S = (0, 2, 5, 10, 20, 30, 45, 60)
PRIMARY_BINS = 3            # [0, 2), [2, 5), [5, 10): the 0-10 s primary metric


def pair_events(pred: Sequence[int], ref: Sequence[int], tol: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """-> (sorted pred, sorted ref, matched pred indices, matched ref indices)."""
    pred = np.sort(np.asarray(pred, dtype=np.int64))
    ref = np.sort(np.asarray(ref, dtype=np.int64))
    if pred.size == 0 or ref.size == 0:
        return pred, ref, np.zeros(0, np.int64), np.zeros(0, np.int64)
    free = np.ones(ref.size, bool)
    pi, ri = [], []
    for i, p in enumerate(pred):
        dist = np.where(free, np.abs(ref - p), np.iinfo(np.int64).max)
        j = int(np.argmin(dist))
        if dist[j] <= tol:
            free[j] = False
            pi.append(i)
            ri.append(j)
    return pred, ref, np.asarray(pi, np.int64), np.asarray(ri, np.int64)


def bin_counts(pred: Sequence[int], ref: Sequence[int], edges_s: Sequence[float] = LEAD_TIME_BINS_S,
               fs: int = 500, tol_s: float = 0.05) -> np.ndarray:
    """(n_bins, 3) array of TP, FP, FN per lead-time bin."""
    pred, ref, pi, ri = pair_events(pred, ref, int(round(tol_s * fs)))
    edges = np.asarray(edges_s, dtype=np.float64) * fs
    nb = edges.size - 1
    out = np.zeros((nb, 3))
    ref_hit = np.zeros(ref.size, bool)
    ref_hit[ri] = True
    pred_hit = np.zeros(pred.size, bool)
    pred_hit[pi] = True
    for j, b in enumerate(np.searchsorted(edges, ref, side="right") - 1):
        if 0 <= b < nb:
            out[b, 0 if ref_hit[j] else 2] += 1
    for i, b in enumerate(np.searchsorted(edges, pred, side="right") - 1):
        if 0 <= b < nb and not pred_hit[i]:
            out[b, 1] += 1
    return out


def f1_from_counts(counts: np.ndarray) -> np.ndarray:
    """F1 per bin; NaN where the bin has neither predictions nor references (excluded from averages)."""
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    den = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, 2 * tp / den, np.nan)


def event_f1(pred: Sequence[int], ref: Sequence[int], edges_s: Sequence[float] = LEAD_TIME_BINS_S,
             fs: int = 500, tol_s: float = 0.05) -> np.ndarray:
    """Event F1 per lead-time bin (NaN where a bin has neither events nor references)."""
    return f1_from_counts(bin_counts(pred, ref, edges_s, fs, tol_s))
