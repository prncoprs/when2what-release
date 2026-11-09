"""Chance level, event skill and patient-cluster bootstrap (Eq. 1; Supplementary Eqs. S2-S4).

Chance F1_c (Eq. S2): expected F1 of a constant-rate pulse train at the window's true
mean beat interval with a uniformly random starting phase. The expectation over the
phase is estimated with 50 uniform phase draws (seed 7000 + window position).
Event skill per bin (Eq. S3): S_b = (mean_w F_wb - mean_w C_wb) / (1 - mean_w C_wb),
where F is first averaged over training seeds within each window. The near-term event
skill (Eq. S4) is the unweighted mean of S_b over [0, 2), [2, 5) and [5, 10) s.
Confidence intervals: patients are resampled with replacement (2,000 replicates,
seed 0) and the skill, or a paired difference of skills on the same windows, is
recomputed on each replicate; the 95 % interval is the 2.5-97.5 percentile range.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .matching import LEAD_TIME_BINS_S, PRIMARY_BINS, event_f1


def chance_f1(ref: Sequence[int], edges_s: Sequence[float] = LEAD_TIME_BINS_S, n_draws: int = 50,
              rng: np.random.Generator = None, fs: int = 500, horizon_pts: int = 30000,
              tol_s: float = 0.05) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    ref = np.sort(np.asarray(ref, np.float64))
    rr = float(np.mean(np.diff(ref))) if ref.size >= 2 else np.nan
    nb = len(edges_s) - 1
    if not np.isfinite(rr) or rr < 1:
        return np.full(nb, np.nan)
    draws = []
    for _ in range(n_draws):
        phase = rng.uniform(0, rr)
        draws.append(event_f1(np.arange(phase, horizon_pts, rr).astype(np.int64), ref, edges_s, fs, tol_s))
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.stack(draws), axis=0)


def chance_table(refs: Sequence[Sequence[int]], edges_s: Sequence[float] = LEAD_TIME_BINS_S,
                 seed_base: int = 7000, fs: int = 500, horizon_pts: int = 30000, tol_s: float = 0.05) -> np.ndarray:
    """(N, n_bins) chance F1, window k drawn with seed ``seed_base + k``."""
    return np.stack([chance_f1(r, edges_s, 50, np.random.default_rng(seed_base + k), fs, horizon_pts, tol_s)
                     for k, r in enumerate(refs)])


def seed_average(per_seed: List[np.ndarray]) -> np.ndarray:
    """Average per-window F1 over training seeds (NaN-aware)."""
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.stack(per_seed), axis=0)


def bin_skill(F: np.ndarray, C: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        f, c = np.nanmean(F, 0), np.nanmean(C, 0)
    return (f - c) / (1 - c)


def primary_skill(F: np.ndarray, C: np.ndarray, n_bins: int = PRIMARY_BINS) -> float:
    return float(np.mean(bin_skill(F[:, :n_bins], C[:, :n_bins])))


class PatientBootstrap:
    """Replicate weights: replicate r counts how often each patient was drawn."""

    def __init__(self, patients: Sequence, n_boot: int = 2000, seed: int = 0):
        uniq, self.inverse = np.unique(np.asarray(patients), return_inverse=True)
        self.n_patients = uniq.size
        rng = np.random.default_rng(seed)
        self.weights = np.stack([np.bincount(rng.integers(0, self.n_patients, self.n_patients),
                                             minlength=self.n_patients) for _ in range(n_boot)]).astype(float)

    def sums(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Per-patient sums and counts of the finite entries of X (N, bins)."""
        ok = np.isfinite(X)
        S = np.zeros((self.n_patients, X.shape[1]))
        N = np.zeros((self.n_patients, X.shape[1]))
        for j in range(X.shape[1]):
            S[:, j] = np.bincount(self.inverse, weights=np.where(ok[:, j], X[:, j], 0.0), minlength=self.n_patients)
            N[:, j] = np.bincount(self.inverse, weights=ok[:, j].astype(float), minlength=self.n_patients)
        return S, N

    def skill_replicates(self, F: np.ndarray, C: np.ndarray) -> np.ndarray:
        """(B,) primary skill on every replicate; F and C restricted to the primary bins."""
        (sf, nf), (sc, nc) = self.sums(F), self.sums(C)
        f = (self.weights @ sf) / (self.weights @ nf)
        c = (self.weights @ sc) / (self.weights @ nc)
        return np.mean((f - c) / (1 - c), axis=1)


def interval(replicates: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    lo, hi = np.nanpercentile(replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def skill_with_ci(F: np.ndarray, C: np.ndarray, boot: PatientBootstrap, n_bins: int = PRIMARY_BINS) -> Dict[str, float]:
    F, C = F[:, :n_bins], C[:, :n_bins]
    reps = boot.skill_replicates(F, C)
    lo, hi = interval(reps)
    return {"skill": primary_skill(F, C, n_bins), "lo": lo, "hi": hi, "replicates": reps}


def skill_difference(Fa: np.ndarray, Fb: np.ndarray, C: np.ndarray, boot: PatientBootstrap,
                     n_bins: int = PRIMARY_BINS) -> Dict[str, float]:
    """Paired difference of primary skills (a - b) on the same windows and replicates."""
    a = skill_with_ci(Fa, C, boot, n_bins)
    b = skill_with_ci(Fb, C, boot, n_bins)
    lo, hi = interval(a["replicates"] - b["replicates"])
    return {"difference": a["skill"] - b["skill"], "lo": lo, "hi": hi, "a": a["skill"], "b": b["skill"]}
