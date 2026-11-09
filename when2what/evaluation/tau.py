"""Beat-time predictability time scale tau (Fig. 1a; Supplementary Section S1, Table S1).

Event skill S is computed in thirty 2-s lead-time bins over 0-60 s (chance seeds
9000 + window position) and a exp(-t / tau) + c is fitted at the bin centres by bounded
least squares. The 95 % interval of tau comes from refitting on 1,000 shared
patient-cluster bootstrap replicates.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

TWO_SECOND_BINS_S = tuple(range(0, 61, 2))


def fit_tau(t: np.ndarray, skill: np.ndarray):
    """(a, tau, c) of a exp(-t / tau) + c; NaNs when the fit fails."""
    from scipy.optimize import curve_fit
    ok = np.isfinite(skill)
    if ok.sum() < 4:
        return (np.nan,) * 3
    try:
        p, _ = curve_fit(lambda t_, a, tau, c: a * np.exp(-t_ / tau) + c, t[ok], skill[ok],
                         p0=(max(skill[ok][0], 0.1), 5.0, 0.01), bounds=([0, 0.1, 0], [2, 1000, 1]),
                         method="trf", maxfev=5000)
        return tuple(float(v) for v in p)
    except Exception:  # noqa: BLE001
        return (np.nan,) * 3


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    return (w @ np.where(ok, x, 0.0)) / np.maximum(w @ ok.astype(np.float64), 1e-12)


def tau_with_ci(F: np.ndarray, C: np.ndarray, patients: Sequence, n_boot: int = 1000, seed: int = 0,
                edges_s: Sequence[float] = TWO_SECOND_BINS_S) -> Dict[str, float]:
    """F, C: (N, 30) per-window event F1 and chance F1 in 2-s bins."""
    ok = np.isfinite(F) & np.isfinite(C)
    F, C = np.where(ok, F, np.nan), np.where(ok, C, np.nan)
    with np.errstate(invalid="ignore"):
        skill = (np.nanmean(F, 0) - np.nanmean(C, 0)) / (1 - np.nanmean(C, 0))
    uniq, inverse = np.unique(np.asarray(patients), return_inverse=True)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(uniq.size, np.full(uniq.size, 1 / uniq.size), size=n_boot)[:, inverse].astype(np.float64)
    bf, bc = _weighted_mean(F, w), _weighted_mean(C, w)
    boot_skill = (bf - bc) / (1 - bc)
    edges = np.asarray(edges_s, dtype=float)
    t = edges[:-1] + (edges[1] - edges[0]) / 2
    a, tau, c = fit_tau(t, skill)
    taus = np.array([fit_tau(t, r)[1] for r in boot_skill])
    return {"tau_s": tau, "tau_lo": float(np.nanpercentile(taus, 2.5)), "tau_hi": float(np.nanpercentile(taus, 97.5)),
            "a": a, "c": c, "skill_by_bin": skill}
