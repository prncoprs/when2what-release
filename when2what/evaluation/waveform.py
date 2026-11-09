"""Waveform channel, samplewise MSE and beat correspondence.

Waveform channel: beats are detected on the forecast waveform with the signal's
standard detector (NeuroKit2 default for ECG; maximal-upslope anchors for ABP / PPG)
and scored against the reference events like any event forecast.
MSE: mean squared error over the horizon on the history-normalised scale.
Beat correspondence (Supplementary Section S5): after full-horizon matching (+-50 ms),
each reference beat in the first 10 s is N11 (matched by the timing head and the
waveform), N10 (head only), N01 (waveform only) or N00 (neither). The beat reproduction
(expression) rate is N11 / (N11 + N10), from seed-averaged, patient-summed counts.
Strict expression also requires the matched head and waveform events to lie within
50 ms of each other.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from ..data.labels import detect_r_peaks_safe
from ..data.pulse import upslope_anchors
from .matching import pair_events


def waveform_events(wave: np.ndarray, signal: str = "ecg", fs: int = 500) -> np.ndarray:
    w = np.asarray(wave, np.float64)
    if signal == "ecg":
        return detect_r_peaks_safe(w, fs)
    return upslope_anchors(w, signal, fs)


def mse(wave: np.ndarray, target: np.ndarray, horizon_s: float = 60.0, fs: int = 500) -> float:
    n = int(horizon_s * fs)
    w, y = np.asarray(wave, np.float64)[:n], np.asarray(target, np.float64)[:n]
    return float(((w - y) ** 2).mean())


def beat_correspondence(head_events: Sequence[int], wave_events: Sequence[int], ref: Sequence[int],
                        horizon_s: float = 10.0, fs: int = 500, tol_s: float = 0.05) -> Dict[str, int]:
    """Counts over reference beats in [0, horizon_s): n11, n10, n01, n00 and n11_strict
    (n11 with the head and waveform events within ``tol_s`` of each other)."""
    tol = int(round(tol_s * fs))
    ph, r, pi_head, ri_head = pair_events(head_events, ref, tol)
    pw, _, pi_wave, ri_wave = pair_events(wave_events, ref, tol)
    head_t = np.full(r.size, np.nan)
    head_t[ri_head] = ph[pi_head]
    wave_t = np.full(r.size, np.nan)
    wave_t[ri_wave] = pw[pi_wave]
    head, wave = np.isfinite(head_t), np.isfinite(wave_t)
    early = r < horizon_s * fs
    with np.errstate(invalid="ignore"):
        close = np.abs(head_t - wave_t) <= tol
    return {"n11": int(np.sum(early & head & wave)), "n10": int(np.sum(early & head & ~wave)),
            "n01": int(np.sum(early & ~head & wave)), "n00": int(np.sum(early & ~head & ~wave)),
            "n11_strict": int(np.sum(early & head & wave & close))}


def expression_rate(counts: np.ndarray, boot=None) -> Dict[str, float]:
    """Expression and strict expression rates from per-window, seed-averaged counts.

    ``counts``: (N, 3) array of n11, n10 and n11_strict per window. With a ``PatientBootstrap``
    the result also holds 95 % intervals and the replicates (for paired differences).
    """
    counts = np.asarray(counts, np.float64)
    total = counts[:, 0] + counts[:, 1]
    out = {"expression": counts[:, 0].sum() / total.sum(), "strict_expression": counts[:, 2].sum() / total.sum()}
    if boot is not None:
        per_patient = np.stack([np.bincount(boot.inverse, weights=c, minlength=boot.n_patients)
                                for c in (counts[:, 0], counts[:, 2], total)], axis=1)
        reps = boot.weights @ per_patient
        out["replicates"] = reps[:, 0] / reps[:, 2]
        out["strict_replicates"] = reps[:, 1] / reps[:, 2]
        out["lo"], out["hi"] = (float(v) for v in np.percentile(out["replicates"], [2.5, 97.5]))
        out["strict_lo"], out["strict_hi"] = (float(v) for v in np.percentile(out["strict_replicates"], [2.5, 97.5]))
    return out
