"""Static historical beat template (Supplementary Section S2).

The template is the pointwise median of the last 40 complete history beats, each cut
over [-250, +450] ms around its R peak. A 60-s forecast places the template at every
given beat time; overlapping segments of closely spaced beats are joined at their
temporal midpoint, and samples between separated beats take the template's isoelectric
level (median of its first and last 30 ms).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

PRE_S, POST_S = 0.25, 0.45
N_BEATS = 40
MIN_BEATS = 5


def history_template(x: np.ndarray, history_events: Sequence[int], fs: int = 500,
                     n_beats: int = N_BEATS) -> Optional[np.ndarray]:
    """Median beat of the last ``n_beats`` complete beats, or None with fewer than five."""
    pre, post = int(PRE_S * fs), int(POST_S * fs)
    e = np.sort(np.asarray(history_events, np.int64))
    complete = e[(e - pre >= 0) & (e + post < x.size)]
    if complete.size < MIN_BEATS:
        return None
    beats = np.stack([np.asarray(x[c - pre:c + post + 1], np.float64) for c in complete])
    return np.median(beats[-n_beats:], axis=0)


def isoelectric_level(template: np.ndarray, fs: int = 500) -> float:
    k = int(0.03 * fs)
    return float(np.median(np.concatenate([template[:k], template[-k:]])))


def place_template(events: Sequence[int], template: np.ndarray, horizon_pts: int, fs: int = 500) -> np.ndarray:
    pre, post = int(PRE_S * fs), int(POST_S * fs)
    out = np.full(horizon_pts, isoelectric_level(template, fs))
    times = np.sort(np.asarray(events, np.int64), kind="stable")
    times = times[(times >= 0) & (times < horizon_pts)]
    for i, t in enumerate(times):
        lo, hi = t - pre, t + post
        if i > 0 and times[i - 1] + post >= lo:
            lo = int(np.floor((times[i - 1] + post + t - pre) / 2.0)) + 1
        if i + 1 < len(times) and times[i + 1] - pre <= hi:
            hi = int(np.floor((t + post + times[i + 1] - pre) / 2.0))
        s0, s1 = max(0, lo), min(horizon_pts - 1, hi)
        if s1 >= s0:
            out[s0:s1 + 1] = template[pre - (t - s0):pre + (s1 - t) + 1]
    return out


def template_forecast(x: np.ndarray, history_events: Sequence[int], events: Sequence[int],
                      horizon_pts: int = 30000, fs: int = 500) -> np.ndarray:
    """History-template forecast at ``events``; a flat zero line without a usable template."""
    template = history_template(x, history_events, fs)
    if template is None:
        return np.zeros(horizon_pts, np.float32)
    return place_template(events, template, horizon_pts, fs).astype(np.float32)
