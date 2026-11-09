"""Non-learned beat-time extrapolators (Supplementary Section S2).

Both rules start from causal R peaks (or pulse anchors) detected in the 120-s history.

Simple RR extrapolation: successive history intervals (the last 128 at most) are first
corrected for detection errors when there are at least three: an interval > 1.75 x its
local median whose half is close to that median is split in two, an interval < 0.6 x
the local median is merged with a neighbour, and a too-short last interval marks an
extra last detection, which is dropped. Intervals outside [0.25, 2.0] s are excluded,
and the median of the last five corrected intervals is projected forward from the final
history beat over the 60-s horizon. Intervals are handled in float32.

Guarded cadence rule: before the extrapolation, an outlier filter checks the most recent
history beats against the patient's own beats, guarding against premature ectopic beats
and missed detections (at most two beats are dropped): the last beat is kept only if its
waveform correlates with the median of the preceding beats (r >= 0.8, +-200 ms) and it
follows the previous beat by at least 0.5 x the local median RR.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

SPLIT_FACTOR, MERGE_FACTOR = 1.75, 0.6
RR_GATE_S = (0.25, 2.0)
N_LAST = 5


def _local_median(r: np.ndarray, i: int, w: int = 5) -> float:
    nb = np.concatenate([r[max(0, i - w):i], r[i + 1:min(r.size, i + w + 1)]])
    return float(np.median(nb)) if nb.size else float(r[i])


def correct_rr(rr: np.ndarray, k_split: float = SPLIT_FACTOR, k_merge: float = MERGE_FACTOR) -> Tuple[np.ndarray, bool, float]:
    """Corrected intervals, whether the last event was dropped, and the dropped interval."""
    r = list(np.asarray(rr, np.float64))
    arr = np.asarray(r)
    out: List[float] = []
    dropped_last, dropped = False, 0.0
    i = 0
    while i < len(r):
        m, x = _local_median(arr, i), r[i]
        if x > k_split * m and abs(x / 2 - m) < 0.25 * m:
            out += [x / 2, x / 2]
            i += 1
            continue
        if x < k_merge * m and i == len(r) - 1 and out:
            dropped_last, dropped = True, x
            i += 1
            continue
        if x < k_merge * m:
            prev = out[-1] if out else None
            nxt = r[i + 1] if i + 1 < len(r) else None
            options = [(abs(prev + x - m), "prev")] if prev is not None else []
            if nxt is not None:
                options.append((abs(nxt + x - m), "next"))
            if options:
                if min(options)[1] == "prev":
                    out[-1] = prev + x
                    i += 1
                else:
                    out.append(x + nxt)
                    i += 2
                continue
        out.append(x)
        i += 1
    return np.asarray(out), dropped_last, dropped


def anchor_and_rr(history_events: Sequence[int], history_pts: int, fs: int = 500) -> Tuple[float, float]:
    """Last history beat (horizon coordinates, <= 0) and the RR to repeat (samples)."""
    e = np.sort(np.asarray(history_events, np.float64))
    if e.size < 4:
        return np.nan, np.nan
    rc, dropped_last, dropped = correct_rr(np.diff(e) / fs)
    anchor = e[-1] - history_pts - (dropped * fs if dropped_last else 0.0)
    valid = rc[(rc >= RR_GATE_S[0]) & (rc <= RR_GATE_S[1])]
    if not valid.size:
        return anchor, np.nan
    return anchor, float(np.median(valid[-N_LAST:])) * fs


def extrapolate(anchor: float, rr: float, horizon_pts: int) -> np.ndarray:
    if not (np.isfinite(anchor) and np.isfinite(rr)):
        return np.zeros(0, np.int64)
    k = np.arange(1, int((horizon_pts - anchor) / rr) + 2)
    t = np.round(anchor + rr * k).astype(np.int64)
    return t[(t >= 0) & (t < horizon_pts)]


def simple_rr_extrapolation(history_events: Sequence[int], history_pts: int = 60000,
                            horizon_pts: int = 30000, fs: int = 500, max_intervals: int = 128) -> np.ndarray:
    e = np.sort(np.asarray(history_events, np.float64))
    if e.size < 2:
        return np.zeros(0, np.int64)
    rr = (np.diff(e)[-max_intervals:] / fs).astype(np.float32)
    elapsed = np.float32((history_pts - e[-1]) / fs)
    if rr.size >= 3:
        corrected, dropped_last, dropped = correct_rr(rr)
        rr = corrected[-max_intervals:].astype(np.float32)
        if dropped_last:
            elapsed = np.float32(elapsed + dropped)
    valid = rr[(rr >= RR_GATE_S[0]) & (rr <= RR_GATE_S[1])][-N_LAST:]
    if not valid.size:
        return np.zeros(0, np.int64)
    period = float(np.median(valid)) * fs
    t = -float(elapsed) * fs + period * np.arange(1, int((horizon_pts + elapsed * fs) / period) + 2)
    return np.round(t[(t >= 0) & (t < horizon_pts)]).astype(np.int64)


class _RecentBeats:
    """Template correlation and local median RR of history beats, computed on demand."""

    def __init__(self, x: np.ndarray, events: np.ndarray, fs: int):
        self.x = np.asarray(x, np.float64)
        self.e = np.sort(np.asarray(events, np.int64))
        self.half = int(0.2 * fs)
        self._r, self._m = {}, {}

    def corr(self, j: int) -> float:
        if j not in self._r:
            x, e, h, n = self.x, self.e, self.half, self.x.size
            prev = [q for q in e[:j] if q - h >= 0 and q + h <= n][-30:]
            p = int(e[j])
            if len(prev) < 3 or p - h < 0:
                self._r[j] = np.inf
            else:
                template = np.median(np.stack([x[q - h:q + h] for q in prev]), axis=0)
                seg = x[p - h:min(p + h, n)]
                tp = template[:seg.size]
                self._r[j] = float(np.corrcoef(seg, tp)[0, 1]) if seg.std() > 1e-9 and tp.std() > 1e-9 else -1.0
        return self._r[j]

    def median_rr(self, j: int) -> float:
        if j not in self._m:
            d = np.diff(self.e[:j])[-8:]
            self._m[j] = float(np.median(d)) if d.size >= 3 else np.nan
        return self._m[j]


def guard_history_events(x: np.ndarray, history_events: Sequence[int], fs: int = 500,
                         min_corr: float = 0.8, min_rr_frac: float = 0.5, max_drops: int = 2) -> np.ndarray:
    beats = _RecentBeats(x, np.asarray(history_events), fs)
    keep = list(range(beats.e.size))
    for _ in range(max_drops):
        if len(keep) < 2:
            break
        j = keep[-1]
        m = beats.median_rr(j)
        ok = beats.corr(j) >= min_corr and (not np.isfinite(m) or beats.e[j] - beats.e[keep[-2]] >= min_rr_frac * m)
        if ok:
            break
        keep = keep[:-1]
    return beats.e[keep]


def guarded_cadence(x: np.ndarray, history_events: Sequence[int], history_pts: int = 60000,
                    horizon_pts: int = 30000, fs: int = 500) -> np.ndarray:
    """Guarded cadence rule on one normalised history ``x`` with its detected events (all history
    intervals, float64; at least four history beats)."""
    kept = guard_history_events(x, history_events, fs)
    return extrapolate(*anchor_and_rr(kept, history_pts, fs), horizon_pts)
