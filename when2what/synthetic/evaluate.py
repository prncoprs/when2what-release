"""Per-record metrics of the synthetic experiment and their aggregation.

Events of a forecast waveform are local maxima >= max(0.15, 0.4 x the waveform maximum) with a 250-ms refractory
(greedy, highest first); timing-map events use a fixed threshold of 0.3. Events are matched to the true future
R times within +-50 ms. Per record: beat-detection F1 (``f1``), squared error to the realised future (``mse_real``;
Supplementary Tables S23 and S24), squared distance to the average of possible futures (``excess_cm``; Fig. 5b),
retained R-peak height (``peak_amp_ratio``: forecast at the matched event / truth at the true event), and per-beat
errors. Seeds are averaged per record; 95 % intervals come from a record bootstrap (B = 1000).

Model names: ``direct``, ``direct_mt``, ``when2what`` (G fed the unthresholded soft map m_hat, Table S24) and
``when2what_committed`` (G fed the committed map m_tilde, Table S23 and Fig. 5).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import generator as G

TOL = 5                       # +-50 ms at 100 Hz
REFRACTORY = 25               # 250 ms
PRE, POST = 25, 40            # beat window -250 .. +400 ms
K_HIT = 8
METRICS = ["mse_real", "excess_cm", "excess_diff", "f1", "f1_native", "beat_rmse_abs", "beat_rmse_aligned",
           "peak_amp_ratio", "peak_amp_dev", "coverage", "amp_ratio_global", "n_pred", "n_true"] + \
          [f"hit_k{k}" for k in range(1, K_HIT + 1)]


def pick(x: np.ndarray, thr: float, refractory: int = REFRACTORY) -> np.ndarray:
    x = np.asarray(x, np.float64)
    if x.size < 3:
        return np.zeros(0, np.int64)
    c = np.flatnonzero((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]) & (x[1:-1] >= thr)) + 1
    keep: List[int] = []
    for i in c[np.argsort(-x[c], kind="stable")]:
        if all(abs(i - j) > refractory for j in keep):
            keep.append(int(i))
    return np.array(sorted(keep), np.int64)


def waveform_events(wave: np.ndarray) -> np.ndarray:
    return pick(wave, max(0.15, 0.4 * float(np.max(wave))))


def map_events(m: np.ndarray) -> np.ndarray:
    return pick(m, 0.3)


def pair_events(pred: np.ndarray, true: np.ndarray, tol: int = TOL) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy one-to-one matching: predictions in time order, each to the closest unmatched true event
    within ``tol`` -> (pred indices, true indices)."""
    pred, true = np.sort(np.asarray(pred, np.int64)), np.sort(np.asarray(true, np.int64))
    if pred.size == 0 or true.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    free = np.ones(true.size, bool)
    pi, ti = [], []
    for i, p in enumerate(pred):
        dist = np.where(free, np.abs(true - p), np.iinfo(np.int64).max)
        j = int(np.argmin(dist))
        if dist[j] <= tol:
            free[j] = False
            pi.append(i)
            ti.append(j)
    return np.asarray(pi, np.int64), np.asarray(ti, np.int64)


def _f1(tp: int, n_pred: int, n_true: int) -> float:
    return 2 * tp / (n_pred + n_true) if n_pred + n_true > 0 else float("nan")


def record_metrics(wave, y, cond_mean, risk: float, true_ev, map_=None) -> Dict[str, float]:
    wave, y, cm = (np.asarray(a, np.float64) for a in (wave, y, cond_mean))
    out: Dict[str, float] = {"mse_real": float(((wave - y) ** 2).mean()), "excess_cm": float(((wave - cm) ** 2).mean())}
    out["excess_diff"] = out["mse_real"] - float(risk)
    ev = waveform_events(wave)
    pi, ti = pair_events(ev, true_ev)
    out["f1"] = _f1(pi.size, ev.size, true_ev.size)
    out["n_pred"], out["n_true"] = float(ev.size), float(true_ev.size)
    hit = np.zeros(true_ev.size, bool)
    hit[ti] = True
    for k in range(1, K_HIT + 1):
        out[f"hit_k{k}"] = float(hit[k - 1]) if true_ev.size >= k else float("nan")
    if map_ is not None:
        evn = map_events(map_)
        out["f1_native"] = _f1(pair_events(evn, true_ev)[0].size, evn.size, true_ev.size)
    else:
        out["f1_native"] = float("nan")
    ra, rl, pa = [], [], []
    for i, j in zip(pi, ti):
        p, t = int(ev[i]), int(true_ev[j])
        if t - PRE < 0 or t + POST > G.N_FUT or p - PRE < 0 or p + POST > G.N_FUT:
            continue
        yt = y[t - PRE:t + POST]
        ra.append(np.sqrt(((wave[t - PRE:t + POST] - yt) ** 2).mean()))
        rl.append(np.sqrt(((wave[p - PRE:p + POST] - yt) ** 2).mean()))
        pa.append(wave[p] / y[t] if abs(y[t]) > 1e-6 else np.nan)
    out["beat_rmse_abs"] = float(np.mean(ra)) if ra else float("nan")
    out["beat_rmse_aligned"] = float(np.mean(rl)) if rl else float("nan")
    out["peak_amp_ratio"] = float(np.nanmean(pa)) if pa else float("nan")
    out["peak_amp_dev"] = abs(1.0 - out["peak_amp_ratio"]) if np.isfinite(out["peak_amp_ratio"]) else float("nan")
    out["coverage"] = pi.size / true_ev.size if true_ev.size else float("nan")
    sy = y.std()
    out["amp_ratio_global"] = float(wave.std() / sy) if sy > 0 else float("nan")
    return out


def load_cell(out_root: Path, cell: str):
    d = dict(np.load(Path(out_root) / "data" / f"{cell}_test.npz"))
    mc = dict(np.load(Path(out_root) / "mc" / f"{cell}_test_mc.npz"))
    return d, mc, [G.future_event_samples(r) for r in d["beat_t"]]


def metrics_table(preds: Dict[str, np.ndarray], d, mc, true_ev) -> np.ndarray:
    """(n_records, len(METRICS)) metric array of one set of predictions."""
    rows = np.full((d["fut"].shape[0], len(METRICS)), np.nan)
    for i in range(rows.shape[0]):
        m = record_metrics(preds["wave"][i], d["fut"][i], mc["cond_mean"][i], float(mc["cond_risk"][i]), true_ev[i],
                           map_=preds["map"][i] if "map" in preds else None)
        rows[i] = [m[k] for k in METRICS]
    return rows


def evaluate_cell(out_root: Path, cell: str, seeds=(0, 1, 2), models=("direct", "direct_mt", "when2what")) -> Dict[str, Dict]:
    """{name: {seed: per-record metrics}}; When2What yields 'when2what' (soft map) and 'when2what_committed'."""
    d, mc, true_ev = load_cell(out_root, cell)
    out: Dict[str, Dict] = {}
    for model in models:
        for s in seeds:
            run = Path(out_root) / "runs" / f"{cell}_{model}_s{s}"
            if not (run / "done.json").exists():
                continue
            p = dict(np.load(run / "preds.npz"))
            out.setdefault(model, {})[s] = metrics_table(p, d, mc, true_ev)
            if "wave_committed" in p:
                committed = {"wave": p["wave_committed"], "map": p["map"]}
                out.setdefault(f"{model}_committed", {})[s] = metrics_table(committed, d, mc, true_ev)
    return out


def bootstrap_ci(x: np.ndarray, n_boot: int = 1000, seed: int = 0) -> Tuple[float, float, float]:
    """Mean and 95 % record-bootstrap interval (records with NaN dropped)."""
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, x.size, size=(n_boot, x.size))
    bs = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _nanmean(x: np.ndarray, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis)


def summarise(per_record: Dict[str, Dict]) -> List[Dict]:
    rows = []
    for name, by_seed in per_record.items():
        arrs = [by_seed[s] for s in sorted(by_seed, key=str)]
        avg = _nanmean(np.stack(arrs), 0) if len(arrs) > 1 else arrs[0]
        for mi, metric in enumerate(METRICS):
            per_seed = [float(_nanmean(a[:, mi])) for a in arrs]
            mean, lo, hi = bootstrap_ci(avg[:, mi])
            rows.append({"name": name, "metric": metric, "mean": mean, "ci_lo": lo, "ci_hi": hi,
                         "seed_sd": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else float("nan"),
                         "n_seeds": len(per_seed), "per_seed": json.dumps([round(v, 5) for v in per_seed])})
    return rows


def paired_difference(per_record: Dict[str, Dict], a: str, b: str, metric: str) -> Tuple[float, float, float]:
    """Mean and 95 % interval of the per-record difference a - b (each model averaged over its own seeds)."""
    if a not in per_record or b not in per_record:
        return float("nan"), float("nan"), float("nan")
    mi = METRICS.index(metric)
    xa = _nanmean(np.stack(list(per_record[a].values())), 0)[:, mi]
    xb = _nanmean(np.stack(list(per_record[b].values())), 0)[:, mi]
    return bootstrap_ci(xa - xb)
