"""Synthetic ECG-like records with controlled conditional uncertainty of beat timing and morphology.

Geometry: 100 Hz, history [0, 20) s (2000 samples), future [20, 30) s (1000 samples).

Timing. R times follow t_k = t_{k-1} + RR_k with
    RR_k = m0 + slope * t_{k-1} + a_resp * sin(2 pi f_resp t_{k-1} + phi) + eps_k,   eps_k ~ N(0, sigma_eps^2),
clipped to [0.4, 2.0] s. m0, slope and the respiratory modulation are record-level and inferable from the history (the
respiratory signal r(t) = sin(2 pi f_resp t + phi) is given to every model as a second input channel); the innovation
eps is not, so the conditional SD of the k-th future beat time grows like sigma_eps * sqrt(k).

Morphology. A beat is a sum of five Gaussians (P, Q, R, S, T) with amplitude A, width W and offset o from R:
    A_{k,w} = s_amp * amp_tpl(k),w * exp(eta^A) * (1 + c_resp * r(t_k)),   W_{k,w} = s_w * width_tpl(k),w * exp(eta^W),
with eta ~ N(0, sigma_eta^2) per beat, wave and parameter. The template index tpl(k) in {0, 1, 2, 3} switches to a
uniformly chosen other template with probability h per beat.
Signal: y(t) = 0.03 r(t) + sum of all beats + N(0, 0.01^2) noise.

The waveform model is the dynamical ECG model of McSharry et al. (IEEE TBME 50:289, 2003): Gaussian P, Q, R, S and T
waves driven by an RR process, evaluated in closed form on the time axis. Both the timing and the morphology processes
are split into a part inferable from the history and an innovation whose size is the controlled quantity:
sigma_eps for timing (T1-T4) and (sigma_eta, h) for morphology (M1-M4), as listed in Supplementary Table S22.

Legal futures (``mc_reference``): everything realised in the history is held fixed (latent parameters, all beats with
t_k < 20 s, and the first future beat when its P wave starts inside the history); innovations, template switches and
noise of the later beats are redrawn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

FS = 100
HIST_S, FUT_S = 20.0, 10.0
T0 = HIST_S
N_HIST, N_FUT = int(HIST_S * FS), int(FUT_S * FS)
SIM_START, SIM_END = -2.0, 31.5          # beats are simulated over this span (tails beyond the future are irrelevant)
NOISE_SD = 0.01
BASELINE_RESP = 0.03
RR_MIN, RR_MAX = 0.4, 2.0
REVEAL_S = 0.25                            # first future beat is "revealed" by its P wave if t_1 - REVEAL_S < T0
WAVES = ("P", "Q", "R", "S", "T")
# templates: rows P,Q,R,S,T; columns (amplitude, width SD in s, offset from R in s)
TEMPLATES = np.array([
    [[0.15, 0.025, -0.180], [-0.10, 0.012, -0.030], [1.00, 0.018, 0.0], [-0.25, 0.014, 0.030], [0.30, 0.060, 0.250]],   # 0 normal
    [[0.15, 0.025, -0.180], [-0.10, 0.012, -0.030], [1.00, 0.018, 0.0], [-0.25, 0.014, 0.030], [-0.30, 0.060, 0.250]],  # 1 inverted T
    [[0.12, 0.025, -0.180], [-0.05, 0.015, -0.045], [0.80, 0.030, 0.0], [-0.45, 0.030, 0.055], [-0.25, 0.070, 0.300]],  # 2 wide QRS
    [[0.05, 0.025, -0.180], [-0.08, 0.012, -0.030], [1.10, 0.018, 0.0], [-0.10, 0.014, 0.030], [0.60, 0.050, 0.230]],   # 3 tall T
], dtype=np.float64)
N_TPL = TEMPLATES.shape[0]

# ---- grid levels ------------------------------------------------------------------------------------------------
TIMING_LEVELS = {1: 0.002, 2: 0.006, 3: 0.018, 4: 0.054}                  # sigma_eps (s)
MORPH_LEVELS = {1: (0.02, 0.00), 2: (0.06, 0.01), 3: (0.15, 0.04), 4: (0.35, 0.12)}   # (sigma_eta, h)
SCENARIOS = {"A": (1, 1), "B": (4, 1), "C": (1, 4), "D": (4, 4)}
N_TRAIN, N_VAL, N_TEST, N_MC = 12000, 1000, 600, 64
SEED_BASE = {"train": 1000, "val": 2000, "test": 3000, "mc": 4000}


def cell_id(it: int, im: int) -> str:
    return f"T{it}M{im}"


def cell_index(it: int, im: int) -> int:
    return 4 * (it - 1) + (im - 1)


def cell_params(it: int, im: int) -> Dict[str, float]:
    s_eta, h = MORPH_LEVELS[im]
    return {"sigma_eps": TIMING_LEVELS[it], "sigma_eta": s_eta, "h": h}


def all_cells() -> List[Tuple[int, int]]:
    return [(it, im) for it in range(1, 5) for im in range(1, 5)]


# ---- record-level latent (inferable) ---------------------------------------------------------------------------------
@dataclass
class Latent:
    m0: float
    slope: float
    f_resp: float
    phi_resp: float
    a_resp: float
    s_amp: float
    s_w: float
    c_resp: float
    tpl0: int
    t_first: float

    def as_array(self) -> np.ndarray:
        return np.array([self.m0, self.slope, self.f_resp, self.phi_resp, self.a_resp, self.s_amp, self.s_w,
                         self.c_resp, self.tpl0, self.t_first], dtype=np.float64)

    @staticmethod
    def from_array(a) -> "Latent":
        a = np.asarray(a, dtype=np.float64)
        return Latent(*[float(v) for v in a[:8]], int(round(a[8])), float(a[9]))


LATENT_NAMES = ["m0", "slope", "f_resp", "phi_resp", "a_resp", "s_amp", "s_w", "c_resp", "tpl0", "t_first"]


def sample_latent(rng: np.random.Generator) -> Latent:
    return Latent(m0=float(rng.uniform(0.70, 1.00)), slope=float(rng.uniform(-0.002, 0.002)),
                  f_resp=float(rng.uniform(0.20, 0.33)), phi_resp=float(rng.uniform(0.0, 2 * np.pi)),
                  a_resp=float(rng.uniform(0.03, 0.06)), s_amp=float(rng.uniform(0.8, 1.2)),
                  s_w=float(rng.uniform(0.85, 1.15)), c_resp=float(rng.uniform(0.05, 0.15)),
                  tpl0=int(rng.integers(0, N_TPL)), t_first=float(rng.uniform(-1.5, -0.5)))


def resp_signal(lat: Latent, t: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * lat.f_resp * t + lat.phi_resp)


def time_axis(n0: int, n: int) -> np.ndarray:
    return (np.arange(n0, n0 + n, dtype=np.float64)) / FS


# ---- beat process -----------------------------------------------------------------------------------------------------
@dataclass
class Beats:
    t: np.ndarray        # (K,) R times (s)
    tpl: np.ndarray      # (K,) template index
    eta: np.ndarray      # (K, 5, 2) log-amplitude / log-width innovations


def _step_rr(lat: Latent, t_prev: float, eps: float) -> float:
    rr = lat.m0 + lat.slope * t_prev + lat.a_resp * np.sin(2 * np.pi * lat.f_resp * t_prev + lat.phi_resp) + eps
    return float(min(max(rr, RR_MIN), RR_MAX))


def simulate_beats(rng: np.random.Generator, lat: Latent, p: Dict[str, float], t_start: float, tpl_start: int,
                    t_end: float = SIM_END, first_is_given: bool = False) -> Beats:
    """Simulate beats after (t_start, tpl_start).  first_is_given: the beat at t_start itself is part of the output
    (used to start the chain at the record's first beat); otherwise beats strictly after t_start are returned."""
    ts, tp, et = [], [], []
    t, tpl = t_start, tpl_start
    if first_is_given:
        ts.append(t); tp.append(tpl); et.append(rng.normal(0.0, p["sigma_eta"], size=(5, 2)))
    while True:
        t = t + _step_rr(lat, t, rng.normal(0.0, p["sigma_eps"]))
        if t > t_end:
            break
        if p["h"] > 0 and rng.random() < p["h"]:
            tpl = int((tpl + rng.integers(1, N_TPL)) % N_TPL)        # uniformly one of the OTHER templates
        ts.append(t); tp.append(tpl); et.append(rng.normal(0.0, p["sigma_eta"], size=(5, 2)))
    return Beats(np.asarray(ts, np.float64), np.asarray(tp, np.int64), np.asarray(et, np.float64).reshape(-1, 5, 2))


def beat_params(lat: Latent, b: Beats, modulate: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> amplitudes (K,5), widths (K,5), offsets (K,5).  modulate=False drops the inferable respiratory amplitude
    modulation (used for the morphology-given-times diagnostic)."""
    tpl = TEMPLATES[b.tpl]                                             # (K,5,3)
    amp = lat.s_amp * tpl[:, :, 0] * np.exp(b.eta[:, :, 0])
    if modulate:
        amp = amp * (1.0 + lat.c_resp * resp_signal(lat, b.t))[:, None]
    wid = lat.s_w * tpl[:, :, 1] * np.exp(b.eta[:, :, 1])
    return amp, wid, tpl[:, :, 2]


def render(lat: Latent, b: Beats, t: np.ndarray, rng: Optional[np.random.Generator] = None,
           baseline: bool = True, noise: bool = True) -> np.ndarray:
    """Clean + (optional) baseline + (optional) noise rendering of beats b on time axis t (s)."""
    y = np.zeros(t.shape[0], np.float64)
    if b.t.size:
        amp, wid, off = beat_params(lat, b)
        centre = b.t[:, None] + off                                    # (K,5)
        # only beats within 1 s of the axis matter
        keep = (b.t >= t[0] - 1.0) & (b.t <= t[-1] + 1.0)
        for k in np.flatnonzero(keep):
            lo = np.searchsorted(t, b.t[k] - 0.6); hi = np.searchsorted(t, b.t[k] + 0.8)
            if hi <= lo:
                continue
            tt = t[lo:hi]
            d = (tt[None, :] - centre[k][:, None]) / wid[k][:, None]
            y[lo:hi] += (amp[k][:, None] * np.exp(-0.5 * d * d)).sum(0)
    if baseline:
        y += BASELINE_RESP * resp_signal(lat, t)
    if noise:
        assert rng is not None
        y += rng.normal(0.0, NOISE_SD, size=t.shape[0])
    return y


def beat_shape(lat: Latent, b: Beats, k: int, rel_t: np.ndarray, modulate: bool = False) -> np.ndarray:
    """The k-th beat's own waveform on a grid relative to its R time (no neighbours, no baseline, no noise)."""
    amp, wid, off = beat_params(lat, b, modulate=modulate)
    d = (rel_t[None, :] - off[k][:, None]) / wid[k][:, None]
    return (amp[k][:, None] * np.exp(-0.5 * d * d)).sum(0)


# ---- records ----------------------------------------------------------------------------------------------------------
@dataclass
class Record:
    lat: Latent
    beats: Beats           # all simulated beats (history + future)
    ecg: np.ndarray        # (N_HIST + N_FUT,) float64, t in [0, 30)
    n_cond: int            # beats [0, n_cond) are conditioned on (history beats + revealed first future beat)

    @property
    def hist(self) -> np.ndarray:
        return self.ecg[:N_HIST]

    @property
    def fut(self) -> np.ndarray:
        return self.ecg[N_HIST:]

    @property
    def fut_beats(self) -> np.ndarray:
        return self.beats.t[self.beats.t >= T0]

    @property
    def hist_beats(self) -> np.ndarray:
        return self.beats.t[(self.beats.t >= 0.0) & (self.beats.t < T0)]


def n_conditioned(bt: np.ndarray) -> int:
    n = int(np.sum(bt < T0))
    if n < bt.size and bt[n] - REVEAL_S < T0:
        n += 1
    return n


def make_record(rng: np.random.Generator, p: Dict[str, float]) -> Record:
    lat = sample_latent(rng)
    b = simulate_beats(rng, lat, p, lat.t_first, lat.tpl0, first_is_given=True)
    t = time_axis(0, N_HIST + N_FUT)
    ecg = render(lat, b, t, rng)
    return Record(lat, b, ecg, n_conditioned(b.t))


def record_rng(split: str, it: int, im: int, i: int) -> np.random.Generator:
    """Deterministic per-record generator: SeedSequence(base + cell_index, i)."""
    return np.random.default_rng(np.random.SeedSequence([SEED_BASE[split] + cell_index(it, im), i]))


def make_split(it: int, im: int, split: str, n: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Arrays for one cell / split: hist (N,2000) f32, fut (N,1000) f32, latent (N,10), beat times padded (N,48) with
    NaN, template per beat (N,48), n_cond (N,).  Deterministic (seeded per record)."""
    p = cell_params(it, im)
    n = {"train": N_TRAIN, "val": N_VAL, "test": N_TEST}[split] if n is None else n
    hist = np.zeros((n, N_HIST), np.float32); fut = np.zeros((n, N_FUT), np.float32)
    latent = np.zeros((n, len(LATENT_NAMES)), np.float64)
    bt = np.full((n, 48), np.nan, np.float64); btpl = np.full((n, 48), -1, np.int64); ncond = np.zeros(n, np.int64)
    for i in range(n):
        r = make_record(record_rng(split, it, im, i), p)
        hist[i] = r.hist; fut[i] = r.fut; latent[i] = r.lat.as_array()
        k = min(r.beats.t.size, 48)
        bt[i, :k] = r.beats.t[:k]; btpl[i, :k] = r.beats.tpl[:k]; ncond[i] = r.n_cond
    return {"hist": hist, "fut": fut, "latent": latent, "beat_t": bt, "beat_tpl": btpl, "n_cond": ncond,
            "params": np.array([p["sigma_eps"], p["sigma_eta"], p["h"]])}


def resp_channels(latent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Known-phase respiratory channel for the history and the future, from the latent array (N,10)."""
    f = latent[:, 2:3]; phi = latent[:, 3:4]
    th = time_axis(0, N_HIST)[None, :]; tf = time_axis(N_HIST, N_FUT)[None, :]
    return (np.sin(2 * np.pi * f * th + phi).astype(np.float32), np.sin(2 * np.pi * f * tf + phi).astype(np.float32))


def future_event_samples(beat_t_row: np.ndarray) -> np.ndarray:
    """Future R-peak sample indices (relative to T0) from a padded beat-time row."""
    t = beat_t_row[np.isfinite(beat_t_row)]
    t = t[(t >= T0) & (t < T0 + FUT_S)]
    return np.round((t - T0) * FS).astype(np.int64)


def history_event_samples(beat_t_row: np.ndarray) -> np.ndarray:
    t = beat_t_row[np.isfinite(beat_t_row)]
    t = t[(t >= 0.0) & (t < T0)]
    return np.round(t * FS).astype(np.int64)


def peak_map(event_samples: np.ndarray, n: int, sigma_s: float) -> np.ndarray:
    """Soft timing map: sum of Gaussians (height 1) at the event samples."""
    m = np.zeros(n, np.float32)
    if event_samples.size == 0:
        return m
    x = np.arange(n, dtype=np.float64)
    s = sigma_s * FS
    for e in event_samples:
        m += np.exp(-0.5 * ((x - e) / s) ** 2).astype(np.float32)
    return np.minimum(m, 1.0)
