"""Monte-Carlo conditional reference of the synthetic test records.

For every test record, 64 legal futures are simulated from the same history (future j of record i uses
SeedSequence([4000 + cell_index, i, j])). Per record:
  cond_mean     mean of the 64 futures, the average of possible futures: the MSE-optimal forecast given the
                generator's inferable state
  cond_risk     mean squared deviation of the futures from cond_mean (unbiased)
  sd_k          SD over futures of the time of the k-th future beat (k = 1..12; NaN when < 90 % of the futures
                have a k-th beat); its mean over k = 1..8 is the conditional beat-time SD of the cell
                (4, 11, 34 and 104 ms for T1-T4; Supplementary Table S22, Fig. 5)
  morph_var     variance of the beat shape across futures (k = 1..8, -250..+400 ms around R, the beat's own
                Gaussians only) relative to its mean energy; the morphology axis of the grid
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import numpy as np

from . import generator as G

K_SD, K_MORPH = 12, 8
REL_GRID = np.arange(-25, 41, dtype=np.float64) / G.FS            # -250 .. +400 ms, 66 points


def _future_rng(it: int, im: int, i: int, j: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([G.SEED_BASE["mc"] + G.cell_index(it, im), i, j]))


def sample_future(rec: G.Record, p: Dict[str, float], rng: np.random.Generator) -> G.Beats:
    """Beats of one legal future: the first n_cond beats of the record + a fresh continuation."""
    n = rec.n_cond
    fixed = G.Beats(rec.beats.t[:n], rec.beats.tpl[:n], rec.beats.eta[:n])
    cont = G.simulate_beats(rng, rec.lat, p, float(fixed.t[-1]), int(fixed.tpl[-1]), first_is_given=False)
    return G.Beats(np.concatenate([fixed.t, cont.t]), np.concatenate([fixed.tpl, cont.tpl]),
                   np.concatenate([fixed.eta, cont.eta], 0))


def mc_record(it: int, im: int, i: int, rec: G.Record, p: Dict[str, float], n_mc: int = G.N_MC) -> Dict[str, np.ndarray]:
    t_fut = G.time_axis(G.N_HIST, G.N_FUT)
    futs = np.zeros((n_mc, G.N_FUT), np.float64)
    times = np.full((n_mc, K_SD), np.nan)
    shapes = np.full((n_mc, K_MORPH, REL_GRID.size), np.nan)
    for j in range(n_mc):
        rng = _future_rng(it, im, i, j)
        b = sample_future(rec, p, rng)
        futs[j] = G.render(rec.lat, b, t_fut, rng)
        idx = np.flatnonzero((b.t >= G.T0) & (b.t < G.T0 + G.FUT_S))
        k = min(idx.size, K_SD)
        times[j, :k] = b.t[idx[:k]]
        for kk in range(min(idx.size, K_MORPH)):
            shapes[j, kk] = G.beat_shape(rec.lat, b, int(idx[kk]), REL_GRID, modulate=False)
    cm = futs.mean(0)
    risk = float(((futs - cm) ** 2).mean(1).sum() / (n_mc - 1))
    have = np.isfinite(times).mean(0)
    sd_k = np.where(have >= 0.9, np.nanstd(times, axis=0, ddof=1), np.nan)
    ok = np.isfinite(shapes).all(2)                                      # (n_mc, K_MORPH)
    var_k = np.full(K_MORPH, np.nan); energy = 0.0; cnt = 0
    for kk in range(K_MORPH):
        s = shapes[ok[:, kk], kk]
        if s.shape[0] >= 0.9 * n_mc:
            var_k[kk] = s.var(0, ddof=1).mean()
            energy += (s ** 2).mean(); cnt += 1
    energy = energy / max(cnt, 1)
    return {"cond_mean": cm.astype(np.float32), "cond_risk": risk, "sd_k": sd_k, "morph_var_k": var_k / max(energy, 1e-12),
            "morph_var": float(np.nanmean(var_k) / max(energy, 1e-12)), "n_fut_true": int(rec.fut_beats.size)}


def run_cell(it: int, im: int, out_dir: Path, n: int = G.N_TEST, n_mc: int = G.N_MC) -> Dict[str, float]:
    """Test data + MC reference of one cell -> out_dir/data/T.M._test.npz and out_dir/mc/T.M._test_mc.npz."""
    t0 = time.time()
    p = G.cell_params(it, im)
    cid = G.cell_id(it, im)
    (out_dir / "data").mkdir(parents=True, exist_ok=True); (out_dir / "mc").mkdir(parents=True, exist_ok=True)
    data = G.make_split(it, im, "test", n=n)
    cm = np.zeros((n, G.N_FUT), np.float32); risk = np.zeros(n); sd = np.zeros((n, K_SD)); mv = np.zeros(n)
    mvk = np.zeros((n, K_MORPH)); nft = np.zeros(n, np.int64)
    for i in range(n):
        rec = G.make_record(G.record_rng("test", it, im, i), p)        # identical to the record in `data`
        assert np.allclose(rec.fut, data["fut"][i], atol=1e-6)
        r = mc_record(it, im, i, rec, p, n_mc)
        cm[i] = r["cond_mean"]; risk[i] = r["cond_risk"]; sd[i] = r["sd_k"]; mv[i] = r["morph_var"]
        mvk[i] = r["morph_var_k"]; nft[i] = r["n_fut_true"]
    np.savez_compressed(out_dir / "data" / f"{cid}_test.npz", **data)
    np.savez_compressed(out_dir / "mc" / f"{cid}_test_mc.npz", cond_mean=cm, cond_risk=risk, sd_k=sd, morph_var=mv,
                        morph_var_k=mvk, n_fut_true=nft, params=data["params"], n_mc=n_mc)
    return {"cell": cid, "sigma_eps": p["sigma_eps"], "sigma_eta": p["sigma_eta"], "h": p["h"],
            "timing_sd_ms": float(np.nanmean(sd[:, :8]) * 1000), "sd1_ms": float(np.nanmean(sd[:, 0]) * 1000),
            "sd8_ms": float(np.nanmean(sd[:, 7]) * 1000), "morph_var": float(mv.mean()), "cond_risk": float(risk.mean()),
            "fut_var": float(data["fut"].var(1).mean()), "wall_s": time.time() - t0}

