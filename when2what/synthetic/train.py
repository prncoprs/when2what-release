"""Train one (cell, model type, seed) synthetic run and predict the cell's test records.

Adam (lr 1e-3), batch 64, gradient clipping 1.0, validation every 250 steps; training stops at a validation
plateau (best loss improved by < 0.5 % over the last 20 % of steps, checked from step 8000) or at 30000 steps,
and the best validation checkpoint is used. Outputs in <out>/runs/<cell>_<model>_s<seed>/: best.pt,
curve.csv, preds.npz (wave, [map], [wave_committed]) and done.json.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from . import generator as G
from . import models as M
from .evaluate import map_events

SIGMA_MAP_S = 0.030
LR, BATCH, CLIP = 1.0e-3, 64, 1.0
MAX_STEPS, VAL_EVERY, MIN_STEPS, PLATEAU_REL = 30000, 250, 8000, 0.005


def run_name(cell: str, kind: str, seed: int) -> str:
    return f"{cell}_{kind}_s{seed}"


def tensors(d: Dict[str, np.ndarray], device):
    rh, rf = G.resp_channels(d["latent"])
    tmap = np.stack([G.peak_map(G.future_event_samples(row), G.N_FUT, SIGMA_MAP_S) for row in d["beat_t"]])
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)  # noqa: E731
    return t(d["hist"]), t(rh), t(rf), t(d["fut"]), t(tmap.astype(np.float32))


@torch.no_grad()
def val_loss(net, kind, val, bs=250) -> float:
    net.eval()
    total, n = 0.0, 0
    for s in range(0, val[0].shape[0], bs):
        b = tuple(x[s:s + bs] for x in val)
        loss, _ = M.losses(net, kind, b)
        total += loss.item() * b[0].shape[0]
        n += b[0].shape[0]
    net.train()
    return total / n


@torch.no_grad()
def predict_records(net, kind, d: Dict[str, np.ndarray], device, bs=200) -> Dict[str, np.ndarray]:
    net.eval()
    hist, rh, rf, _, _ = tensors(d, device)
    out: Dict[str, list] = {}
    for s in range(0, hist.shape[0], bs):
        sl = slice(s, s + bs)
        cm = None
        if kind == "when2what":
            m = torch.sigmoid(net(hist[sl], rh[sl], rf[sl], torch.zeros_like(rf[sl]))[1]).cpu().numpy()
            cm = torch.from_numpy(np.stack([G.peak_map(map_events(row), G.N_FUT, SIGMA_MAP_S) for row in m])).to(device)
        for k, v in M.predict(net, kind, hist[sl], rh[sl], rf[sl], committed_map=cm).items():
            out.setdefault(k, []).append(v.cpu().numpy().astype(np.float32))
    return {k: np.concatenate(v, 0) for k, v in out.items()}


def _plateau(curve, step):
    cands = [v for s, _, v in curve if s <= 0.8 * step]
    b80 = min(cands) if cands else curve[0][2]
    best = min(v for _, _, v in curve)
    return (b80 - best) / b80 if b80 > 0 else 0.0


def train_run(cell: str, kind: str, seed: int, out_root: Path, device: str = "cpu", threads: int = 2,
              max_steps: int = MAX_STEPS) -> Dict:
    it, im = int(cell[1]), int(cell[3])
    out_dir = Path(out_root) / "runs" / run_name(cell, kind, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)
    train = tensors(G.make_split(it, im, "train"), dev)
    val = tensors(G.make_split(it, im, "val"), dev)
    n_train = train[0].shape[0]
    net = M.SynthNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    best, best_step, best_state, curve, ema = float("inf"), 0, None, [], None
    step, plateau = 0, False
    net.train()
    while step < max_steps:
        idx = torch.from_numpy(rng.integers(0, n_train, BATCH))
        loss, parts = M.losses(net, kind, tuple(x[idx] for x in train))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
        opt.step()
        step += 1
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % VAL_EVERY == 0:
            v = val_loss(net, kind, val)
            if v < best:
                best, best_step, best_state = v, step, copy.deepcopy(net.state_dict())
            curve.append((step, ema, v))
            print(f"{cell} {kind} s{seed} step {step} train {ema:.5f} val {v:.5f} best {best:.5f}@{best_step}", flush=True)
            if step >= MIN_STEPS and _plateau(curve, step) < PLATEAU_REL:
                plateau = True
                break
    net.load_state_dict(best_state)
    torch.save(best_state, out_dir / "best.pt")
    with open(out_dir / "curve.csv", "w") as f:
        f.write("step,train_ema,val\n")
        f.writelines(f"{s},{e:.6f},{v:.6f}\n" for s, e, v in curve)
    test_file = Path(out_root) / "data" / f"{cell}_test.npz"
    d = dict(np.load(test_file)) if test_file.exists() else G.make_split(it, im, "test")
    np.savez_compressed(out_dir / "preds.npz", **predict_records(net, kind, d, dev))
    info = {"cell": cell, "model": kind, "seed": seed, "params": M.count(net), "steps": step, "best_step": best_step,
            "best_val": best, "plateau": plateau or _plateau(curve, step) < PLATEAU_REL, "wall_s": time.time() - t_start}
    (out_dir / "done.json").write_text(json.dumps(info, indent=1))
    return info
