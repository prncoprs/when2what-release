"""Run Predict -> Commit -> Generate on evaluation windows.

  python scripts/predict.py --run runs/when2what_s0 --windows data/vitaldb/ecg/test_windows.csv \
      --out preds/when2what_s0_test.npz

Writes one .npz with, per window: window_id, subject, the raw timing map and the forecast waveform (float16, as
stored in the paper's pipeline), committed events, history events, offline reference events of the true future,
and the normalised target y.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from when2what.config import load_config
from when2what.data import EvalWindows
from when2what.inference import Forecaster
from when2what.models import build_model, use_reference_mamba


def object_array(items):
    out = np.empty(len(items), dtype=object)
    for i, x in enumerate(items):
        out[i] = np.asarray(x, np.int64)
    return out


def load_model(run_dir: Path, cfg, device):
    model = build_model(cfg)
    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(state)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training run directory with config.json and best.pt")
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cpu", action="store_true", help="run Mamba with the reference scan on CPU (slow)")
    a = ap.parse_args()
    run = Path(a.run)
    values = json.loads((run / "config.json").read_text())
    if a.data:
        values["data_dir"] = a.data
    cfg = load_config(overrides=values)
    device = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cpu":
        use_reference_mamba()
    forecaster = Forecaster(load_model(run, cfg, device), cfg, device)
    data = EvalWindows(cfg, a.windows)
    maps, waves, ys, mus, sds, events, hist_events, refs = [], [], [], [], [], [], [], []
    for batch in DataLoader(data, batch_size=a.batch, shuffle=False, num_workers=2):
        out = forecaster(batch["x"])
        maps.append(out["timing_map"].astype(np.float16))
        waves.append(out["wave"].astype(np.float16))
        ys.append(batch["y"].numpy())
        mus.append(batch["mu"].numpy())
        sds.append(batch["sd"].numpy())
        events += out["events"]
        hist_events += out["history_events"]
        for k in batch["index"].tolist():
            i, start = data.windows[k]
            labels = data.store.labels(i).astype(np.int64)
            origin = start + cfg.history_pts
            refs.append(labels[(labels >= origin) & (labels < origin + cfg.horizon_pts)] - origin)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, window_id=np.array([r["window_id"] for r in data.rows]),
             subject=np.array([r["subject"] for r in data.rows]), timing_map=np.concatenate(maps),
             wave=np.concatenate(waves), y=np.concatenate(ys), mu=np.concatenate(mus), sd=np.concatenate(sds),
             events=object_array(events), history_events=object_array(hist_events), reference=object_array(refs),
             variant=cfg.variant, signal=cfg.signal)
    print(f"{len(data)} windows -> {a.out}")


if __name__ == "__main__":
    main()
