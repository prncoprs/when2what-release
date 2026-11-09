"""Train When2What or one of its matched controls.

  python scripts/train.py --config configs/ecg.yaml --variant when2what --seed-index 0 --out runs/when2what_s0
  python scripts/train.py --config configs/ecg.yaml --variant direct_mt --wave-loss split_mse --seed-index 1 ...
  python scripts/train.py --config configs/ecg.yaml --variant when2what --backbone tcn --seed-index 2 ...

Seed index 0 uses seed 42; seed index k >= 1 uses seed k. ``--split-bound`` sets the Split-MSE amplitude band
(default 1.6; the validation-tuned configurations of Supplementary Table S18 use 1.2 for When2What and 2.0 for
Direct-MT, both with the default 5 : 1 weights).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from when2what.config import load_config
from when2what.models import use_reference_mamba
from when2what.training import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ecg.yaml")
    ap.add_argument("--variant", choices=["when2what", "direct", "direct_mt"], default="when2what")
    ap.add_argument("--wave-loss", choices=["mse", "split_mse"], default="mse")
    ap.add_argument("--backbone", choices=["hybrid", "tcn"], default="hybrid")
    ap.add_argument("--split-bound", type=float, default=None)
    ap.add_argument("--seed-index", type=int, default=0)
    ap.add_argument("--data", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    overrides = {"variant": a.variant, "wave_loss": a.wave_loss, "backbone": a.backbone,
                 "seed": 42 if a.seed_index == 0 else a.seed_index}
    if a.split_bound is not None:
        overrides["split_bound"] = a.split_bound
    if a.data:
        overrides["data_dir"] = a.data
    if a.steps:
        overrides["max_steps"] = a.steps
    if a.batch:
        overrides["batch_size"] = a.batch
    if a.workers is not None:
        overrides["num_workers"] = a.workers
    cfg = load_config(a.config, overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("no GPU: Mamba runs with the reference scan on CPU (only practical for small checks)")
        use_reference_mamba()
    best = train(cfg, a.out, device)
    print(f"best checkpoint: {best}")


if __name__ == "__main__":
    main()
