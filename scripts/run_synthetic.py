"""Synthetic experiment (Section 4.5; Supplementary Section S7): data and Monte-Carlo reference, training, summary.

  python scripts/run_synthetic.py reference --cell T4M1 --out results/synthetic
  python scripts/run_synthetic.py train --cell T4M1 --model when2what --seed 0 --out results/synthetic [--device cuda]
  python scripts/run_synthetic.py summary --out results/synthetic

The summary writes, for every cell with a reference:
  grid_differences.csv  per-record differences with 95 % record-bootstrap intervals:
                        when2what_committed - direct  (f1, mse_real: Table S23; excess_cm: Fig. 5b)
                        when2what (soft map) - direct (f1, mse_real: Table S24)
  grid_levels.csv       per-model levels (f1, mse_real, excess_cm, peak_amp_ratio) and the cell's conditional
                        beat-time SD over the first 8 beats (Table S22, x axis of Fig. 5)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from when2what.synthetic import evaluate as E
from when2what.synthetic import generator as G
from when2what.synthetic.mc_reference import run_cell
from when2what.synthetic.models import MODEL_TYPES
from when2what.synthetic.train import train_run

COMPARISONS = (("when2what_committed", "direct", ("f1", "mse_real", "excess_cm")),
               ("when2what", "direct", ("f1", "mse_real")))
LEVEL_METRICS = ("f1", "mse_real", "excess_cm", "peak_amp_ratio")


def write_csv(path: Path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summary(out: Path):
    diffs, levels = [], []
    for it, im in G.all_cells():
        cell = G.cell_id(it, im)
        mc_path = out / "mc" / f"{cell}_test_mc.npz"
        if not mc_path.exists():
            continue
        timing_sd_ms = float(np.nanmean(np.load(mc_path)["sd_k"][:, :8]) * 1000)
        per_record = E.evaluate_cell(out, cell)
        for a, b, metrics in COMPARISONS:
            for metric in metrics:
                mean, lo, hi = E.paired_difference(per_record, a, b, metric)
                diffs.append({"cell": cell, "comparison": f"{a} - {b}", "metric": metric,
                              "mean": mean, "ci_lo": lo, "ci_hi": hi})
        for row in E.summarise(per_record):
            if row["metric"] in LEVEL_METRICS:
                levels.append({"cell": cell, "timing_sd_ms": round(timing_sd_ms, 1), "model": row["name"],
                               "metric": row["metric"], "mean": row["mean"], "ci_lo": row["ci_lo"],
                               "ci_hi": row["ci_hi"], "n_seeds": row["n_seeds"]})
    if not diffs:
        print("no cell with a Monte-Carlo reference found")
        return
    write_csv(out / "grid_differences.csv", diffs)
    write_csv(out / "grid_levels.csv", levels)
    print(f"{len({d['cell'] for d in diffs})} cells -> {out / 'grid_differences.csv'}, {out / 'grid_levels.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["reference", "train", "summary"])
    ap.add_argument("--cell", default=None, help="e.g. T4M1")
    ap.add_argument("--model", choices=MODEL_TYPES, default="when2what")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if a.command == "reference":
        print(json.dumps(run_cell(int(a.cell[1]), int(a.cell[3]), out)))
    elif a.command == "train":
        print(json.dumps(train_run(a.cell, a.model, a.seed, out, a.device, a.threads)))
    else:
        summary(out)


if __name__ == "__main__":
    main()
