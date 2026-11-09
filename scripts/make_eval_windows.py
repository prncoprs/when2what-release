"""Draw evaluation windows: at most three per patient with non-overlapping horizons.

  python scripts/make_eval_windows.py --data data/vitaldb/ecg --split test --out data/vitaldb/ecg/test_windows.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from when2what.config import load_config
from when2what.data import SegmentStore
from when2what.data.windows import draw_patient_windows, patient_rng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-patient", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config, {"data_dir": a.data})
    store = SegmentStore(a.data, a.split, cfg.window_pts)
    by_patient = defaultdict(list)
    for i, s in enumerate(store.subjects):
        by_patient[s].append(i)
    rows = []
    for subject in sorted(by_patient, key=lambda v: (len(v), v)):
        segs = by_patient[subject]
        chosen = draw_patient_windows([int(store.lengths[i]) for i in segs], patient_rng(a.seed, a.split, subject),
                                      cfg.history_pts, cfg.horizon_pts, cfg.fs, a.per_patient)
        for k, start in chosen:
            fn = store.files[segs[k]]
            rows.append({"window_id": f"{Path(fn).stem}_{start}", "file": fn, "subject": subject, "start": start})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["window_id", "file", "subject", "start"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} windows from {len({r['subject'] for r in rows})} patients -> {a.out}")


if __name__ == "__main__":
    main()
