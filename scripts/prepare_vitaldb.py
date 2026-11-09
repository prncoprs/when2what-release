"""Build clean segments, raw arrays and offline labels from VitalDB .vital files.

  python scripts/prepare_vitaldb.py --vital-dir /path/to/vital_files --clinical /path/to/clinical_data.csv \
      --signal ecg --out data/vitaldb/ecg --split-csv splits.csv --workers 32

Without --split-csv, patients are assigned by ``hash_split`` (70/10/10/10 % train/val/test/lockbox).
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from when2what.data import splits
from when2what.data.vitaldb import process_case


def _job(args):
    path, kind, out = args
    case = Path(path).stem
    rows = []
    try:
        for k, s in enumerate(process_case(path, kind)):
            fn = f"{case}__seg{k:03d}.npy"
            np.save(out / "segments" / fn, s["segment"])
            np.save(out / "raw" / fn, s["raw"])
            np.save(out / "labels" / fn, s["labels"])
            rows.append({"file": fn, "case_id": str(int(case)), "length": int(s["segment"].size), "pre": int(s["pre"])})
    except Exception as e:  # noqa: BLE001
        print(f"{case}: {type(e).__name__}: {e}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vital-dir", required=True)
    ap.add_argument("--clinical", required=True, help="VitalDB clinical_data.csv (case -> subject)")
    ap.add_argument("--signal", choices=["ecg", "abp", "ppg"], default="ecg")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-csv", default=None, help="subject,split table; default: hash split")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N cases only (a quick check)")
    a = ap.parse_args()

    out = Path(a.out)
    for d in ("segments", "raw", "labels"):
        (out / d).mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.vital_dir).glob("*.vital"))[:a.limit]
    subject_of = splits.vitaldb_subjects(a.clinical)
    split_of = splits.read_split(a.split_csv) if a.split_csv else splits.hash_split(sorted(set(subject_of.values())))

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    rows = []
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=8) as pool:
        for k, r in enumerate(pool.imap_unordered(_job, [(str(f), a.signal, out) for f in files], chunksize=1)):
            rows += r
            if k % 200 == 0:
                print(f"{k}/{len(files)} cases, {len(rows)} segments", flush=True)
    for r in rows:
        r["subject"] = subject_of[r["case_id"]]
        r["split"] = split_of.get(r["subject"], "excluded")
    rows.sort(key=lambda r: r["file"])
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "case_id", "subject", "split", "length", "pre"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} segments from {len({r['case_id'] for r in rows})} cases -> {out}")


if __name__ == "__main__":
    main()
