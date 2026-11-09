"""Event skill (event and waveform channels), beat expression, waveform MSE and paired differences.

  python scripts/evaluate.py --preds preds/when2what_s{0..4}_test.npz \
      --compare preds/direct_mt_s{0..4}_test.npz --out results/when2what_vs_direct_mt.json

Every prediction file must cover the same windows in the same order (one file per training seed). Per-window
F1 is averaged over seeds before the skill is computed; 95 % intervals come from a patient-cluster bootstrap
(2,000 replicates, seed 0) shared by all quantities, so differences are paired. For models with a timing head,
the beat expression rate N11 / (N11 + N10) over the first 10 s is also reported (Supplementary Section S5).
``--reference`` replaces the offline reference events stored with the predictions by events from another .npz
(keys window_id, events); ``--tolerance-ms`` sets the matching tolerance (50 ms; 25 and 100 ms in Table S14).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from when2what.evaluation import (LEAD_TIME_BINS_S, PatientBootstrap, beat_correspondence, chance_table, event_f1,
                                  expression_rate, mse, seed_average, skill_difference, skill_with_ci,
                                  waveform_events)
from when2what.evaluation.skill import bin_skill


def _detect(args):
    wave, signal = args
    return waveform_events(np.asarray(wave, np.float32).astype(np.float64), signal)


def load_runs(paths, reference=None):
    runs = [dict(np.load(p, allow_pickle=True)) for p in paths]
    ids = runs[0]["window_id"]
    for r in runs[1:]:
        if not np.array_equal(r["window_id"], ids):
            raise ValueError("prediction files cover different windows")
    refs = list(runs[0]["reference"])
    if reference is not None:
        z = dict(np.load(reference, allow_pickle=True))
        lookup = dict(zip(z["window_id"].tolist(), z["events"]))
        refs = [np.asarray(lookup[w], np.int64) for w in ids.tolist()]
    return runs, refs


def channel_f1(runs, refs, workers, tol_s):
    """Seed-averaged (N, 7) F1 of the committed events and of the events detected on the waveform,
    seed-averaged (N, 3) beat-correspondence counts (n11, n10, n11_strict) and the mean MSE per window."""
    event, wave, corr, err = [], [], [], []
    with mp.get_context("spawn").Pool(workers) as pool:
        for r in runs:
            signal = str(r["signal"])
            detected = pool.map(_detect, [(w, signal) for w in r["wave"]], chunksize=8)
            event.append(np.stack([event_f1(e, ref, tol_s=tol_s) for e, ref in zip(r["events"], refs)]))
            wave.append(np.stack([event_f1(e, ref, tol_s=tol_s) for e, ref in zip(detected, refs)]))
            counts = [beat_correspondence(h, w, ref, tol_s=tol_s) for h, w, ref in zip(r["events"], detected, refs)]
            corr.append(np.array([[c["n11"], c["n10"], c["n11_strict"]] for c in counts], np.float64))
            err.append(np.array([mse(w, y) for w, y in zip(r["wave"], r["y"])]))
    return seed_average(event), seed_average(wave), np.mean(np.stack(corr), 0), np.mean(np.stack(err), 0)


def expression_summary(counts, boot):
    e = expression_rate(counts, boot)
    return {k: e[k] for k in ("expression", "lo", "hi", "strict_expression", "strict_lo", "strict_hi")}, e


def summary(F, C, boot):
    s = skill_with_ci(F, C, boot)
    return {"skill_0_10s": s["skill"], "lo": s["lo"], "hi": s["hi"],
            "f1_by_bin": dict(zip([f"[{a},{b})" for a, b in zip(LEAD_TIME_BINS_S[:-1], LEAD_TIME_BINS_S[1:])],
                                  np.nanmean(F, 0).round(4).tolist())),
            "skill_by_bin": bin_skill(F, C).round(4).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--compare", nargs="*", default=None)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--tolerance-ms", type=float, default=50.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    tol_s = a.tolerance_ms / 1000.0
    runs, refs = load_runs(a.preds, a.reference)
    C = chance_table(refs, tol_s=tol_s)
    boot = PatientBootstrap(runs[0]["subject"], 2000, 0)
    ev, wf, corr, err = channel_f1(runs, refs, a.workers, tol_s)
    result = {"windows": len(refs), "patients": boot.n_patients, "seeds": len(runs), "tolerance_ms": a.tolerance_ms,
              "waveform_channel": summary(wf, C, boot), "mse_0_60s": float(np.mean(err))}
    has_head = str(runs[0]["variant"]) != "direct"
    if has_head:
        result["event_channel"] = summary(ev, C, boot)
        result["beat_expression"], expr_a = expression_summary(corr, boot)
    if a.compare:
        runs_b, _ = load_runs(a.compare)
        if not np.array_equal(runs_b[0]["window_id"], runs[0]["window_id"]):
            raise ValueError("--compare covers different windows")
        ev_b, wf_b, corr_b, err_b = channel_f1(runs_b, refs, a.workers, tol_s)
        d = skill_difference(wf, wf_b, C, boot)
        result["difference_waveform_channel"] = {k: d[k] for k in ("difference", "lo", "hi", "a", "b")}
        if has_head and str(runs_b[0]["variant"]) != "direct":
            d = skill_difference(ev, ev_b, C, boot)
            result["difference_event_channel"] = {k: d[k] for k in ("difference", "lo", "hi", "a", "b")}
            result["beat_expression_compare"], expr_b = expression_summary(corr_b, boot)
            diff = expr_a["replicates"] - expr_b["replicates"]
            result["difference_beat_expression"] = {
                "difference": expr_a["expression"] - expr_b["expression"],
                "lo": float(np.percentile(diff, 2.5)), "hi": float(np.percentile(diff, 97.5))}
        result["mse_0_60s_compare"] = float(np.mean(err_b))
    text = json.dumps(result, indent=1)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
