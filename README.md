# When2What

Code for **When2What: A Timing-Conditioned Framework for Long-Horizon Forecasting of Continuous Cardiovascular
Waveforms** ([arXiv:2611.08993](https://arxiv.org/abs/2611.08993)).

Given a 120-s history sampled at 500 Hz, When2What forecasts the next 60 s of an electrocardiogram (ECG),
arterial blood pressure (ABP) or photoplethysmogram (PPG) waveform in three stages (Eq. 2 of the paper):

1. **Predict (P)** – a timing predictor estimates a dense timing map `m_hat` of future beat times on the
   500-Hz horizon grid.
2. **Commit (C)** – a non-neural operator selects discrete beat times `E_hat` from `m_hat` (local-percentile
   threshold, refractory spacing from the history RR) and re-encodes them as a committed map `m_tilde` of
   unit-height 10-ms Gaussians.
3. **Generate (G)** – a waveform generator conditions on `m_tilde` and the history, at the 2-s token scale and
   at 500 Hz.

P and G share a convolutional autoencoder (2-s tokens) and a bidirectional Mamba–attention backbone. G is
trained with the ground-truth timing map (teacher forcing) and receives the committed map at inference, so no
future information is used.

## Repository layout

```
when2what/
  config.py              all hyperparameters (defaults = Supplementary Table S2)
  models/                autoencoder, Mamba–attention and TCN backbones, the When2What network (Sections 3.3–3.4)
  data/                  VitalDB preprocessing, leakage-free history input, timing targets (Eq. 3), windows,
                         datasets, ABP/PPG gating, detectors and pulse anchors
  training/              losses (Eqs. 4, 6, 7; Split-MSE) and the training loop
  inference/             commit step C, the Predict -> Commit -> Generate forecaster, event perturbations
  evaluation/            event matching, chance level, event skill (Eq. 1, Eqs. S1–S4), patient-cluster bootstrap,
                         waveform channel, MSE, beat expression, predictability time scale tau
  baselines/             simple RR extrapolation, guarded cadence rule, static historical beat template
  synthetic/             controlled synthetic experiment (generator, Monte-Carlo reference, models, metrics)
configs/                 ecg.yaml, abp.yaml, ppg.yaml
scripts/                 prepare_vitaldb.py, make_eval_windows.py, train.py, predict.py, evaluate.py,
                         run_synthetic.py
tests/                   unit tests (CPU, no data needed)
```

## Installation

Mamba needs CUDA kernels compiled for the local CUDA toolkit. Install a CUDA build of PyTorch first,
then `mamba-ssm` and `causal-conv1d` for the same toolkit, then the remaining requirements:

```bash
pip install torch            # CUDA build matching your driver, see pytorch.org
pip install mamba-ssm causal-conv1d
pip install -r requirements.txt
python -m pytest tests
```

The versions used for the paper are listed in `requirements.txt`. `when2what.models.use_reference_mamba()`
switches Mamba to its pure PyTorch reference scan, which runs on CPU (slow; used by the tests and by
`predict.py --cpu`).

## Data

The experiments use VitalDB (https://vitaldb.net) lead II ECG (`SNUADC/ECG_II`), arterial pressure
(`SNUADC/ART`) and finger photoplethysmogram (`SNUADC/PLETH`) tracks. No data are distributed with this code;
download VitalDB under its data-use terms. MOVER, SHDB-AF and the PhysioNet databases used for external
evaluation are available from their original repositories.

```bash
python scripts/prepare_vitaldb.py --vital-dir /path/to/vital_files --clinical /path/to/clinical_data.csv \
    --signal ecg --out data/vitaldb/ecg --split-csv splits.csv --workers 32
```

This writes, per kept segment, the filtered segment, its raw samples and offline beat labels, plus
`manifest.csv` (`file, case_id, subject, split, length, pre`). Splits are by patient; pass a
`subject,split` table with `--split-csv` (sets `train`, `val`, `test`, `lockbox`) or let
`when2what.data.splits.hash_split` assign patients deterministically.

Preprocessing (`when2what/data/vitaldb.py`):

* ECG: finite runs split around saturation (|x| ≥ 4.5 mV, ±0.2 s), 0.67–40 Hz zero-phase Butterworth
  band-pass, split around spikes (|x| > 10 mV, ±0.2 s), flat-line and NeuroKit2 quality (averageQRS
  ≥ 0.4) screening; training labels are NeuroKit2 R peaks of the filtered segment.
* ABP / PPG: 10-s block artefact gate, 10-Hz low-pass (ABP) or 0.5–8 Hz band-pass (PPG); beat events are
  maximal-upslope pulse anchors (slope-sum onsets for ABP, Elgendi peaks for PPG). The pulse models use
  signal-specific timing-map widths and commit windows (`configs/abp.yaml`, `configs/ppg.yaml`).

**Leakage-free input.** For every window the history is rebuilt from raw samples before the forecast
origin only (up to 10 s of extra past context, odd reflection on the left, a 5-s periodic extension on the
right), filtered, and z-scored by its own mean and standard deviation; the same statistics normalise the target.

## Training

```bash
python scripts/train.py --config configs/ecg.yaml --variant when2what --seed-index 0 --out runs/when2what_s0
python scripts/train.py --config configs/ecg.yaml --variant direct_mt  --seed-index 0 --out runs/direct_mt_s0
python scripts/train.py --config configs/ecg.yaml --variant direct     --seed-index 0 --out runs/direct_s0
```

| option | models in the paper |
|---|---|
| `--variant when2what / direct_mt / direct` | matched conditioning ablation (Tables 3 and 4) |
| `--wave-loss split_mse` | Split-MSE models (Tables 4 and 5) |
| `--wave-loss split_mse --split-bound 1.2` (When2What) / `2.0` (Direct-MT) | validation-tuned Split-MSE (Table 5, Table S18) |
| `--backbone tcn` | TCN backbone (Table 5) |
| `--config configs/abp.yaml`, `configs/ppg.yaml` | ABP and PPG models (Table 6) |
| `--seed-index 0..4` | random seeds: 5 for the ECG models (3 for TCN Direct), 3 for the pulse models; index 0 uses seed 42, index k uses seed k |

Recipe (Table S2): AdamW (peak learning rate 3e-4, weight decay 1e-4), 3,000 warm-up steps then cosine decay
over 60,000 steps, batch 24, bf16, gradient clipping 1.0, validation loss every 200 steps; `best.pt` has the
lowest validation loss. The matched ECG models have 24.88 M parameters.

## Forecasting and evaluation

```bash
python scripts/make_eval_windows.py --data data/vitaldb/ecg --split test --out data/vitaldb/ecg/test_windows.csv
python scripts/predict.py --run runs/when2what_s0 --windows data/vitaldb/ecg/test_windows.csv --out preds/when2what_s0.npz
python scripts/evaluate.py --preds preds/when2what_s*.npz --compare preds/direct_mt_s*.npz --out results/w2w_vs_direct_mt.json
```

Evaluation windows are 120 s + 60 s, at most three per patient with non-overlapping horizons.
`evaluate.py` reports:

* **event channel** – the committed events `E_hat = C(m_hat)` scored against the reference events;
* **waveform channel** – beats detected on the forecast waveform (NeuroKit2 detector for ECG,
  maximal-upslope anchors for ABP/PPG) scored against the same reference events;
* **event skill** `S = (F1 − F1c) / (1 − F1c)` per lead-time bin (Eq. 1), where `F1c` is the expected F1 of a
  constant-rate pulse train at the window's true mean beat interval with a uniformly random phase; the primary
  metric is the near-term event skill, the mean over the [0, 2), [2, 5) and [5, 10) s bins. Events are paired
  greedily one-to-one within ±50 ms over the whole horizon (`--tolerance-ms` for 25 and 100 ms); F1 is averaged
  over seeds within each window; 95 % intervals come from 2,000 patient-cluster bootstrap replicates, shared by
  paired differences;
* **beat expression** – among reference beats in the first 10 s matched by the timing head, the share also
  detected in the generated waveform, N11 / (N11 + N10) (strict: matched events within 50 ms of each other);
* samplewise **MSE** over 0–60 s on z-scored waveforms.

By default the reference events are the offline labels of the true future. The paper evaluates VitalDB and
MOVER ECG against a quality-screened reference (Supplementary Section S3); `evaluate.py --reference` accepts
any reference event set (`window_id`, `events`).

### Where the paper's analyses live

| paper | code |
|---|---|
| Fig. 1a, Table S1: predictability time scale tau | `evaluation.tau_with_ci`, `baselines.simple_rr_extrapolation` |
| Eqs. 2–7: Predict, Commit, Generate, training losses | `models/`, `inference/commit.py`, `training/losses.py` |
| Tables 3, 4, S9, S10 (RQ2): Direct, Direct-MT, When2What | `scripts/train.py --variant`, `scripts/evaluate.py` |
| Table 4, S11: raw map to G (w/o C), timing-source swaps, Direct waveform peaks to G | `Forecaster.generate_from_map`, `Forecaster.generate_from_events` |
| Table 4, S11: static historical template, guarded cadence rule | `baselines.template_forecast`, `baselines.guarded_cadence` |
| Table S11: 30 % event deletion and insertion | `inference.perturb_events`, `inference.window_rng` |
| Tables S15–S17: beat expression | `evaluation.beat_correspondence`, `evaluation.expression_rate` |
| Tables 5, S18, S19 (RQ3): losses, TCN backbone, lockbox and external cohorts | `scripts/train.py` options, `scripts/evaluate.py --reference` |
| Tables 6, S20: ABP and PPG | `configs/abp.yaml`, `configs/ppg.yaml` |
| Section 4.5, Fig. 5, Tables S22–S24 (RQ4): synthetic sharpness–MSE experiment | `when2what/synthetic/`, `scripts/run_synthetic.py` |

Not included: the third-party forecasting baselines of Table 2, the construction of the fused ECG reference
(Supplementary Section S3), the loaders of the external cohorts, the latency measurements (Tables S3 and S4) and
the device-triggering simulation (Supplementary Section S8).

## Synthetic experiment

```bash
python scripts/run_synthetic.py reference --cell T4M1 --out results/synthetic
for m in direct direct_mt when2what; do for s in 0 1 2; do
  python scripts/run_synthetic.py train --cell T4M1 --model $m --seed $s --out results/synthetic --device cuda
done; done
python scripts/run_synthetic.py summary --out results/synthetic
```

100-Hz synthetic ECG (20-s history, 10-s horizon) from the dynamical model of McSharry et al. Cells `T1M1` …
`T4M4` cross four levels of beat-time innovation (per-beat RR innovation SD 2, 6, 18 and 54 ms, i.e.
conditional beat-time SD over the first 8 beats of 4, 11, 34 and 104 ms) with four levels of per-beat morphology
variability (Table S22). Each cell has 12,000 training, 1,000 validation and 600 test records, generated from
fixed seeds; 64 Monte-Carlo futures per test record give the average of possible futures. `summary` writes
committed When2What minus Direct (Table S23, Fig. 5), soft-map When2What minus Direct (Table S24) and the
per-model levels, with 95 % record-bootstrap intervals.

## Citation

```bibtex
@article{zhang2026when2what,
  title   = {When2What: A Timing-Conditioned Framework for Long-Horizon Forecasting of Continuous
             Cardiovascular Waveforms},
  author  = {Zhang, Chaoqi and Yan, Shujie and Zhang, Zhao and Sun, Yanhua and Wang, Yu and Li, Li and
             Meng, Lingzhong and Chen, Xiaoyi and Tang, Haixu},
  journal = {arXiv preprint arXiv:2611.08993},
  year    = {2026}
}
```

## License

MIT (see `LICENSE`). The license covers the code only; VitalDB and the other datasets remain under their own
terms. The code is intended for research and not for clinical diagnosis or patient monitoring.
