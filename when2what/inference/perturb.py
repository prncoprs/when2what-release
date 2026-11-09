"""Event-input perturbations for the generator-versus-template comparison (Supplementary Table S11).

A fraction (30 %) of the input beats is deleted, or the same number of extra beats is inserted,
before the beat times are given to the waveform generator G or to the static historical template.

  deletion   k = round(0.3 n) beats chosen uniformly without replacement are removed;
  insertion  k extra beats, each placed uniformly at random between a randomly chosen pair of
             adjacent input beats (nothing before the first or after the last beat); an inserted
             beat that coincides with an existing one is merged.

Each window uses its own generator, np.random.default_rng(seed + window index).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

PERTURBATION_SEED = 20261007
KINDS = ("delete", "insert")


def perturb_events(events: Sequence[int], kind: str, rng: np.random.Generator, fraction: float = 0.3) -> np.ndarray:
    e = np.unique(np.asarray(events, np.int64))
    n = e.size
    k = int(np.floor(fraction * n + 0.5))
    if kind == "delete":
        drop = rng.choice(n, k, replace=False) if k > 0 else np.zeros(0, np.int64)
        return np.delete(e, drop)
    if kind == "insert":
        if n < 2 or k == 0:
            return e.copy()
        j = rng.integers(0, n - 1, k)
        u = rng.uniform(0.0, 1.0, k)
        return np.unique(np.concatenate([e, np.rint(e[j] + u * (e[j + 1] - e[j])).astype(np.int64)]))
    raise ValueError(f"kind must be one of {KINDS}")


def window_rng(window_index: int, seed: int = PERTURBATION_SEED) -> np.random.Generator:
    return np.random.default_rng(seed + int(window_index))
