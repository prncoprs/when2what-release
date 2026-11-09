"""Experiment configuration.

All hyperparameters live in one dataclass. A YAML file (see ``configs/``) and
command-line overrides are layered on top of the defaults below, which are the
ECG settings reported in the paper (Supplementary Table S2).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

SIGNALS = ("ecg", "abp", "ppg")
VARIANTS = ("when2what", "direct", "direct_mt")
WAVE_LOSSES = ("mse", "split_mse")
BACKBONES = ("hybrid", "tcn")

# Signal-specific settings. The commit rule of ABP and PPG was selected on the
# validation cohort of each signal; ECG uses the defaults of ``Config``.
SIGNAL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "ecg": {},
    "abp": {"peak_sigma_s": 0.005, "commit_window_s": 5.0, "commit_floor": 0.15},
    "ppg": {"peak_sigma_s": 0.020, "commit_window_s": 10.0, "commit_floor": 0.10},
}


@dataclass
class Config:
    # signal and window geometry
    signal: str = "ecg"
    fs: int = 500
    history_s: int = 120
    horizon_s: int = 60

    # convolutional autoencoder (one 512-d token per 2 s)
    token_patch: int = 1000
    latent_dim: int = 512
    ae_channels: Tuple[int, ...] = (128, 256)
    ae_strides: Tuple[int, ...] = (10, 10, 10)

    # shared sequence backbone
    backbone: str = "hybrid"
    n_layers: int = 4
    attn_every: int = 2
    n_heads: int = 8
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    spectral_band_hz: float = 10.0
    timing_head_hidden: Tuple[int, ...] = (256, 128)

    # model variant and waveform loss
    variant: str = "when2what"
    wave_loss: str = "mse"

    # timing map
    peak_sigma_s: float = 0.010

    # loss weights
    peak_pos_weight: float = 50.0
    soft_f1_weight: float = 0.5
    w_timing: float = 1.0
    w_wave: float = 1.0
    w_hist: float = 0.3
    diff_weight: float = 1.0
    split_bound: float = 1.6
    split_w_inner: float = 5.0
    split_w_outer: float = 1.0

    # optimisation
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 3000
    max_steps: int = 60000
    batch_size: int = 24
    grad_clip: float = 1.0
    bf16: bool = True
    val_every: int = 200
    val_batches: int = 20
    max_val_windows: int = 1000
    train_stride_s: int = 1
    val_stride_s: int = 10
    seed: int = 42
    num_workers: int = 12

    # leakage-free history filtering
    context_s: float = 10.0
    pad_s: float = 5.0

    # commit step C
    commit_window_s: float = 5.0
    commit_quantile: float = 90.0
    commit_floor: float = 0.15
    commit_spacing_frac: float = 0.5

    # paths
    data_dir: str = "data/vitaldb/ecg"
    out_dir: str = "runs"

    # ------------------------------------------------------------ derived
    @property
    def history_pts(self) -> int:
        return self.history_s * self.fs

    @property
    def horizon_pts(self) -> int:
        return self.horizon_s * self.fs

    @property
    def window_pts(self) -> int:
        return self.history_pts + self.horizon_pts

    @property
    def n_hist_tokens(self) -> int:
        return self.history_pts // self.token_patch

    @property
    def n_fut_tokens(self) -> int:
        return self.horizon_pts // self.token_patch

    @property
    def spectral_n_freq(self) -> int:
        return int(self.spectral_band_hz * self.history_pts / self.fs) + 1

    @property
    def train_stride_pts(self) -> int:
        return self.train_stride_s * self.fs

    @property
    def val_stride_pts(self) -> int:
        return self.val_stride_s * self.fs

    @property
    def uses_timing_loss(self) -> bool:
        return self.variant in ("when2what", "direct_mt")

    @property
    def conditioned(self) -> bool:
        """True when the generator receives a timing map (When2What)."""
        return self.variant == "when2what"

    def validate(self) -> "Config":
        assert self.signal in SIGNALS, self.signal
        assert self.variant in VARIANTS, self.variant
        assert self.wave_loss in WAVE_LOSSES, self.wave_loss
        assert self.backbone in BACKBONES, self.backbone
        assert self.history_pts % self.token_patch == 0
        assert self.horizon_pts % self.token_patch == 0
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _coerce(name: str, value: Any) -> Any:
    tuple_fields = {"ae_channels", "ae_strides", "timing_head_hidden"}
    if name in tuple_fields and value is not None:
        return tuple(value)
    return value


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Defaults <- signal defaults <- YAML file <- overrides."""
    values: Dict[str, Any] = {}
    if path is not None:
        values.update(yaml.safe_load(Path(path).read_text()) or {})
    values.update(overrides or {})
    signal = values.get("signal", Config.signal)
    merged = {**SIGNAL_DEFAULTS[signal], **values}
    known = {f.name for f in fields(Config)}
    unknown = set(merged) - known
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    return Config(**{k: _coerce(k, v) for k, v in merged.items()}).validate()
