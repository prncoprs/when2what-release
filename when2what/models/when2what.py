"""When2What network: timing predictor P and waveform generator G (Sections 3.3 and 3.4).

P and G share the autoencoder (Enc, Dec) and the sequence backbone:

    history x --Enc--> 60 history tokens (2 s each) -------------------------+
    30 horizon query tokens + spectral embedding of x -----------------------+--> backbone
    P:  queries                               -> timing head -> timing map m_hat
    G:  queries + token embedding of the map  -> Dec -> 500-Hz fusion with the
                                                         timing map -> waveform y_hat

The timing map fed to G is the ground-truth map m when When2What is trained (teacher
forcing), the committed map m_tilde = C(m_hat) at inference, and an all-zero map for
the Direct and Direct-MT controls (training and inference; Table 3).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from .autoencoder import ConvAutoencoder
from .backbone import HybridBackbone, TCNBackbone


GATE_INIT = math.log(math.log(2.0) / (1.0 - math.log(2.0)))     # initial gate value sigmoid(GATE_INIT) = ln 2


class SpectralEmbedding(nn.Module):
    """Low-frequency spectrum of the history -> one d-dim vector added to every horizon query.

    Discrete Fourier transform magnitudes of the history below 10 Hz are log-compressed,
    weighted by a learnable sigmoid frequency gate, and projected to the token space by an MLP.
    """

    def __init__(self, d_model: int, n_freq: int):
        super().__init__()
        self.n_freq = n_freq
        self.gate = nn.Parameter(torch.full((n_freq,), GATE_INIT))
        self.proj = nn.Sequential(nn.Linear(n_freq, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag = torch.fft.rfft(x.float(), dim=1).abs()
        if mag.shape[1] < self.n_freq:
            mag = F.pad(mag, (0, self.n_freq - mag.shape[1]))
        mag = torch.log1p(mag[:, :self.n_freq]) * torch.sigmoid(self.gate)
        return self.proj(mag.to(x.dtype))


class TimingHead(nn.Module):
    """Horizon tokens (B, n_fut, d) -> timing map (B, n_fut * patch) by transposed convolutions."""

    def __init__(self, d_model: int, strides: Sequence[int], hidden: Sequence[int] = (256, 128)):
        super().__init__()
        ch = [d_model, *hidden, 1]
        layers = []
        for i, s in enumerate(strides):
            layers.append(nn.ConvTranspose1d(ch[i], ch[i + 1], kernel_size=2 * s, stride=s, padding=s // 2))
            if i < len(strides) - 1:
                layers.append(nn.GELU())
        self.up = nn.Sequential(*layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.up(tokens.transpose(1, 2).contiguous()).squeeze(1)


class Generator(nn.Module):
    """Timing-map conditioning of the waveform generator G at two scales.

    ``condition`` (2-s token scale): convolutional token embedding of the timing map, a
    strided convolution with kernel = stride = 1000 samples written as a linear map of
    each 2-s chunk, added to the matching horizon query before the backbone.
    ``to_latents``: residual MLP from backbone features to decoder tokens.
    ``fuse`` (500-Hz scale): convolutional fusion of [decoded waveform, timing map],
    added to the decoded waveform.
    """

    def __init__(self, d_model: int, patch: int, channels: int = 32, kernel: int = 31):
        super().__init__()
        self.patch = patch
        self.map_embed = nn.Linear(patch, d_model)
        self.morph = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
                                   nn.Linear(d_model, d_model))
        self.refine = nn.Sequential(nn.Conv1d(2, channels, kernel, padding=kernel // 2), nn.GELU(),
                                    nn.Conv1d(channels, 1, kernel, padding=kernel // 2))

    def condition(self, queries: torch.Tensor, timing_map: torch.Tensor) -> torch.Tensor:
        b, n, _ = queries.shape
        return queries + self.map_embed(timing_map.reshape(b, n, self.patch))

    def to_latents(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.morph(features)

    def fuse(self, decoded: torch.Tensor, timing_map: torch.Tensor) -> torch.Tensor:
        return decoded + self.refine(torch.stack([decoded, timing_map], dim=1)).squeeze(1)


class When2What(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.latent_dim
        self.n_fut = cfg.n_fut_tokens
        self.horizon_pts = cfg.horizon_pts
        self.autoencoder = ConvAutoencoder(cfg.token_patch, d, cfg.ae_channels, cfg.ae_strides)
        if cfg.backbone == "tcn":
            self.backbone = TCNBackbone(d)
        else:
            self.backbone = HybridBackbone(d, cfg.n_layers, cfg.d_state, cfg.d_conv, cfg.expand,
                                           cfg.attn_every, cfg.n_heads)
        self.fut_query = nn.Parameter(torch.randn(self.n_fut, d) * 0.02)
        self.fut_pos = nn.Parameter(torch.randn(self.n_fut, d) * 0.02)
        self.hist_type = nn.Parameter(torch.zeros(d))
        self.fut_type = nn.Parameter(torch.zeros(d))
        self.timing_head = TimingHead(d, cfg.ae_strides, cfg.timing_head_hidden)
        self.generator = Generator(d, cfg.token_patch)
        self.spectral = SpectralEmbedding(d, cfg.spectral_n_freq)

    # ------------------------------------------------------------ shared parts
    def tokens(self, x: torch.Tensor):
        """History tokens and horizon queries (with the spectral embedding)."""
        z_hist = self.autoencoder.encode(x) + self.hist_type
        q = (self.fut_query + self.fut_pos + self.fut_type).unsqueeze(0).expand(x.shape[0], -1, -1)
        return z_hist, q + self.spectral(x).unsqueeze(1)

    def _horizon_features(self, z_hist: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        n_hist = z_hist.shape[1]
        return self.backbone(torch.cat([z_hist, queries], dim=1))[:, n_hist:]

    def reconstruct_history(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder(x)

    # ------------------------------------------------------------ the two passes
    def predict_timing(self, x: torch.Tensor, tokens=None) -> torch.Tensor:
        """Pass 1 (P): raw timing map m_hat, shape (B, horizon_pts)."""
        z_hist, q = tokens if tokens is not None else self.tokens(x)
        return self.timing_head(self._horizon_features(z_hist, q))

    def generate(self, x: torch.Tensor, timing_map: torch.Tensor, tokens=None) -> torch.Tensor:
        """Pass 2 (G): forecast waveform conditioned on ``timing_map`` (B, horizon_pts)."""
        z_hist, q = tokens if tokens is not None else self.tokens(x)
        features = self._horizon_features(z_hist, self.generator.condition(q, timing_map))
        decoded = self.autoencoder.decode(self.generator.to_latents(features))
        return self.generator.fuse(decoded, timing_map)

    def forward(self, x: torch.Tensor, timing_map: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Both passes on a shared encoding. ``timing_map=None`` feeds an all-zero map to G."""
        tokens = self.tokens(x)
        m_hat = self.predict_timing(x, tokens)
        if timing_map is None:
            timing_map = torch.zeros(x.shape[0], self.horizon_pts, device=x.device)
        return {"timing": m_hat, "wave": self.generate(x, timing_map, tokens)}


def build_model(cfg: Config) -> When2What:
    return When2What(cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
