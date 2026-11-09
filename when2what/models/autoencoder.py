"""Convolutional autoencoder (Enc, Dec) that maps a 500-Hz waveform to 2-s tokens.

Each encoder stage is a strided Conv1d with kernel 2s, stride s and padding s//2,
so the time axis shrinks by exactly s; the decoder mirrors it with ConvTranspose1d.
Kernels overlap neighbouring tokens, and the decoder reconstructs the whole token
sequence jointly, so there are no seams at token boundaries.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, patch: int = 1000, latent_dim: int = 512,
                 channels: Sequence[int] = (128, 256), strides: Sequence[int] = (10, 10, 10)):
        super().__init__()
        strides, channels = list(strides), list(channels)
        if int(torch.tensor(strides).prod()) != patch:
            raise ValueError("the product of the strides must equal the token patch")
        if len(channels) != len(strides) - 1:
            raise ValueError("channels must have len(strides) - 1 entries")
        self.patch = patch
        self.latent_dim = latent_dim
        self.strides = strides

        enc_ch = [1, *channels, latent_dim]
        dec_ch = enc_ch[::-1]
        enc: List[nn.Module] = []
        for i, s in enumerate(strides):
            enc.append(nn.Conv1d(enc_ch[i], enc_ch[i + 1], kernel_size=2 * s, stride=s, padding=s // 2))
            if i < len(strides) - 1:
                enc.append(nn.GELU())
        dec: List[nn.Module] = []
        for i, s in enumerate(reversed(strides)):
            dec.append(nn.ConvTranspose1d(dec_ch[i], dec_ch[i + 1], kernel_size=2 * s, stride=s, padding=s // 2))
            if i < len(strides) - 1:
                dec.append(nn.GELU())
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L) waveform -> (B, L / patch, latent_dim) tokens."""
        if x.dim() != 2 or x.shape[1] % self.patch:
            raise ValueError(f"expected (B, L) with L a multiple of {self.patch}")
        return self.encoder(x.unsqueeze(1)).transpose(1, 2).contiguous()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, n_tokens, latent_dim) tokens -> (B, n_tokens * patch) waveform."""
        return self.decoder(z.transpose(1, 2).contiguous()).squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
