"""A sequence-aware SB3 features extractor — the principled way to add modern sequence architectures.

The env flattens each observation TIME-MAJOR: a [lookback, per_bar] grid (the per-layer OHLCV columns
plus the per-bar portfolio extras) flattened row-major. The existing `custom_net_arch` attention/recurrent
tokens run on that FLAT vector (no real bar axis), so they aren't true temporal models. This extractor
reshapes the flat obs back to [B, lookback, per_bar] and applies a genuine temporal encoder over the
bars, then pools to a fixed feature vector the policy MLP consumes — wired via SB3's
`features_extractor_class` on a plain MlpPolicy (no custom policy needed).

Encoders: 'attn' (single-block self-attention with a learned positional embedding) and 'tcn' (dilated
causal residual convolutions). Both are deliberately small — on a single noisy asset under 0.1% fees,
architecture is a variance-reduction / diagnostic lever, not a regime oracle; keep depth at 1.
"""

from typing import List

import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class _Chomp1d(nn.Module):
    """Trim the right padding so a dilated conv stays CAUSAL (no peeking at future bars)."""

    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, : -self.chomp].contiguous() if self.chomp > 0 else x


class _TemporalBlock(nn.Module):
    """One dilated causal residual conv block (Bai et al. TCN)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
        )
        self.chomp = _Chomp1d(pad)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.out_act = nn.ReLU()

    def forward(self, x):
        out = self.drop(self.act(self.chomp(self.conv(x))))
        res = x if self.downsample is None else self.downsample(x)
        return self.out_act(out + res)


class SequenceFeaturesExtractor(BaseFeaturesExtractor):
    """Reshape the flat obs to a bar sequence and encode it with a temporal model.

    Args:
        observation_space: the env's flat Box space.
        lookback: number of bars per observation (the per_bar width = obs_dim / lookback).
        encoder: 'attn' (self-attention) or 'tcn' (dilated causal conv).
        features_dim: size of the pooled vector handed to the policy MLP.
        d_model / heads: attention width + head count. tcn_channels / tcn_kernel / tcn_levels: TCN shape.
        pool: 'mean' or 'last' over the bar axis.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        lookback: int,
        encoder: str = "attn",
        features_dim: int = 64,
        d_model: int = 48,
        heads: int = 2,
        tcn_channels: int = 24,
        tcn_kernel: int = 3,
        tcn_levels: int = 4,
        pool: str = "mean",
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim)
        obs_dim = int(observation_space.shape[0])
        lookback = max(1, int(lookback))
        if obs_dim % lookback != 0:
            raise ValueError(
                f"SequenceFeaturesExtractor: obs_dim {obs_dim} is not divisible by lookback {lookback} "
                f"— the flattened observation is not a clean [lookback, per_bar] grid."
            )
        if encoder not in ("attn", "tcn"):
            raise ValueError(f"SequenceFeaturesExtractor: unknown encoder '{encoder}' (use 'attn' or 'tcn').")
        if pool not in ("mean", "last"):
            raise ValueError(f"SequenceFeaturesExtractor: unknown pool '{pool}' (use 'mean' or 'last').")

        self.lookback = lookback
        self.per_bar = obs_dim // lookback
        self.encoder = encoder
        self.pool = pool

        if encoder == "attn":
            self.in_proj = nn.Linear(self.per_bar, d_model)
            self.pos = nn.Parameter(th.zeros(1, lookback, d_model))
            self.attn = nn.MultiheadAttention(d_model, num_heads=heads, batch_first=True, dropout=dropout)
            self.norm1 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
            self.norm2 = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, features_dim)
        else:
            blocks: List[nn.Module] = []
            in_ch = self.per_bar
            for level in range(max(1, int(tcn_levels))):
                blocks.append(_TemporalBlock(in_ch, tcn_channels, tcn_kernel, 2**level, dropout))
                in_ch = tcn_channels
            self.tcn = nn.Sequential(*blocks)
            self.head = nn.Linear(tcn_channels, features_dim)

    def _to_sequence(self, observations: th.Tensor) -> th.Tensor:
        """Flat [B, lookback*per_bar] -> [B, lookback, per_bar], matching the env's time-major flatten."""
        return observations.view(observations.shape[0], self.lookback, self.per_bar)

    def _pool(self, seq: th.Tensor) -> th.Tensor:
        """Reduce a [B, lookback, dim] sequence to [B, dim]."""
        return seq.mean(dim=1) if self.pool == "mean" else seq[:, -1, :]

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self._to_sequence(observations)
        if self.encoder == "attn":
            h = self.in_proj(x) + self.pos
            attended, _ = self.attn(h, h, h)
            h = self.norm1(h + attended)
            h = self.norm2(h + self.ffn(h))
            return self.head(self._pool(h))
        # tcn: encode over the bar (time) axis -> channels-major [B, per_bar, lookback]
        encoded = self.tcn(x.transpose(1, 2)).transpose(1, 2)  # -> [B, lookback, channels]
        return self.head(self._pool(encoded))
