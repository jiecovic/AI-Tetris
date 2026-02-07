# src/tetris_rl/core/policies/sb3/spatial_heads/global_pool.py
from __future__ import annotations

"""
GlobalPoolHead (spatial -> feature vector)

(B,H,W,C) -> optional conv2d stack -> global pool over (H,W) -> optional 1-hidden MLP
-> always projects to (B,features_dim)
"""

from typing import Sequence

import torch
from torch import nn

from tetris_rl.core.policies.sb3.spatial_heads.config import GlobalPoolParams, Pool2D
from tetris_rl.core.policies.sb3.api import Specials, SpatialFeatures
from tetris_rl.core.policies.sb3.layers.activations import make_activation
from tetris_rl.core.policies.sb3.spatial_heads.base import BaseSpatialHead


def _as_int_tuple(xs: Sequence[int] | None) -> tuple[int, ...]:
    if xs is None:
        return ()
    return tuple(int(x) for x in xs)


class GlobalPoolHead(BaseSpatialHead):
    def __init__(self, *, in_channels: int, features_dim: int, spec: GlobalPoolParams) -> None:
        super().__init__(features_dim=int(features_dim))

        C_in = int(in_channels)
        if C_in <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}")

        p = float(spec.dropout)
        if p < 0.0:
            raise ValueError("dropout must be >= 0")

        pool = str(spec.pool).strip().lower()
        if pool not in {"avg", "max", "avgmax"}:
            raise ValueError(f"pool must be one of avg|max|avgmax, got {spec.pool!r}")
        self._pool: Pool2D = pool  # type: ignore[assignment]

        self.spec = spec
        self._C_in = C_in

        # ---- optional conv stack (channel-first conv2d) ----
        conv_channels = _as_int_tuple(spec.conv_channels)
        conv_kernels = _as_int_tuple(spec.conv_kernel_sizes)
        if len(conv_channels) != len(conv_kernels):
            raise ValueError("conv_channels and conv_kernel_sizes must have same length")

        strides = _as_int_tuple(spec.conv_strides)
        if len(conv_channels) > 0 and len(strides) == 0:
            strides = tuple(1 for _ in conv_channels)
        if len(strides) not in (0, len(conv_channels)):
            raise ValueError("conv_strides must be empty or have same length as conv_channels")

        layers: list[nn.Module] = []
        c_prev = C_in
        for c_out, k, s in zip(conv_channels, conv_kernels, strides):
            k = int(k)
            s = int(s)
            if k <= 0:
                raise ValueError(f"conv kernel must be > 0, got {k}")

            pad = (k // 2) if (k % 2 == 1) else 0  # "mostly same"
            layers.append(nn.Conv2d(c_prev, int(c_out), kernel_size=k, stride=s, padding=pad, bias=True))
            if bool(spec.use_batchnorm):
                layers.append(nn.BatchNorm2d(int(c_out)))
            layers.append(make_activation(spec.activation))
            if p > 0.0:
                layers.append(nn.Dropout(p))
            c_prev = int(c_out)

        self.conv = nn.Sequential(*layers) if len(layers) > 0 else nn.Identity()
        self._C_after_conv = int(c_prev)

        pooled_dim = int(self._C_after_conv) if self._pool in {"avg", "max"} else int(2 * self._C_after_conv)

        # post always ends at features_dim
        if int(spec.mlp_hidden) > 0:
            Hh = int(spec.mlp_hidden)
            if Hh <= 0:
                raise ValueError("mlp_hidden must be > 0 when enabled")
            self.post = nn.Sequential(
                nn.Linear(pooled_dim, Hh, bias=True),
                make_activation(spec.activation),
                nn.Dropout(p) if p > 0.0 else nn.Identity(),
                nn.Linear(Hh, self.features_dim, bias=True),
            )
        else:
            self.post = nn.Identity() if pooled_dim == self.features_dim else nn.Linear(pooled_dim, self.features_dim, bias=True)

    def forward(self, *, spatial: SpatialFeatures, specials: Specials) -> torch.Tensor:
        _ = specials
        x = self._check_spatial(spatial)  # (B,H,W,C)
        B, H, W, C = x.shape

        if int(C) != int(self._C_in):
            raise ValueError(f"in_channels mismatch: head expects C={self._C_in}, got C={int(C)}")

        x_cf = x.permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)
        y = self.conv(x_cf)  # (B,C',H',W')

        if self._pool == "avg":
            pooled = y.mean(dim=(2, 3))
        elif self._pool == "max":
            pooled = y.amax(dim=(2, 3))
        elif self._pool == "avgmax":
            avg = y.mean(dim=(2, 3))
            mx = y.amax(dim=(2, 3))
            pooled = torch.cat([avg, mx], dim=1)
        else:
            raise ValueError(f"unknown pool mode: {self._pool!r}")

        out = self.post(pooled)
        return self._check_out(out)


__all__ = ["GlobalPoolHead"]


