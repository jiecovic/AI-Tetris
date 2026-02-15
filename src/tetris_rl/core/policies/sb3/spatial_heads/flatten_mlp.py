# src/tetris_rl/core/policies/sb3/spatial_heads/flatten_mlp.py
"""
FlattenMLPHead (spatial -> feature vector)

Pipeline:
  (B,H,W,C) -> flatten (B,H*W*C) -> MLP(hidden_dims) -> (B,features_dim)
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from tetris_rl.core.policies.sb3.api import SpatialFeatures, SpatialSpec, Specials
from tetris_rl.core.policies.sb3.layers.activations import make_activation
from tetris_rl.core.policies.sb3.spatial_heads.base import BaseSpatialHead
from tetris_rl.core.policies.sb3.spatial_heads.config import FlattenMLPParams


class FlattenMLPHead(BaseSpatialHead):
    @classmethod
    def infer_auto_features_dim(cls, *, spec: Any, in_spec: SpatialSpec) -> int:
        _ = spec
        return int(in_spec.h) * int(in_spec.w) * int(in_spec.c)

    def __init__(
        self,
        *,
        features_dim: int,
        spec: FlattenMLPParams,
        in_h: int,
        in_w: int,
        in_channels: int,
    ) -> None:
        super().__init__(features_dim=int(features_dim))
        self.spec = spec

        self.in_h = int(in_h)
        self.in_w = int(in_w)
        self.in_channels = int(in_channels)
        if self.in_h <= 0 or self.in_w <= 0 or self.in_channels <= 0:
            raise ValueError(
                f"invalid FlattenMLPHead input shape: (H,W,C)=({self.in_h},{self.in_w},{self.in_channels})"
            )

        self.in_dim = int(self.in_h) * int(self.in_w) * int(self.in_channels)

        hidden = tuple(int(h) for h in (self.spec.hidden_dims or ()))
        if any(int(h) <= 0 for h in hidden):
            raise ValueError(f"hidden_dims must contain only > 0 values, got {hidden}")

        p = float(self.spec.dropout)
        if p < 0.0:
            raise ValueError("dropout must be >= 0")

        dims = [int(self.in_dim), *hidden, int(self.features_dim)]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            d_in = int(dims[i])
            d_out = int(dims[i + 1])
            layers.append(nn.Linear(d_in, d_out, bias=True))
            is_last = i == (len(dims) - 2)
            if not is_last:
                layers.append(make_activation(str(self.spec.activation)))
                if p > 0.0:
                    layers.append(nn.Dropout(p))

        self.net = nn.Sequential(*layers)

    def forward(self, *, spatial: SpatialFeatures, specials: Specials) -> torch.Tensor:
        _ = specials
        x = self._check_spatial(spatial)  # (B,H,W,C)
        B, H, W, C = x.shape
        flat = x.reshape(int(B), int(H) * int(W) * int(C))
        if int(flat.shape[1]) != int(self.in_dim):
            raise ValueError(
                f"FlattenMLPHead input dim changed at runtime: expected {self.in_dim}, got {int(flat.shape[1])}"
            )

        out = self.net(flat)
        return self._check_out(out)


__all__ = ["FlattenMLPHead"]
