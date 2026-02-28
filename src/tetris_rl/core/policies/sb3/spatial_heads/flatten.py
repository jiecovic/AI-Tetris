# src/tetris_rl/core/policies/sb3/spatial_heads/flatten.py
"""
FlattenHead (spatial -> feature vector)

Baseline:
  (B,H,W,C) -> flatten to (B, H*W*C) -> projection to (B,F)
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from tetris_rl.core.policies.sb3.api import SpatialFeatures, SpatialSpec, Specials
from tetris_rl.core.policies.sb3.spatial_heads.base import BaseSpatialHead
from tetris_rl.core.policies.sb3.spatial_heads.config import FlattenParams


class FlattenHead(BaseSpatialHead):
    @classmethod
    def infer_auto_features_dim(cls, *, spec: Any, in_spec: SpatialSpec) -> int:
        _ = spec
        return int(in_spec.h) * int(in_spec.w) * int(in_spec.c)

    def __init__(
        self,
        *,
        features_dim: int,
        spec: FlattenParams,
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
            raise ValueError(f"invalid FlattenHead input shape: (H,W,C)=({self.in_h},{self.in_w},{self.in_channels})")

        self.in_dim = int(self.in_h) * int(self.in_w) * int(self.in_channels)
        self.proj: nn.Module = (
            nn.Identity() if self.in_dim == self.features_dim else nn.Linear(self.in_dim, self.features_dim)
        )

    def forward(self, *, spatial: SpatialFeatures, specials: Specials) -> torch.Tensor:
        _ = specials
        x = self._check_spatial(spatial)  # (B,H,W,C)
        B, H, W, C = x.shape
        flat = x.reshape(int(B), int(H) * int(W) * int(C))
        if int(flat.shape[1]) != int(self.in_dim):
            raise ValueError(
                f"FlattenHead input dim changed at runtime: expected {self.in_dim}, got {int(flat.shape[1])}"
            )
        out = self.proj(flat)
        return self._check_out(out)


__all__ = ["FlattenHead"]
