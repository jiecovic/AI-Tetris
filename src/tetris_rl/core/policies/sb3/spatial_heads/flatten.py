# src/tetris_rl/core/policies/sb3/spatial_heads/flatten.py
"""
FlattenHead (spatial -> feature vector)

Baseline:
  (B,H,W,C) -> flatten to (B, H*W*C) -> (optional) linear projection -> (B,F)

Built lazily because H/W/C can vary by preprocessor/stem.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from tetris_rl.core.policies.sb3.api import SpatialFeatures, Specials
from tetris_rl.core.policies.sb3.spatial_heads.base import BaseSpatialHead
from tetris_rl.core.policies.sb3.spatial_heads.config import FlattenParams


class FlattenHead(BaseSpatialHead):
    def __init__(self, *, features_dim: int, spec: FlattenParams) -> None:
        super().__init__(features_dim=int(features_dim))
        self.spec = spec

        self._in_dim: Optional[int] = None
        self.proj: Optional[nn.Module] = None

    def forward(self, *, spatial: SpatialFeatures, specials: Specials) -> torch.Tensor:
        _ = specials
        x = self._check_spatial(spatial)  # (B,H,W,C)
        B, H, W, C = x.shape
        flat = x.reshape(int(B), int(H) * int(W) * int(C))

        if self._in_dim is None:
            self._in_dim = int(flat.shape[1])

            if bool(self.spec.proj):
                self.proj = nn.Identity() if self._in_dim == self.features_dim else nn.Linear(self._in_dim, self.features_dim)
            else:
                if int(self._in_dim) != int(self.features_dim):
                    raise ValueError(
                        f"FlattenHead proj=False requires features_dim == H*W*C, got features_dim={self.features_dim} vs {self._in_dim}"
                    )
                self.proj = nn.Identity()

            self.proj = self.proj.to(device=flat.device)

        assert self.proj is not None
        out = self.proj(flat)
        return self._check_out(out)


__all__ = ["FlattenHead"]


