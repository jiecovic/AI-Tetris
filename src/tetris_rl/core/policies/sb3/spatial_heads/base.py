# src/tetris_rl/core/policies/sb3/spatial_heads/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from tetris_rl.core.policies.sb3.api import SpatialFeatures, SpatialSpec


class BaseSpatialHead(nn.Module, ABC):
    """
    Tiny base class for SpatialHead implementations.

    Provides:
      - features_dim storage
      - input/output shape validation helpers
    """

    def __init__(self, *, features_dim: int) -> None:
        super().__init__()
        F = int(features_dim)
        if F <= 0:
            raise ValueError(f"features_dim must be > 0, got {features_dim}")
        self.features_dim: int = F

    @classmethod
    @abstractmethod
    def infer_auto_features_dim(cls, *, spec: Any, in_spec: SpatialSpec) -> int:
        """
        Compute output feature width for `features_dim='auto'`.
        """

    @staticmethod
    def _check_spatial(spatial: SpatialFeatures) -> torch.Tensor:
        x = spatial.x
        if x.dim() != 4:
            raise ValueError(f"SpatialFeatures.x must be (B,H,W,C), got {tuple(x.shape)}")
        B, H, W, C = x.shape
        if int(B) <= 0 or int(H) <= 0 or int(W) <= 0 or int(C) <= 0:
            raise ValueError(f"invalid SpatialFeatures.x shape {tuple(x.shape)}")
        if not x.is_floating_point():
            raise ValueError("SpatialFeatures.x must be floating point (e.g. float32)")
        return x

    def _check_out(self, out: torch.Tensor) -> torch.Tensor:
        if out.dim() != 2:
            raise ValueError(f"SpatialHead must return (B,F), got {tuple(out.shape)}")
        if int(out.shape[1]) != int(self.features_dim):
            raise ValueError(f"expected features_dim={self.features_dim}, got F={int(out.shape[1])}")
        return out


__all__ = ["BaseSpatialHead"]


