# src/tetris_rl/core/policies/sb3/tokenizers/layout/column.py
"""
ColumnTokenizer: EXACTLY one column per token.

Semantics (special case of PatchTokenizer):
  patch_h = H
  patch_w = 1
  stride_h = H  (exactly one patch per column)
  stride_w = 1

Return:
  tokens: (B, W, H*C)
  pos_h:  None
  pos_w:  (W,) col indices 0..W-1
"""

from __future__ import annotations

from typing import Optional

import torch

from tetris_rl.core.policies.sb3.api import SpatialFeatures
from tetris_rl.core.policies.sb3.tokenizers.layout.patch import PatchTokenizer


class ColumnTokenizer(PatchTokenizer):
    def __init__(self) -> None:
        # dummy; patch_h depends on H at call time
        super().__init__(patch_h=1, patch_w=1, stride_h=1, stride_w=1)

    def __call__(
            self,
            *,
            spatial: SpatialFeatures,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        x = spatial.x
        if x.dim() != 4:
            raise ValueError(f"SpatialFeatures.x must be (B,H,W,C), got {tuple(x.shape)}")
        H = int(x.shape[1])

        tokens, _pos_h, pos_w = self._extract_patches_2d(
            x=x,
            patch_h=H,
            patch_w=1,
            stride_h=H,
            stride_w=1,
        )
        return tokens, None, pos_w


__all__ = ["ColumnTokenizer"]


