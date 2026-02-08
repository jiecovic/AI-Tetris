# src/tetris_rl/core/policies/sb3/tokenizers/layout/row.py
"""
RowTokenizer: EXACTLY one row per token.

Semantics (special case of PatchTokenizer):
  patch_h = 1
  patch_w = W
  stride_h = 1
  stride_w = W  (exactly one patch per row)

Return:
  tokens: (B, H, W*C)
  pos_h:  (H,) row indices 0..H-1
  pos_w:  None
"""

from __future__ import annotations

from typing import Optional

import torch

from tetris_rl.core.policies.sb3.api import SpatialFeatures
from tetris_rl.core.policies.sb3.tokenizers.layout.patch import PatchTokenizer


class RowTokenizer(PatchTokenizer):
    def __init__(self) -> None:
        # dummy; patch_w depends on W at call time
        super().__init__(patch_h=1, patch_w=1, stride_h=1, stride_w=1)

    def __call__(
        self,
        *,
        spatial: SpatialFeatures,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        x = spatial.x
        if x.dim() != 4:
            raise ValueError(f"SpatialFeatures.x must be (B,H,W,C), got {tuple(x.shape)}")
        W = int(x.shape[2])

        tokens, pos_h, _pos_w = self._extract_patches_2d(
            x=x,
            patch_h=1,
            patch_w=W,
            stride_h=1,
            stride_w=W,
        )
        return tokens, pos_h, None


__all__ = ["RowTokenizer"]


