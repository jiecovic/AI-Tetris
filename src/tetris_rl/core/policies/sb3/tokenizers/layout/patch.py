# src/tetris_rl/core/policies/sb3/tokenizers/layout/patch.py
"""
PatchTokenizer: general 2D patches with (patch_h, patch_w) and (stride_h, stride_w).

- No padding (only full patches).
- Row-major order over patch grid:
    for r in rows:
      for c in cols:
        emit patch starting at (r, c)

Return:
  tokens: (B,T, patch_h*patch_w*C)
  pos_h:  (T,) token-grid row indices (0..T_h-1)
  pos_w:  (T,) token-grid col indices (0..T_w-1)
"""

from __future__ import annotations

from typing import Optional

import torch

from tetris_rl.core.policies.sb3.api import SpatialFeatures


class PatchTokenizer:
    def __init__(
            self,
            *,
            patch_h: int,
            patch_w: int,
            stride_h: int | None = None,
            stride_w: int | None = None,
    ) -> None:
        self.patch_h = int(patch_h)
        self.patch_w = int(patch_w)
        self.stride_h = int(stride_h) if stride_h is not None else int(patch_h)
        self.stride_w = int(stride_w) if stride_w is not None else int(patch_w)

        if self.patch_h <= 0 or self.patch_w <= 0:
            raise ValueError("patch_h and patch_w must be > 0")
        if self.stride_h <= 0 or self.stride_w <= 0:
            raise ValueError("stride_h and stride_w must be > 0")

    def _extract_patches_2d(
            self,
            *,
            x: torch.Tensor,  # (B,H,W,C)
            patch_h: int,
            patch_w: int,
            stride_h: int,
            stride_w: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if x.dim() != 4:
            raise ValueError(f"SpatialFeatures.x must be (B,H,W,C), got {tuple(x.shape)}")

        B, H, W, C = x.shape
        ph, pw = int(patch_h), int(patch_w)
        sh, sw = int(stride_h), int(stride_w)

        if ph <= 0 or pw <= 0:
            raise ValueError("patch_h and patch_w must be > 0")
        if sh <= 0 or sw <= 0:
            raise ValueError("stride_h and stride_w must be > 0")
        if H < ph or W < pw:
            raise ValueError(f"patch ({ph},{pw}) cannot exceed grid ({H},{W})")

        # x_h:  (B, T_h, W, C, ph)
        x_h = x.unfold(dimension=1, size=ph, step=sh)
        # x_hw: (B, T_h, T_w, C, ph, pw)
        x_hw = x_h.unfold(dimension=2, size=pw, step=sw)

        # patches: (B, T_h, T_w, ph, pw, C)
        patches = x_hw.permute(0, 1, 2, 4, 5, 3).contiguous()

        T_h = patches.shape[1]
        T_w = patches.shape[2]
        T = T_h * T_w

        tokens = patches.reshape(B, T, ph * pw * C)

        rows = torch.arange(0, T_h, step=1, device=x.device, dtype=torch.int64)
        cols = torch.arange(0, T_w, step=1, device=x.device, dtype=torch.int64)
        grid_h, grid_w = torch.meshgrid(rows, cols, indexing="ij")
        pos_h = grid_h.reshape(T)
        pos_w = grid_w.reshape(T)

        return tokens, pos_h, pos_w

    def __call__(
            self,
            *,
            spatial: SpatialFeatures,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._extract_patches_2d(
            x=spatial.x,
            patch_h=self.patch_h,
            patch_w=self.patch_w,
            stride_h=self.stride_h,
            stride_w=self.stride_w,
        )


__all__ = ["PatchTokenizer"]


