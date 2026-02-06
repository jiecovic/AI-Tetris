# src/tetris_rl/models/spatial/stems/conv3x3_32_32_64_row1_col2_128.py
from __future__ import annotations

"""
Preset CNN stem for 20x10-ish boards (Tetris):

  conv3x3-32 -> ReLU -> (drop)
  conv3x3-32 -> ReLU -> (drop)
  conv3x3-64 -> ReLU -> (drop)
  conv1x3-64  -> ReLU -> (drop)    # row-wise mixing
  conv3x1-64  -> ReLU -> (drop)    # col-wise mixing
  conv3x1-64  -> ReLU -> (drop)    # col-wise mixing
  conv1x1-128 -> ReLU -> (drop)    # channel expansion only

Contract:
  SpatialFeatures (B,H,W,C) -> SpatialFeatures (B,H,W,128)

Notes:
- All convs are stride=1 with "same" padding per kernel:
  - 3x3  => pad=(1,1)
  - 1x3  => pad=(0,1)
  - 3x1  => pad=(1,0)
  - 1x1  => pad=(0,0)
- Still spatial (no pooling/flattening).
- This stem is PRESET. No spec object.
"""

from torch import nn

from tetris_rl.models.api import SpatialFeatures, SpatialSpec


class Conv3x3_32_32_64Row1Col2_128Stem(nn.Module):
    OUT_CHANNELS: int = 128

    def __init__(self, *, in_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        C = int(in_channels)
        if C <= 0:
            raise ValueError(f"in_channels must be > 0, got {C}")

        p_drop = float(dropout)
        if p_drop < 0.0:
            raise ValueError(f"dropout must be >= 0, got {p_drop}")

        self.in_channels: int = C
        self.out_channels: int = int(self.OUT_CHANNELS)

        def _drop() -> nn.Module:
            return nn.Dropout(p_drop) if p_drop > 0.0 else nn.Identity()

        layers: list[nn.Module] = [
            # --- 2D mixing (32/32/64) ---
            nn.Conv2d(C, 32, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=True),
            nn.ReLU(),
            _drop(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=True),
            nn.ReLU(),
            _drop(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=True),
            nn.ReLU(),
            _drop(),
            # --- row-wise mixing ---
            nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            nn.ReLU(),
            _drop(),
            # --- col-wise mixing ---
            nn.Conv2d(64, 64, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=True),
            nn.ReLU(),
            _drop(),
            # nn.Conv2d(64, self.out_channels, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=True),
            # nn.ReLU(),
            # _drop(),
            # --- channel expansion ---
            nn.Conv2d(64, self.out_channels, kernel_size=(1, 1), stride=1, padding=(0, 0), bias=True),
            nn.ReLU(),
            # _drop(),
        ]

        self.net = nn.Sequential(*layers)

    def out_spec(self, *, in_spec: SpatialSpec) -> SpatialSpec:
        h = int(in_spec.h)
        w = int(in_spec.w)
        if h <= 0 or w <= 0:
            raise ValueError(f"invalid SpatialSpec size: (h,w)=({h},{w})")

        # All convs are stride=1 with "same" padding per kernel => spatial size unchanged.
        return SpatialSpec(h=h, w=w, c=self.out_channels, is_discrete_binary=False)

    def forward(self, spatial: SpatialFeatures) -> SpatialFeatures:
        x = spatial.x
        if x.dim() != 4:
            raise ValueError(f"SpatialFeatures.x must be (B,H,W,C), got {tuple(x.shape)}")
        if int(x.shape[-1]) != int(self.in_channels):
            raise ValueError(f"stem expected C={self.in_channels}, got C={int(x.shape[-1])}")

        # channel-last -> channel-first
        x_cf = x.permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)
        y_cf = self.net(x_cf)  # (B,128,H,W)
        y = y_cf.permute(0, 2, 3, 1).contiguous()  # (B,H,W,128)

        return SpatialFeatures(x=y, is_discrete_binary=False)


__all__ = ["Conv3x3_32_32_64Row1Col2_128Stem"]
