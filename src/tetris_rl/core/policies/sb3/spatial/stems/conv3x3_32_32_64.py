# src/tetris_rl/core/policies/sb3/spatial/stems/conv3x3_32_32_64.py
"""
Fixed, paper-exact CNN stem:
  conv3x3-32 -> GELU -> conv3x3-32 -> GELU -> conv3x3-64 -> GELU

Contract:
  SpatialFeatures (B,H,W,C) -> SpatialFeatures (B,H',W',64)

Notes:
- This stem is PRESET. No spec object. Kernel/stride/padding are constants.
- Still spatial (no pooling/flattening).
"""

from __future__ import annotations

from torch import nn

from tetris_rl.core.policies.sb3.api import SpatialFeatures, SpatialSpec


class Conv3x3_32_32_64Stem(nn.Module):
    # authoritative constants (paper-exact)
    KERNEL: int = 3
    STRIDE: int = 1
    PADDING: int = 1
    OUT_CHANNELS: int = 64
    NUM_CONVS: int = 3

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

        k, s, pad = self.KERNEL, self.STRIDE, self.PADDING

        layers: list[nn.Module] = [
            nn.Conv2d(C, 32, kernel_size=k, stride=s, padding=pad, bias=True),
            nn.ReLU(),
            nn.Dropout(p_drop) if p_drop > 0.0 else nn.Identity(),
            nn.Conv2d(32, 32, kernel_size=k, stride=s, padding=pad, bias=True),
            nn.ReLU(),
            nn.Dropout(p_drop) if p_drop > 0.0 else nn.Identity(),
            nn.Conv2d(32, self.out_channels, kernel_size=k, stride=s, padding=pad, bias=True),
            nn.ReLU(),
            nn.Dropout(p_drop) if p_drop > 0.0 else nn.Identity(),
        ]
        self.net = nn.Sequential(*layers)

    def out_spec(self, *, in_spec: SpatialSpec) -> SpatialSpec:
        h = int(in_spec.h)
        w = int(in_spec.w)

        k, s, pad = self.KERNEL, self.STRIDE, self.PADDING

        # Conv2d output size (dilation=1):
        # out = floor((in + 2*pad - k)/s) + 1
        for _ in range(self.NUM_CONVS):
            h = (h + 2 * pad - k) // s + 1
            w = (w + 2 * pad - k) // s + 1
            if h <= 0 or w <= 0:
                raise ValueError(f"invalid spatial size after stem: (h,w)=({h},{w})")

        return SpatialSpec(h=h, w=w, c=self.out_channels, is_discrete_binary=False)

    def forward(self, spatial: SpatialFeatures) -> SpatialFeatures:
        x = spatial.x
        if x.dim() != 4:
            raise ValueError(f"SpatialFeatures.x must be (B,H,W,C), got {tuple(x.shape)}")
        if int(x.shape[-1]) != int(self.in_channels):
            raise ValueError(f"stem expected C={self.in_channels}, got C={int(x.shape[-1])}")

        # channel-last -> channel-first
        x_cf = x.permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)
        y_cf = self.net(x_cf)  # (B,64,H',W')
        y = y_cf.permute(0, 2, 3, 1).contiguous()  # (B,H',W',64)

        return SpatialFeatures(x=y, is_discrete_binary=False)


__all__ = ["Conv3x3_32_32_64Stem"]
