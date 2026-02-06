# src/tetris_rl/models/spatial/stems/conv1x3_32x4_64_5l.py
from __future__ import annotations

"""
Fixed CNN stem (horizontal-only):
  (conv1x3-32 -> ReLU -> Dropout) × 4
  conv1x3-64 -> ReLU -> Dropout

Contract:
  SpatialFeatures (B,H,W,C) -> SpatialFeatures (B,H',W',64)

Notes:
- This stem is PRESET. No spec object. Kernel/stride/padding are constants.
- Horizontal mixing only (kernel=(1,3)), no vertical mixing.
- Still spatial (no pooling/flattening).
"""

from torch import nn

from tetris_rl.models.api import SpatialFeatures, SpatialSpec


class Conv1x3_32x4_64_5LStem(nn.Module):
    # authoritative constants
    KERNEL: tuple[int, int] = (1, 3)
    STRIDE: int = 1
    PADDING: tuple[int, int] = (0, 1)  # "same" for 1x3
    OUT_CHANNELS: int = 64
    NUM_CONVS: int = 5

    HIDDEN_CHANNELS: int = 32  # internal width

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

        k = self.KERNEL
        s = self.STRIDE
        pad = self.PADDING
        hid = int(self.HIDDEN_CHANNELS)

        layers: list[nn.Module] = []

        # 4 hidden convs @ 32 channels
        in_ch = C
        for _ in range(4):
            layers += [
                nn.Conv2d(in_ch, hid, kernel_size=k, stride=s, padding=pad, bias=True),
                nn.ReLU(),
                nn.Dropout(p_drop) if p_drop > 0.0 else nn.Identity(),
            ]
            in_ch = hid

        # final conv to OUT_CHANNELS (64)
        layers += [
            nn.Conv2d(hid, self.out_channels, kernel_size=k, stride=s, padding=pad, bias=True),
            nn.ReLU(),
            nn.Dropout(p_drop) if p_drop > 0.0 else nn.Identity(),
        ]

        self.net = nn.Sequential(*layers)

    def out_spec(self, *, in_spec: SpatialSpec) -> SpatialSpec:
        h = int(in_spec.h)
        w = int(in_spec.w)

        (kh, kw) = self.KERNEL
        s = int(self.STRIDE)
        (ph, pw) = self.PADDING

        # Conv2d output size (dilation=1):
        # out = floor((in + 2*pad - k)/s) + 1
        for _ in range(self.NUM_CONVS):
            h = (h + 2 * ph - kh) // s + 1
            w = (w + 2 * pw - kw) // s + 1
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


__all__ = ["Conv1x3_32x4_64_5LStem"]
