# src/tetris_rl/core/policies/sb3/spatial/stems/cnn_stem.py
"""
Generic configurable CNN stem (spatial -> spatial).

A stem:
- consumes SpatialFeatures
- produces SpatialFeatures
- preserves grid structure (no pooling / flattening)

This version supports FULL per-layer configuration:
- channels
- kernel sizes
- strides
- activation
- batchnorm
- dropout
"""

from __future__ import annotations

from typing import Sequence

from torch import nn

from tetris_rl.core.policies.sb3.api import SpatialFeatures, SpatialSpec
from tetris_rl.core.policies.sb3.layers.activations import make_activation
from tetris_rl.core.policies.sb3.spatial.config import CNNStemParams


def _as_int_tuple(xs: Sequence[int] | None) -> tuple[int, ...]:
    if xs is None:
        return ()
    return tuple(int(x) for x in xs)


class CNNStem(nn.Module):
    """
    Fully generic CNN stem.

    Input:
      SpatialFeatures.x : (B,H,W,C) channel-last

    Output:
      SpatialFeatures.x : (B,H',W',C') channel-last
      is_discrete_binary = False
    """

    def __init__(self, *, in_channels: int, spec: CNNStemParams) -> None:
        super().__init__()

        C_in = int(in_channels)
        if C_in <= 0:
            raise ValueError(f"in_channels must be > 0, got {C_in}")
        self.in_channels: int = C_in

        channels = _as_int_tuple(spec.channels)
        kernels = _as_int_tuple(spec.kernel_sizes)

        if len(channels) == 0:
            raise ValueError("spec.channels must be non-empty")
        if len(channels) != len(kernels):
            raise ValueError("channels and kernel_sizes must have same length")

        strides = _as_int_tuple(spec.strides)
        if len(strides) == 0:
            strides = tuple(1 for _ in channels)
        if len(strides) != len(channels):
            raise ValueError("strides must have same length as channels")

        for k in kernels:
            if k <= 0 or k % 2 == 0:
                raise ValueError(f"kernel sizes must be odd and >0, got {k}")

        if float(spec.dropout) < 0.0:
            raise ValueError("dropout must be >= 0")
        p = float(spec.dropout)

        # geometry metadata (authoritative for out_spec)
        self._kernels: tuple[int, ...] = kernels
        self._strides: tuple[int, ...] = strides
        self._paddings: tuple[int, ...] = tuple(k // 2 for k in kernels)

        layers: list[nn.Module] = []
        c_prev = C_in

        for c_out, k, s, pad in zip(channels, kernels, strides, self._paddings):
            layers.append(
                nn.Conv2d(
                    int(c_prev),
                    int(c_out),
                    kernel_size=int(k),
                    stride=int(s),
                    padding=int(pad),
                    bias=True,
                )
            )
            if bool(spec.use_batchnorm):
                layers.append(nn.BatchNorm2d(int(c_out)))
            layers.append(make_activation(spec.activation))
            if p > 0.0:
                layers.append(nn.Dropout(p))
            c_prev = int(c_out)

        self.net = nn.Sequential(*layers)
        self.out_channels: int = int(c_prev)

    def out_spec(self, *, in_spec: SpatialSpec) -> SpatialSpec:
        h = int(in_spec.h)
        w = int(in_spec.w)

        # Conv2d output size (dilation=1):
        # out = floor((in + 2*pad - k)/s) + 1
        for k, s, pad in zip(self._kernels, self._strides, self._paddings):
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
        x_cf = x.permute(0, 3, 1, 2).contiguous()
        y_cf = self.net(x_cf)
        y = y_cf.permute(0, 2, 3, 1).contiguous()

        return SpatialFeatures(x=y, is_discrete_binary=False)


__all__ = ["CNNStem"]


