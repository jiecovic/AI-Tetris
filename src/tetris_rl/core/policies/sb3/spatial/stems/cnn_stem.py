# src/tetris_rl/core/policies/sb3/spatial/stems/cnn_stem.py
"""
Generic configurable CNN stem (spatial -> spatial).

A stem:
- consumes SpatialFeatures
- produces SpatialFeatures
- preserves grid structure (no pooling / flattening)

Supports two config styles:
- Legacy tuples: channels / kernel_sizes / strides + global activation/bn/dropout
- Explicit layers: [{out,k,s,p,act,pool}, ...] + global bn/dropout
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

        if float(spec.dropout) < 0.0:
            raise ValueError("dropout must be >= 0")

        # Geometry metadata (authoritative for out_spec):
        # list of (kind, kernel, stride, padding), where kind is "conv" or "pool".
        self._geom_ops: list[tuple[str, int, int, int]] = []

        layers: list[nn.Module] = []
        c_prev = C_in

        if spec.layers is not None:
            for i, layer in enumerate(spec.layers):
                c_out = int(layer.out)
                k = int(layer.k)
                s = int(layer.s)
                pad = int(layer.p) if layer.p is not None else 0
                if c_out <= 0:
                    raise ValueError(f"layers[{i}].out must be > 0, got {c_out}")
                if k <= 0:
                    raise ValueError(f"layers[{i}].k must be > 0, got {k}")
                if s <= 0:
                    raise ValueError(f"layers[{i}].s must be > 0, got {s}")
                if pad < 0:
                    raise ValueError(f"layers[{i}].p must be >= 0, got {pad}")

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
                self._geom_ops.append(("conv", int(k), int(s), int(pad)))

                if bool(spec.use_batchnorm):
                    layers.append(nn.BatchNorm2d(int(c_out)))

                if layer.act is None:
                    raise ValueError(f"layers[{i}].act is required when using explicit layers")
                act_name = str(layer.act)
                layers.append(make_activation(act_name))

                p_drop = float(spec.dropout)
                if p_drop < 0.0:
                    raise ValueError(f"dropout must be >= 0, got {p_drop}")
                if p_drop > 0.0:
                    layers.append(nn.Dropout(p_drop))

                if layer.pool is not None:
                    pk = int(layer.pool.k)
                    ps = int(layer.pool.s)
                    pp = int(layer.pool.p)
                    if pk <= 0:
                        raise ValueError(f"layers[{i}].pool.k must be > 0, got {pk}")
                    if ps <= 0:
                        raise ValueError(f"layers[{i}].pool.s must be > 0, got {ps}")
                    if pp < 0:
                        raise ValueError(f"layers[{i}].pool.p must be >= 0, got {pp}")
                    pool_type = str(layer.pool.type).strip().lower()
                    if pool_type == "avg":
                        layers.append(nn.AvgPool2d(kernel_size=pk, stride=ps, padding=pp))
                    elif pool_type == "max":
                        layers.append(nn.MaxPool2d(kernel_size=pk, stride=ps, padding=pp))
                    else:
                        raise ValueError(f"layers[{i}].pool.type must be avg|max, got {layer.pool.type!r}")
                    self._geom_ops.append(("pool", int(pk), int(ps), int(pp)))

                c_prev = int(c_out)
        else:
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

            for i, (c_out, k, s) in enumerate(zip(channels, kernels, strides)):
                if k <= 0 or k % 2 == 0:
                    raise ValueError(f"legacy kernel_sizes must be odd and >0, got {k} at index {i}")
                pad = int(k // 2)
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
                self._geom_ops.append(("conv", int(k), int(s), int(pad)))
                if bool(spec.use_batchnorm):
                    layers.append(nn.BatchNorm2d(int(c_out)))
                if spec.activation is None:
                    raise ValueError("legacy mode requires activation")
                layers.append(make_activation(str(spec.activation)))
                if float(spec.dropout) > 0.0:
                    layers.append(nn.Dropout(float(spec.dropout)))
                c_prev = int(c_out)

        self.net = nn.Sequential(*layers)
        self.out_channels: int = int(c_prev)

    def out_spec(self, *, in_spec: SpatialSpec) -> SpatialSpec:
        h = int(in_spec.h)
        w = int(in_spec.w)

        # Conv2d / Pool2d output size (dilation=1):
        # out = floor((in + 2*pad - k)/s) + 1
        for _kind, k, s, pad in self._geom_ops:
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


