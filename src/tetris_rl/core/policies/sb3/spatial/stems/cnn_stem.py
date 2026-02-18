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

import torch
from torch import nn
from torch.nn import functional as F

from tetris_rl.core.policies.sb3.api import SpatialFeatures, SpatialSpec
from tetris_rl.core.policies.sb3.layers.activations import make_activation
from tetris_rl.core.policies.sb3.spatial.config import CNNStemParams


def _as_int_tuple(xs: Sequence[int] | None) -> tuple[int, ...]:
    if xs is None:
        return ()
    return tuple(int(x) for x in xs)


def _as_hw_pair(value: int | tuple[int, int], *, field: str) -> tuple[int, int]:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be int or (h,w), got bool")
    if isinstance(value, int):
        return (int(value), int(value))
    if len(value) != 2:
        raise ValueError(f"{field} pair must have exactly 2 values")
    h, w = value
    if isinstance(h, bool) or isinstance(w, bool):
        raise ValueError(f"{field} pair values must be ints (bool is not allowed)")
    return (int(h), int(w))


def _as_lrtb(
    value: int | tuple[int, int] | tuple[int, int, int, int],
    *,
    field: str,
) -> tuple[int, int, int, int]:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be int, (h,w), or (l,r,t,b), got bool")
    if isinstance(value, int):
        v = int(value)
        return (v, v, v, v)
    if len(value) == 2:
        h, w = value
        if isinstance(h, bool) or isinstance(w, bool):
            raise ValueError(f"{field} pair values must be ints (bool is not allowed)")
        hh, ww = int(h), int(w)
        return (ww, ww, hh, hh)
    if len(value) == 4:
        left, right, top, bottom = value
        if any(isinstance(x, bool) for x in (left, right, top, bottom)):
            raise ValueError(f"{field} quad values must be ints (bool is not allowed)")
        return (int(left), int(right), int(top), int(bottom))
    raise ValueError(f"{field} must be int, pair, or quad")


class SideValuePad2d(nn.Module):
    """
    Constant pad with side-specific values:
      left, right, top, bottom may use different constants.
    """

    def __init__(
        self,
        *,
        padding: tuple[int, int, int, int],
        values: tuple[float, float, float, float],
    ) -> None:
        super().__init__()
        left, right, top, bottom = padding
        if min(left, right, top, bottom) < 0:
            raise ValueError(f"padding must be >= 0, got {padding}")
        self.padding = (int(left), int(right), int(top), int(bottom))
        self.values = tuple(float(v) for v in values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right, top, bottom = self.padding
        if (left + right + top + bottom) == 0:
            return x

        left_value, right_value, top_value, bottom_value = self.values
        y = F.pad(x, (left, right, top, bottom), mode="constant", value=0.0)

        if left > 0:
            y[..., :, :left] = left_value
        if right > 0:
            y[..., :, -right:] = right_value
        # Top/bottom are written last so their values win at corners.
        if top > 0:
            y[..., :top, :] = top_value
        if bottom > 0:
            y[..., -bottom:, :] = bottom_value

        return y


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
        # list of (kind, (k_h,k_w), (s_h,s_w), (p_h,p_w)), kind in {"conv","pool","prepad"}.
        self._geom_ops: list[tuple[str, tuple[int, int], tuple[int, int], tuple[int, int]]] = []

        layers: list[nn.Module] = []
        c_prev = C_in

        if spec.layers is not None:
            for i, layer in enumerate(spec.layers):
                c_out = int(layer.out)
                k_h, k_w = _as_hw_pair(layer.k, field=f"layers[{i}].k")
                s_h, s_w = _as_hw_pair(layer.s, field=f"layers[{i}].s")
                if layer.p is None:
                    p_h, p_w = (0, 0)
                else:
                    p_h, p_w = _as_hw_pair(layer.p, field=f"layers[{i}].p")
                if c_out <= 0:
                    raise ValueError(f"layers[{i}].out must be > 0, got {c_out}")
                if k_h <= 0 or k_w <= 0:
                    raise ValueError(f"layers[{i}].k values must be > 0, got ({k_h},{k_w})")
                if s_h <= 0 or s_w <= 0:
                    raise ValueError(f"layers[{i}].s values must be > 0, got ({s_h},{s_w})")
                if p_h < 0 or p_w < 0:
                    raise ValueError(f"layers[{i}].p values must be >= 0, got ({p_h},{p_w})")

                pre_pad = getattr(layer, "pre_pad", None)
                if pre_pad is not None:
                    left, right, top, bottom = _as_lrtb(pre_pad.p, field=f"layers[{i}].pre_pad.p")
                    if min(left, right, top, bottom) < 0:
                        raise ValueError(
                            f"layers[{i}].pre_pad.p values must be >= 0, got ({left},{right},{top},{bottom})"
                        )

                    if (left + right + top + bottom) > 0:
                        default = float(pre_pad.value)
                        left_value = float(pre_pad.left_value) if pre_pad.left_value is not None else default
                        right_value = float(pre_pad.right_value) if pre_pad.right_value is not None else default
                        top_value = float(pre_pad.top_value) if pre_pad.top_value is not None else default
                        bottom_value = float(pre_pad.bottom_value) if pre_pad.bottom_value is not None else default

                        vals = (left_value, right_value, top_value, bottom_value)
                        if left_value == right_value == top_value == bottom_value:
                            layers.append(nn.ConstantPad2d((left, right, top, bottom), default))
                        else:
                            layers.append(SideValuePad2d(padding=(left, right, top, bottom), values=vals))
                        self._geom_ops.append(
                            ("prepad", (int(top), int(bottom)), (int(left), int(right)), (0, 0))
                        )

                layers.append(
                    nn.Conv2d(
                        int(c_prev),
                        int(c_out),
                        kernel_size=(int(k_h), int(k_w)),
                        stride=(int(s_h), int(s_w)),
                        padding=(int(p_h), int(p_w)),
                        bias=True,
                    )
                )
                self._geom_ops.append(("conv", (int(k_h), int(k_w)), (int(s_h), int(s_w)), (int(p_h), int(p_w))))

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
                    pk_h, pk_w = _as_hw_pair(layer.pool.k, field=f"layers[{i}].pool.k")
                    ps_h, ps_w = _as_hw_pair(layer.pool.s, field=f"layers[{i}].pool.s")
                    pp_h, pp_w = _as_hw_pair(layer.pool.p, field=f"layers[{i}].pool.p")
                    if pk_h <= 0 or pk_w <= 0:
                        raise ValueError(f"layers[{i}].pool.k values must be > 0, got ({pk_h},{pk_w})")
                    if ps_h <= 0 or ps_w <= 0:
                        raise ValueError(f"layers[{i}].pool.s values must be > 0, got ({ps_h},{ps_w})")
                    if pp_h < 0 or pp_w < 0:
                        raise ValueError(f"layers[{i}].pool.p values must be >= 0, got ({pp_h},{pp_w})")
                    pool_type = str(layer.pool.type).strip().lower()
                    if pool_type == "avg":
                        layers.append(nn.AvgPool2d(kernel_size=(pk_h, pk_w), stride=(ps_h, ps_w), padding=(pp_h, pp_w)))
                    elif pool_type == "max":
                        layers.append(nn.MaxPool2d(kernel_size=(pk_h, pk_w), stride=(ps_h, ps_w), padding=(pp_h, pp_w)))
                    else:
                        raise ValueError(f"layers[{i}].pool.type must be avg|max, got {layer.pool.type!r}")
                    self._geom_ops.append(
                        ("pool", (int(pk_h), int(pk_w)), (int(ps_h), int(ps_w)), (int(pp_h), int(pp_w)))
                    )

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
                self._geom_ops.append(("conv", (int(k), int(k)), (int(s), int(s)), (int(pad), int(pad))))
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
        for kind, (k_h, k_w), (s_h, s_w), (p_h, p_w) in self._geom_ops:
            if kind == "prepad":
                h = h + int(k_h) + int(k_w)  # top + bottom
                w = w + int(s_h) + int(s_w)  # left + right
                if h <= 0 or w <= 0:
                    raise ValueError(f"invalid spatial size after stem pre_pad: (h,w)=({h},{w})")
                continue
            h = (h + 2 * p_h - k_h) // s_h + 1
            w = (w + 2 * p_w - k_w) // s_w + 1
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


