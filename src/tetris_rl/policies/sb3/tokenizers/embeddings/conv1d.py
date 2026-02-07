# src/tetris_rl/policies/sb3/tokenizers/embeddings/conv1d.py
from __future__ import annotations

"""
Conv1D stripe embedder (tokenizer-internal).

Input:
  stripes: (B, T, L, C) float, channel-last
Output:
  emb:     (B, T, D) float
Pooling:
  mean over length after the conv stack

Padding modes
-------------
- valid: no padding
- same:  conv padding=k//2 (odd kernels only)
- tetris: semantic constant padding before EACH Conv1d layer (odd kernels only):
    - row stripes: left/right filled with WALL (= 1.0)
    - col stripes: left filled with AIR (= 0.0), right filled with FLOOR (= 1.0)

CoordConv
---------
Optional: append a 1D coordinate channel to the stripe input:
  - row stripes: coord is x-position along columns (W axis)
  - col stripes: coord is y-position along rows (H axis)

Coord is normalized to [0,1] to match binary board scale.

Notes
-----
- Tetris padding fill values are fixed (not user-configurable).
- Module is agnostic to input range / channel count, but requires floating dtype.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Literal

import torch
from torch import nn
from torch.nn import functional as F

from tetris_rl.policies.sb3.tokenizers.config import Conv1DEmbedParams, PaddingMode

StripeKind = Literal["row", "col"]

# ---------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ProfileContext:
    in_ch: int
    d_model: int
    dropout: float = 0.0


ProfileFn = Callable[[ProfileContext], list[nn.Module]]


def _drop(p: float) -> nn.Module:
    return nn.Dropout(p) if p > 0.0 else nn.Identity()


def tiny(ctx: ProfileContext) -> list[nn.Module]:
    p = float(ctx.dropout)
    return [
        nn.Conv1d(ctx.in_ch, ctx.d_model, kernel_size=3, stride=1, padding=0, bias=True),
        nn.GELU(),
        _drop(p),
    ]


def base(ctx: ProfileContext) -> list[nn.Module]:
    D = int(ctx.d_model)
    p = float(ctx.dropout)
    return [
        nn.Conv1d(ctx.in_ch, D, kernel_size=3, stride=1, padding=0, bias=True),
        nn.GELU(),
        _drop(p),
        nn.Conv1d(D, D, kernel_size=3, stride=1, padding=0, bias=True),
        nn.GELU(),
        _drop(p),
    ]


def deep(ctx: ProfileContext) -> list[nn.Module]:
    D = int(ctx.d_model)
    c = max(1, D // 2)
    p = float(ctx.dropout)
    return [
        nn.Conv1d(ctx.in_ch, c, kernel_size=3, stride=1, padding=0, bias=True),
        nn.ReLU(inplace=False),
        _drop(p),
        nn.Conv1d(c, c, kernel_size=3, stride=1, padding=0, bias=True),
        nn.ReLU(inplace=False),
        _drop(p),
        nn.Conv1d(c, D, kernel_size=3, stride=1, padding=0, bias=True),
        nn.ReLU(inplace=False),
        _drop(p),
    ]

def deep_l5(ctx: ProfileContext) -> list[nn.Module]:
    D = int(ctx.d_model)
    p = float(ctx.dropout)

    c = max(1, D // 8)

    c1 = c
    c2 = min(D, 2 * c1)
    c3 = min(D, 2 * c2)
    c4 = min(D, 2 * c3)

    return [
        nn.Conv1d(ctx.in_ch, c1, kernel_size=3, stride=1, padding=0, bias=True),
        nn.ReLU(inplace=False),
        _drop(p),

        nn.Conv1d(c1, c2, kernel_size=3, stride=1, padding=0, bias=True),
        nn.ReLU(inplace=False),
        _drop(p),

        nn.Conv1d(c2, c3, kernel_size=3, stride=1, padding=0, bias=True),
        nn.ReLU(inplace=False),
        _drop(p),

        nn.Conv1d(c3, c4, kernel_size=3, stride=1, padding=0, bias=True),
        nn.ReLU(inplace=False),
        _drop(p),

        nn.Conv1d(c4, D, kernel_size=3, stride=1, padding=0, bias=True),
        nn.ReLU(inplace=False),
        _drop(p),
    ]





CONV1D_PRESETS: Dict[str, ProfileFn] = {
    "tiny": tiny,
    "base": base,
    "deep": deep,
    "deep_l5": deep_l5,
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _as_int_tuple(xs: Sequence[int] | None) -> tuple[int, ...]:
    if xs is None:
        return ()
    return tuple(int(x) for x in xs)


def _make_activation(name: str) -> nn.Module:
    n = str(name).strip().lower()
    if n == "gelu":
        return nn.GELU()
    if n == "relu":
        return nn.ReLU(inplace=False)
    if n in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError(f"unknown activation: {name!r}")


def _validate_stripes(stripes: torch.Tensor) -> tuple[int, int, int, int]:
    if stripes.dim() != 4:
        raise ValueError(f"stripes must be (B,T,L,C), got {tuple(stripes.shape)}")
    B, T, L, C = stripes.shape
    if B <= 0 or T <= 0 or L <= 0 or C <= 0:
        raise ValueError(f"invalid stripes shape (B,T,L,C)={tuple(stripes.shape)}")
    if not stripes.is_floating_point():
        raise ValueError("stripes must be floating point (e.g. float32)")
    return int(B), int(T), int(L), int(C)


def _to_conv1d_input(stripes: torch.Tensor) -> torch.Tensor:
    # (B,T,L,C) -> (B*T, C, L)
    B, T, L, C = _validate_stripes(stripes)
    return stripes.reshape(B * T, L, C).permute(0, 2, 1).contiguous()


def _from_conv1d_output(y: torch.Tensor, *, B: int, T: int) -> torch.Tensor:
    # (B*T, D) -> (B, T, D)
    return y.reshape(B, T, -1)


def _pad_lr_scalar(
        x: torch.Tensor,
        *,
        pad_left: int,
        pad_right: int,
        left_val: float,
        right_val: float,
) -> torch.Tensor:
    """
    x: (N, C, L)
    returns: (N, C, L+pad_left+pad_right)

    Uses F.pad when fill is symmetric; otherwise uses concat with torch.full.
    """
    if pad_left < 0 or pad_right < 0:
        raise ValueError("pad sizes must be >= 0")
    if pad_left == 0 and pad_right == 0:
        return x

    if float(left_val) == float(right_val):
        return F.pad(x, (pad_left, pad_right), mode="constant", value=float(left_val))

    N, C, _L = x.shape
    device, dtype = x.device, x.dtype

    left = torch.full((N, C, pad_left), float(left_val), device=device, dtype=dtype) if pad_left > 0 else None
    right = torch.full((N, C, pad_right), float(right_val), device=device, dtype=dtype) if pad_right > 0 else None

    if left is None:
        return torch.cat([x, right], dim=-1)  # type: ignore[arg-type]
    if right is None:
        return torch.cat([left, x], dim=-1)
    return torch.cat([left, x, right], dim=-1)


def _append_coord_channel(stripes: torch.Tensor) -> torch.Tensor:
    """
    stripes: (B,T,L,C) -> (B,T,L,C+1)

    Adds a coordinate channel normalized to [0,1] along the length axis L.
    Broadcasted across batch B and tokens T.
    """
    B, T, L, _C = _validate_stripes(stripes)
    device = stripes.device
    dtype = stripes.dtype

    if L == 1:
        coord = torch.zeros((1, 1, 1, 1), device=device, dtype=dtype)
    else:
        coord_1d = torch.linspace(0.0, 1.0, steps=L, device=device, dtype=dtype)  # (L,)
        coord = coord_1d.view(1, 1, L, 1)

    coord = coord.expand(B, T, L, 1).contiguous()
    return torch.cat([stripes, coord], dim=-1)


# ---------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------


class Conv1DEmbedder(nn.Module):
    """
    Conv1D stripe embedder.

    - Builds a stack from either:
        * preset function (CONV1D_PRESETS)
        * generic params (preset == "generic")
    - Optional CoordConv (append coord channel to stripes)
    - Applies optional padding mode (valid/same/tetris)
    - Mean-pools over length
    - Ensures output dim == d_model via final projection if needed

    For padding="tetris", forward requires kind="row"|"col".
    """

    def __init__(self, *, in_channels: int, d_model: int, params: Conv1DEmbedParams) -> None:
        super().__init__()

        C_in = int(in_channels)
        if C_in <= 0:
            raise ValueError(f"in_channels must be > 0, got {C_in}")

        D = int(d_model)
        if D <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")

        p = float(params.dropout)
        if p < 0.0:
            raise ValueError("dropout must be >= 0")

        preset = str(params.preset).strip().lower()
        pad_mode = str(params.padding).strip().lower()
        if pad_mode not in {"valid", "same", "tetris"}:
            raise ValueError(f"padding must be one of valid|same|tetris, got {params.padding!r}")
        self._pad_mode: PaddingMode = pad_mode  # type: ignore[assignment]

        self.params = params
        self.out_dim: int = D

        # CoordConv increases input channels by +1
        self._coordconv: bool = bool(params.coordconv)
        C_eff = C_in + (1 if self._coordconv else 0)

        if preset == "generic":
            layers, last_ch = self._build_generic_layers(in_channels=C_eff, params=params)
        else:
            if preset not in CONV1D_PRESETS:
                have = sorted(CONV1D_PRESETS.keys()) + ["generic"]
                raise ValueError(f"unknown conv1d preset: {preset!r} (have {have})")
            ctx = ProfileContext(in_ch=C_eff, d_model=D, dropout=p)
            layers = CONV1D_PRESETS[preset](ctx)
            last_ch = self._infer_last_out_channels_from_layers(layers)

        self.net = nn.Sequential(*layers)
        self.proj = nn.Linear(int(last_ch), D, bias=True) if int(last_ch) != D else nn.Identity()

    @staticmethod
    def _infer_last_out_channels_from_layers(layers: Sequence[nn.Module]) -> int:
        last_out: int | None = None
        for m in layers:
            if isinstance(m, nn.Conv1d):
                last_out = int(m.out_channels)
        if last_out is None:
            raise ValueError("conv1d stack must contain at least one nn.Conv1d layer")
        return int(last_out)

    def _build_generic_layers(self, *, in_channels: int, params: Conv1DEmbedParams) -> tuple[list[nn.Module], int]:
        channels = _as_int_tuple(params.channels)
        kernels = _as_int_tuple(params.kernel_sizes)
        if len(channels) == 0 or len(kernels) == 0:
            raise ValueError("preset='generic' requires channels and kernel_sizes")
        if len(channels) != len(kernels):
            raise ValueError("channels and kernel_sizes must have same length")

        strides = _as_int_tuple(params.strides)
        if len(strides) == 0:
            strides = tuple(1 for _ in channels)
        if len(strides) != len(channels):
            raise ValueError("strides must have same length as channels")

        p = float(params.dropout)

        layers: list[nn.Module] = []
        c_prev = int(in_channels)

        for c_out, k, s in zip(channels, kernels, strides):
            k = int(k)
            s = int(s)
            c_out = int(c_out)

            if k <= 0:
                raise ValueError(f"kernel sizes must be > 0, got {k}")

            if self._pad_mode in {"same", "tetris"} and (k % 2 == 0):
                raise ValueError(f"kernel sizes must be odd for padding='{self._pad_mode}', got {k}")

            # - same: conv padding=k//2
            # - valid/tetris: conv padding=0 (tetris pads manually)
            conv_pad = (k // 2) if self._pad_mode == "same" else 0

            layers.append(
                nn.Conv1d(
                    c_prev,
                    c_out,
                    kernel_size=k,
                    stride=s,
                    padding=conv_pad,
                    bias=True,
                )
            )
            if bool(params.use_batchnorm):
                layers.append(nn.BatchNorm1d(c_out))
            layers.append(_make_activation(params.activation))
            if p > 0.0:
                layers.append(nn.Dropout(p))

            c_prev = c_out

        return layers, int(c_prev)

    def forward(self, stripes: torch.Tensor, *, kind: StripeKind | None = None) -> torch.Tensor:
        """
        stripes: (B,T,L,C) channel-last float
        kind: required if padding="tetris"
        returns: (B,T,D)
        """
        B, T, _L, _C = _validate_stripes(stripes)

        if self._coordconv:
            stripes = _append_coord_channel(stripes)

        x = _to_conv1d_input(stripes)  # (B*T, C(+1), L)

        if self._pad_mode == "tetris":
            if kind not in {"row", "col"}:
                raise ValueError("kind must be 'row' or 'col' when padding='tetris'")
            x = self._forward_tetris(x, kind=kind)
        else:
            x = self.net(x)

        x = x.mean(dim=-1)  # (B*T, C_last)
        x = self.proj(x)  # (B*T, D)
        return _from_conv1d_output(x, B=B, T=T)

    def _forward_tetris(self, x: torch.Tensor, *, kind: StripeKind) -> torch.Tensor:
        """
        Semantic padding before each Conv1d layer (odd kernels only).

        - row: left/right = 1.0
        - col: left = 0.0, right = 1.0
        """
        left_val, right_val = (1.0, 1.0) if kind == "row" else (0.0, 1.0)

        for m in self.net:
            if isinstance(m, nn.Conv1d):
                k = int(m.kernel_size[0])
                if k % 2 == 0:
                    raise ValueError("tetris padding requires odd kernel sizes")
                if int(m.padding[0]) != 0:
                    raise ValueError("tetris padding requires conv padding=0 (internal invariant)")

                pad = k // 2
                x = _pad_lr_scalar(x, pad_left=pad, pad_right=pad, left_val=left_val, right_val=right_val)
                x = m(x)
            else:
                x = m(x)

        return x


__all__ = [
    "ProfileContext",
    "ProfileFn",
    "CONV1D_PRESETS",
    "Conv1DEmbedder",
]


