# src/tetris_rl/models/layers/activations.py
from __future__ import annotations

from typing import Literal

from torch import nn

ActivationName = Literal["gelu", "relu", "silu", "swish"]


def make_activation(name: str) -> nn.Module:
    """
    Map a string to an activation module.

    Conventions:
    - ReLU is non-inplace (safer for residual-style models + debugging).
    - "swish" is treated as alias for SiLU.
    """
    n = str(name).strip().lower()
    if n == "gelu":
        return nn.GELU()
    if n == "relu":
        return nn.ReLU(inplace=False)
    if n in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError(f"unknown activation: {name!r}")
