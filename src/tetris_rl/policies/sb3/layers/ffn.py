# src/tetris_rl/policies/sb3/layers/ffn.py
from __future__ import annotations

"""
Standard per-token FFN: (B,T,D) -> (B,T,D)
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FFNSpec:
    mult: float = 4.0
    dropout: float = 0.0


class FFN(nn.Module):
    def __init__(self, *, d_model: int, spec: FFNSpec) -> None:
        super().__init__()
        D = int(d_model)
        if D <= 0:
            raise ValueError(f"d_model must be > 0, got {D}")

        mult = float(spec.mult)
        p = float(spec.dropout)
        if mult <= 0.0:
            raise ValueError("mult must be > 0")
        if p < 0.0:
            raise ValueError("dropout must be >= 0")

        H = int(max(1, round(mult * D)))
        self.net = nn.Sequential(
            nn.Linear(D, H),
            nn.GELU(),
            nn.Dropout(p) if p > 0.0 else nn.Identity(),
            nn.Linear(H, D),
            nn.Dropout(p) if p > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"FFN expects (B,T,D), got {tuple(x.shape)}")
        return self.net(x)


__all__ = ["FFNSpec", "FFN"]

