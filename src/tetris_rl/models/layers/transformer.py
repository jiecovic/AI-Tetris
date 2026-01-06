# src/tetris_rl/models/layers/transformer.py
from __future__ import annotations

"""
Transformer encoder block (pre-norm).

LN -> MHA -> resid
LN -> FFN -> resid
"""

from dataclasses import dataclass

import torch
from torch import nn

from tetris_rl.models.layers.ffn import FFN, FFNSpec


@dataclass(frozen=True)
class TransformerBlockSpec:
    n_heads: int
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    ffn: FFNSpec = FFNSpec()


class TransformerBlock(nn.Module):
    def __init__(self, *, d_model: int, spec: TransformerBlockSpec) -> None:
        super().__init__()
        D = int(d_model)
        H = int(spec.n_heads)

        if D <= 0:
            raise ValueError(f"d_model must be > 0, got {D}")
        if H <= 0:
            raise ValueError(f"n_heads must be > 0, got {H}")
        if D % H != 0:
            raise ValueError(f"d_model must be divisible by n_heads (got d_model={D}, n_heads={H})")

        a_p = float(spec.attn_dropout)
        r_p = float(spec.resid_dropout)
        if a_p < 0.0 or r_p < 0.0:
            raise ValueError("dropout must be >= 0")

        self.ln1 = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(embed_dim=D, num_heads=H, dropout=a_p, batch_first=True)
        self.drop_attn = nn.Dropout(r_p) if r_p > 0.0 else nn.Identity()

        self.ln2 = nn.LayerNorm(D)
        self.ffn = FFN(d_model=D, spec=spec.ffn)
        self.drop_ffn = nn.Dropout(r_p) if r_p > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"TransformerBlock expects (B,T,D), got {tuple(x.shape)}")

        h = self.ln1(x)
        y, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop_attn(y)

        y2 = self.ffn(self.ln2(x))
        x = x + self.drop_ffn(y2)
        return x


__all__ = ["TransformerBlockSpec", "TransformerBlock"]
