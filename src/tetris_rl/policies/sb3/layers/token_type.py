# src/tetris_rl/policies/sb3/layers/token_type.py
from __future__ import annotations

"""
Token type encoding (learned) + global token-type identifiers.

This module provides two things:

1) TokenType (IntEnum):
   A stable shared vocabulary of token-family/type ids that both the tokenizer and
   mixers agree on. The tokenizer assigns a TokenType to each emitted token, and
   mixers may prepend CLS tokens using TokenType.CLS.

2) TokenTypeEncoding (nn.Module):
   A learned embedding lookup that maps token type ids -> ℝ^D and returns a tensor
   reminding shape conventions used across the codebase.

Contract (TokenTypeEncoding)
----------------------------
Inputs:
  - types: (T,) int64 token-type ids (one id per token position)

Output:
  - enc: (1, T, D) float32, broadcastable to (B, T, D)
    Intended use:
      x = x + type_enc(types=types)

Notes
-----
- Token type ids are stable across runs/configs. Some ids may be unused depending on
  tokenizer layout or whether specials are enabled. 
- CLS is mixer-owned, but lives in the same shared vocabulary so the type embedding
  table can treat it uniformly.
"""

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import nn


class TokenType(IntEnum):
    # board-derived tokens
    ROW = 0
    COL = 1
    PATCH = 2

    # specials (environment-derived)
    ACTIVE = 3
    NEXT = 4

    # mixer-owned
    CLS = 5


# Total number of token types (for nn.Embedding size).
# NOTE: If you add a new TokenType, this constant updates automatically.
NUM_TOKEN_TYPES: int = len(TokenType)


@dataclass(frozen=True)
class TokenTypeEncodingSpec:
    """
    Spec for TokenTypeEncoding.

    n_types:
      Size of the token-type vocabulary (must be > max(type_id)).
      In this project you typically set n_types=NUM_TOKEN_TYPES.

    dropout:
      Dropout applied to the looked-up type embeddings (independent of other dropouts).
    """

    n_types: int
    dropout: float = 0.0


class TokenTypeEncoding(nn.Module):
    """
    Learned token-type embedding.

    Given token type ids (T,), returns (1,T,D) so it can be added to a batch token
    stream (B,T,D) via broadcasting.
    """

    def __init__(self, *, d_model: int, spec: TokenTypeEncodingSpec) -> None:
        super().__init__()
        D = int(d_model)
        n = int(spec.n_types)
        p = float(spec.dropout)

        if D <= 0:
            raise ValueError(f"d_model must be > 0, got {D}")
        if n <= 0:
            raise ValueError(f"n_types must be > 0, got {n}")
        if p < 0.0:
            raise ValueError("dropout must be >= 0")

        self.emb = nn.Embedding(n, D)
        self.drop = nn.Dropout(p) if p > 0.0 else nn.Identity()

    def forward(self, *, types: torch.Tensor) -> torch.Tensor:
        """
        Args:
          types: (T,) int64 token type ids.

        Returns:
          (1,T,D) float32 type embeddings, broadcastable to (B,T,D).
        """
        if types.dim() != 1:
            raise ValueError(f"types must be (T,), got {tuple(types.shape)}")
        if types.dtype not in (torch.int64, torch.long):
            types = types.to(dtype=torch.int64)

        y = self.emb(types)  # (T,D)
        y = self.drop(y)
        return y.unsqueeze(0)  # (1,T,D)


__all__ = [
    "TokenType",
    "NUM_TOKEN_TYPES",
    "TokenTypeEncodingSpec",
    "TokenTypeEncoding",
]

