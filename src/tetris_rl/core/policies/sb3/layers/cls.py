# src/tetris_rl/core/policies/sb3/layers/cls.py
"""
CLS token utilities.

Prepend K learned CLS tokens to a token stream and prepend their type ids.
"""

from __future__ import annotations

import torch
from torch import nn


def prepend_cls(
    *,
    x: torch.Tensor,  # (B,T,D)
    types: torch.Tensor,  # (T,)
    cls: nn.Parameter,  # (1,K,D)
    cls_type_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 3:
        raise ValueError(f"x must be (B,T,D), got {tuple(x.shape)}")
    if types.dim() != 1:
        raise ValueError(f"types must be (T,), got {tuple(types.shape)}")

    B, T, D = x.shape
    if cls.dim() != 3 or int(cls.shape[0]) != 1 or int(cls.shape[2]) != int(D):
        raise ValueError(f"cls must be (1,K,D) with D={D}, got {tuple(cls.shape)}")

    K = int(cls.shape[1])
    if K <= 0:
        return x, types

    cls_x = cls.expand(int(B), -1, -1).to(device=x.device, dtype=x.dtype)  # (B,K,D)
    x2 = torch.cat([cls_x, x], dim=1)  # (B,K+T,D)

    t_cls = torch.full((K,), int(cls_type_id), device=x.device, dtype=torch.int64)
    t2 = torch.cat([t_cls, types.to(device=x.device, dtype=torch.int64)], dim=0)  # (K+T,)

    return x2, t2


__all__ = ["prepend_cls"]
