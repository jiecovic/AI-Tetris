# src/tetris_rl/core/policies/sb3/layers/pooling.py
"""
Token pooling utilities.

All pooling operates on token tensors shaped (B, T, D) and returns (B, H).

Supported pooling kinds:
- mean
- max
- meanmax
- flatten
- cls
- cls_mean
- cls_max
- cls_meanmax

Conventions:
- For cls* modes, the first K tokens are treated as CLS tokens.
- CLS tokens never get positional encodings (handled elsewhere).
"""

from __future__ import annotations

from typing import Optional, cast

import torch

from tetris_rl.core.policies.sb3.types import TokenPoolKind

PoolKind = TokenPoolKind


def pooled_dim(*, kind: PoolKind, T: int, D: int, num_cls_tokens: int = 0) -> int:
    """
    Compute pooled feature dimension for pool_tokens(kind=...).

    Args:
      kind: pooling strategy
      T: total token count (including CLS tokens if present)
      D: embedding dim
      num_cls_tokens: K (number of CLS tokens at prefix). Required for cls*.

    Returns:
      H where H depends on kind:
        - mean/max:     H = D
        - meanmax:      H = 2D
        - flatten:      H = T*D
        - cls:          H = K*D
        - cls_mean/max: H = K*D + D
        - cls_meanmax:  H = K*D + 2D
    """
    T = int(T)
    D = int(D)
    k = int(num_cls_tokens)
    kind = cast(PoolKind, str(kind))

    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    if D <= 0:
        raise ValueError(f"D must be > 0, got {D}")

    if kind == "flatten":
        return int(T * D)
    if kind in ("mean", "max"):
        return int(D)
    if kind == "meanmax":
        return int(2 * D)

    # --- cls-based pooling requires CLS prefix ---
    if k <= 0:
        raise ValueError(f"pooling={kind!r} requires num_cls_tokens>0")
    if k > int(T):
        raise ValueError(f"num_cls_tokens={k} cannot exceed T={int(T)}")

    if kind == "cls":
        return int(k * D)

    # cls_* modes require at least one non-CLS token
    if int(T - k) <= 0:
        raise ValueError(f"pooling={kind!r} requires at least one non-CLS token (T={int(T)}, k={k})")

    if kind in ("cls_mean", "cls_max"):
        return int(k * D + D)
    if kind == "cls_meanmax":
        return int(k * D + 2 * D)

    raise ValueError(f"unknown pooling kind: {kind!r}")


def pool_tokens(*, x: torch.Tensor, kind: PoolKind, num_cls_tokens: int = 0) -> torch.Tensor:
    """
    Pool tokens.

    Args:
      x: (B, T, D)
      kind: pooling strategy
      num_cls_tokens: K (number of CLS tokens at prefix). Required for cls*.

    Returns:
      (B, H) where H depends on kind (see pooled_dim()).
    """
    if x.dim() != 3:
        raise ValueError(f"x must be (B,T,D), got {tuple(x.shape)}")

    B, T, D = x.shape
    k = int(num_cls_tokens)
    kind = cast(PoolKind, str(kind))

    if kind == "flatten":
        return x.reshape(int(B), int(T) * int(D))

    if kind == "mean":
        return x.mean(dim=1)

    if kind == "max":
        return x.amax(dim=1)

    if kind == "meanmax":
        m = x.mean(dim=1)
        mx = x.amax(dim=1)
        return torch.cat([m, mx], dim=-1)

    # --- cls-based pooling requires CLS prefix ---
    if k <= 0:
        raise ValueError(f"pooling={kind!r} requires num_cls_tokens>0")

    if k > int(T):
        raise ValueError(f"num_cls_tokens={k} cannot exceed T={int(T)}")

    cls = x[:, :k, :]  # (B, k, D)
    rest = x[:, k:, :]  # (B, T-k, D)

    cls_flat = cls.reshape(int(B), int(k) * int(D))

    if kind == "cls":
        return cls_flat

    if rest.shape[1] == 0:
        raise ValueError(f"pooling={kind!r} requires at least one non-CLS token (T={int(T)}, k={k})")

    if kind == "cls_mean":
        pooled_rest = rest.mean(dim=1)
        return torch.cat([cls_flat, pooled_rest], dim=-1)

    if kind == "cls_max":
        pooled_rest = rest.amax(dim=1)
        return torch.cat([cls_flat, pooled_rest], dim=-1)

    if kind == "cls_meanmax":
        m = rest.mean(dim=1)
        mx = rest.amax(dim=1)
        return torch.cat([cls_flat, m, mx], dim=-1)

    raise ValueError(f"unknown pooling kind: {kind!r}")


def concat_extra(*, base: torch.Tensor, extra: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Concatenate bypass features after pooling/flattening.

    Args:
      base:  (B, H)
      extra: (B, E) or None

    Returns:
      (B, H+E) if extra is provided, else (B, H)
    """
    if extra is None:
        return base
    if base.dim() != 2 or extra.dim() != 2:
        raise ValueError(f"base and extra must be 2D, got base={tuple(base.shape)} extra={tuple(extra.shape)}")
    if int(base.shape[0]) != int(extra.shape[0]):
        raise ValueError(f"batch mismatch: base B={int(base.shape[0])} vs extra B={int(extra.shape[0])}")
    return torch.cat([base, extra.to(device=base.device, dtype=base.dtype)], dim=1)


__all__ = ["PoolKind", "pooled_dim", "pool_tokens", "concat_extra"]

