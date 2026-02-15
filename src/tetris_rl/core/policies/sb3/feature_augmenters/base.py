# src/tetris_rl/core/policies/sb3/feature_augmenters/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from torch import nn

from tetris_rl.core.policies.sb3.api import Specials
from tetris_rl.core.policies.sb3.layers.activations import make_activation


def _validate_features(x: torch.Tensor) -> tuple[int, int]:
    if x.dim() != 2:
        raise ValueError(f"features must be (B,F), got {tuple(x.shape)}")
    B, F = x.shape
    if int(B) <= 0 or int(F) <= 0:
        raise ValueError(f"invalid features shape (B,F)={tuple(x.shape)}")
    if not x.is_floating_point():
        raise ValueError("features must be floating point (e.g. float32)")
    return int(B), int(F)


def _as_batched_int64(t: torch.Tensor, *, B: int, name: str) -> torch.Tensor:
    """
    Accepts:
      - scalar ()
      - batched (B,)
    Returns:
      - (B,) int64
    """
    if t.dim() == 0:
        t = t.view(1).expand(B)
    if t.dim() != 1 or int(t.shape[0]) != int(B):
        raise ValueError(f"{name} must be scalar or (B,), got {tuple(t.shape)} with B={B}")
    return t.to(dtype=torch.int64)


def _one_hot(ids: torch.Tensor, *, n: int) -> torch.Tensor:
    """
    ids: (B,) int64
    returns: (B,n) float32
    """
    if ids.dim() != 1:
        raise ValueError(f"ids must be (B,), got {tuple(ids.shape)}")
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if torch.any((ids < 0) | (ids >= n)):
        bad = ids[(ids < 0) | (ids >= n)]
        raise ValueError(f"one_hot ids out of range [0,{n}): examples={bad[:8].tolist()}")
    return torch.nn.functional.one_hot(ids, num_classes=int(n)).to(dtype=torch.float32)


def _as_int_tuple(xs: Sequence[int] | None) -> tuple[int, ...]:
    if xs is None:
        return ()
    return tuple(int(x) for x in xs)


def _build_mlp(
    *,
    in_dim: int,
    hidden_dims: tuple[int, ...],
    out_dim: int,
    activation: str,
    dropout: float,
) -> nn.Module:
    if in_dim <= 0:
        raise ValueError(f"in_dim must be > 0, got {in_dim}")
    if out_dim <= 0:
        raise ValueError(f"out_dim must be > 0, got {out_dim}")
    if dropout < 0.0:
        raise ValueError("dropout must be >= 0")

    p = float(dropout)

    layers: list[nn.Module] = []
    d_prev = int(in_dim)

    for h in hidden_dims:
        if int(h) <= 0:
            raise ValueError(f"hidden_dims entries must be > 0, got {h}")
        layers.append(nn.Linear(d_prev, int(h), bias=True))
        layers.append(make_activation(activation))
        if p > 0.0:
            layers.append(nn.Dropout(p))
        d_prev = int(h)

    layers.append(nn.Linear(d_prev, int(out_dim), bias=True))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class FeatureAugmenterBaseParams:
    """
    Intentionally feature-dim agnostic:
    augmenters return (B, F + extra) with NO projection back.
    """
    pass


class BaseFeatureAugmenter(nn.Module, ABC):
    """
    Helper base:
      - validate feature tensor
      - build one-hot vectors from Specials

    NO features_dim, NO out_proj, NO lazy init.
    """

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    @abstractmethod
    def infer_extra_dim(cls, *, params: Any, n_kinds: Optional[int]) -> int:
        """
        Return how many dimensions this augmenter appends to base features.
        """

    @staticmethod
    def _maybe_onehot_specials(
        *,
        specials: Specials,
        B: int,
        n_kinds: int,
        use_active: bool,
        use_next: bool,
        device: torch.device,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []

        if use_active:
            a = _as_batched_int64(specials.active_kind, B=B, name="active_kind").to(device=device)
            parts.append(_one_hot(a, n=int(n_kinds)).to(device=device))

        if use_next:
            if specials.next_kind is None:
                parts.append(torch.zeros((B, int(n_kinds)), device=device, dtype=torch.float32))
            else:
                n = _as_batched_int64(specials.next_kind, B=B, name="next_kind").to(device=device)
                parts.append(_one_hot(n, n=int(n_kinds)).to(device=device))

        if len(parts) == 0:
            return torch.zeros((B, 0), device=device, dtype=torch.float32)

        return torch.cat(parts, dim=1)  # (B,K) or (B,2K)


__all__ = [
    "FeatureAugmenterBaseParams",
    "BaseFeatureAugmenter",
    "_validate_features",
    "_as_batched_int64",
    "_one_hot",
    "_as_int_tuple",
    "_build_mlp",
]


