# src/tetris_rl/policies/sb3/feature_augmenters/mlp_joint.py
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tetris_rl.policies.sb3.api import Specials
from tetris_rl.policies.sb3.feature_augmenters.base import (
    BaseFeatureAugmenter,
    FeatureAugmenterBaseParams,
    _as_int_tuple,
    _build_mlp,
    _validate_features,
)


@dataclass(frozen=True)
class JointMLPParams(FeatureAugmenterBaseParams):
    """
    One-hot specials -> (joint) MLP -> z, concat to features.

    Output:
      features': (B, F + out_dim)  (if enabled and out_dim>0)
    """
    use_active: bool = True
    use_next: bool = False

    out_dim: int = 64
    hidden_dims: tuple[int, ...] = (64,)
    activation: str = "gelu"
    dropout: float = 0.0


class JointMLPAugmenter(BaseFeatureAugmenter):
    def __init__(self, *, params: JointMLPParams, n_kinds: int) -> None:
        super().__init__()
        if int(n_kinds) <= 0:
            raise ValueError("n_kinds must be > 0")
        if int(params.out_dim) < 0:
            raise ValueError("out_dim must be >= 0")
        if float(params.dropout) < 0.0:
            raise ValueError("dropout must be >= 0")

        self.params = params
        self.n_kinds = int(n_kinds)

        use_active = bool(params.use_active)
        use_next = bool(params.use_next)

        self._enabled = (use_active or use_next) and int(params.out_dim) > 0
        if not self._enabled:
            self.mlp = None
            return

        in_dim = (self.n_kinds if use_active else 0) + (self.n_kinds if use_next else 0)
        if in_dim <= 0:
            self._enabled = False
            self.mlp = None
            return

        self.mlp: nn.Module = _build_mlp(
            in_dim=int(in_dim),
            hidden_dims=_as_int_tuple(params.hidden_dims),
            out_dim=int(params.out_dim),
            activation=str(params.activation),
            dropout=float(params.dropout),
        )

    def forward(self, *, features: torch.Tensor, specials: Specials) -> torch.Tensor:
        B, _F = _validate_features(features)
        if not getattr(self, "_enabled", False):
            return features

        onehot = self._maybe_onehot_specials(
            specials=specials,
            B=B,
            n_kinds=int(self.n_kinds),
            use_active=bool(self.params.use_active),
            use_next=bool(self.params.use_next),
            device=features.device,
        )  # (B,K/2K)

        if int(onehot.shape[1]) == 0 or int(self.params.out_dim) == 0:
            return features

        assert self.mlp is not None
        z = self.mlp(onehot.to(device=features.device, dtype=features.dtype))  # (B,out_dim)
        return torch.cat([features, z], dim=1)


__all__ = ["JointMLPParams", "JointMLPAugmenter"]


