# src/tetris_rl/core/policies/sb3/feature_augmenters/mlp_split.py
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tetris_rl.core.policies.sb3.api import Specials
from tetris_rl.core.policies.sb3.feature_augmenters.base import (
    BaseFeatureAugmenter,
    FeatureAugmenterBaseParams,
    _as_batched_int64,
    _as_int_tuple,
    _build_mlp,
    _one_hot,
    _validate_features,
)


@dataclass(frozen=True)
class SplitMLPParams(FeatureAugmenterBaseParams):
    """
    Active and next are embedded separately (two MLPs), then concatenated:
      z = [mlp_active(onehot_active), mlp_next(onehot_next)]
    Then concat to features.

    Output:
      features': (B, F + z_used)

    NOTE: NO projection back to any fixed dim.
    """
    use_active: bool = True
    use_next: bool = False

    out_dim_total: int = 64
    out_dim_active: int | None = None
    out_dim_next: int | None = None

    hidden_dims: tuple[int, ...] = (64,)
    activation: str = "gelu"
    dropout: float = 0.0


class SplitMLPAugmenter(BaseFeatureAugmenter):
    def __init__(self, *, params: SplitMLPParams, n_kinds: int) -> None:
        super().__init__()
        if int(n_kinds) <= 0:
            raise ValueError("n_kinds must be > 0")
        if int(params.out_dim_total) < 0:
            raise ValueError("out_dim_total must be >= 0")
        if float(params.dropout) < 0.0:
            raise ValueError("dropout must be >= 0")

        self.params = params
        self.n_kinds = int(n_kinds)

        self._da, self._dn = self._resolve_dims()

        # Build MLPs eagerly (NO lazy init)
        self.mlp_active: nn.Module | None = None
        self.mlp_next: nn.Module | None = None

        if bool(params.use_active) and self._da > 0:
            self.mlp_active = _build_mlp(
                in_dim=int(self.n_kinds),
                hidden_dims=_as_int_tuple(params.hidden_dims),
                out_dim=int(self._da),
                activation=str(params.activation),
                dropout=float(params.dropout),
            )
        if bool(params.use_next) and self._dn > 0:
            self.mlp_next = _build_mlp(
                in_dim=int(self.n_kinds),
                hidden_dims=_as_int_tuple(params.hidden_dims),
                out_dim=int(self._dn),
                activation=str(params.activation),
                dropout=float(params.dropout),
            )

    def _resolve_dims(self) -> tuple[int, int]:
        total = int(self.params.out_dim_total)
        if total == 0:
            return 0, 0

        da = self.params.out_dim_active
        dn = self.params.out_dim_next

        if da is None and dn is None:
            da = total // 2
            dn = total - int(da)
        elif da is None:
            dn_i = int(dn)
            da = total - dn_i
        elif dn is None:
            da_i = int(da)
            dn = total - da_i

        da_i = int(da)
        dn_i = int(dn)

        if da_i < 0 or dn_i < 0:
            raise ValueError("out_dim_active/out_dim_next must be >= 0")
        if da_i + dn_i != total:
            raise ValueError(f"out_dim_active + out_dim_next must equal out_dim_total ({total}), got {da_i}+{dn_i}")

        return da_i, dn_i

    def forward(self, *, features: torch.Tensor, specials: Specials) -> torch.Tensor:
        B, _F = _validate_features(features)
        K = int(self.n_kinds)

        parts: list[torch.Tensor] = []

        # --- active branch ---
        if self.mlp_active is not None:
            a = _as_batched_int64(specials.active_kind, B=B, name="active_kind").to(device=features.device)
            oh_a = _one_hot(a, n=K).to(device=features.device, dtype=features.dtype)
            parts.append(self.mlp_active(oh_a))

        # --- next branch ---
        if self.mlp_next is not None:
            if specials.next_kind is None:
                oh_n = torch.zeros((B, K), device=features.device, dtype=features.dtype)
            else:
                n = _as_batched_int64(specials.next_kind, B=B, name="next_kind").to(device=features.device)
                oh_n = _one_hot(n, n=K).to(device=features.device, dtype=features.dtype)
            parts.append(self.mlp_next(oh_n))

        if len(parts) == 0:
            return features

        z = torch.cat(parts, dim=1)  # (B, z_used)
        return torch.cat([features, z], dim=1)


__all__ = ["SplitMLPParams", "SplitMLPAugmenter"]


