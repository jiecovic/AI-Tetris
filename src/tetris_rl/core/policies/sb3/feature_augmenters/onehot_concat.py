# src/tetris_rl/core/policies/sb3/feature_augmenters/onehot_concat.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from tetris_rl.core.policies.sb3.api import Specials
from tetris_rl.core.policies.sb3.feature_augmenters.base import (
    BaseFeatureAugmenter,
    FeatureAugmenterBaseParams,
    _validate_features,
)


@dataclass(frozen=True)
class OneHotConcatParams(FeatureAugmenterBaseParams):
    """
    Concatenate one-hot encoded specials to the feature vector.

    Output:
      features': (B, F + (use_active?K:0) + (use_next?K:0))

    NOTE: NO projection back to any fixed dim.
    """

    use_active: bool = True
    use_next: bool = False


class OneHotConcatAugmenter(BaseFeatureAugmenter):
    @classmethod
    def infer_extra_dim(cls, *, params: Any, n_kinds: int | None) -> int:
        if n_kinds is None or int(n_kinds) <= 0:
            raise ValueError("n_kinds must be > 0")
        K = int(n_kinds)
        use_active = bool(getattr(params, "use_active", True))
        use_next = bool(getattr(params, "use_next", False))
        return (K if use_active else 0) + (K if use_next else 0)

    def __init__(self, *, params: OneHotConcatParams, n_kinds: int) -> None:
        super().__init__()
        if int(n_kinds) <= 0:
            raise ValueError("n_kinds must be > 0")
        self.params = params
        self.n_kinds = int(n_kinds)

    def forward(self, *, features: torch.Tensor, specials: Specials) -> torch.Tensor:
        B, _F = _validate_features(features)

        z = self._maybe_onehot_specials(
            specials=specials,
            B=B,
            n_kinds=int(self.n_kinds),
            use_active=bool(self.params.use_active),
            use_next=bool(self.params.use_next),
            device=features.device,
        )  # (B,K) or (B,2K) or (B,0)

        if int(z.shape[1]) == 0:
            return features

        return torch.cat([features, z.to(device=features.device, dtype=features.dtype)], dim=1)


__all__ = ["OneHotConcatParams", "OneHotConcatAugmenter"]
