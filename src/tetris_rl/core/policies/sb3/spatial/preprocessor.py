# src/tetris_rl/core/policies/sb3/spatial/preprocessor.py
"""
Spatial preprocessors.

A SpatialPreprocessor is the FIRST model-side stage after the environment.

Responsibilities:
- Validate and parse the observation dict
- Convert raw board representation into a spatial feature grid
- Extract global (non-spatial) signals as Specials

Non-responsibilities:
- No tokenization
- No pooling
- No CNN mixing
- No policy logic
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from tetris_rl.core.policies.sb3.api import BoardSpec, SpatialFeatures, SpatialSpec, Specials


class BaseSpatialPreprocessor:
    """
    Base class for spatial preprocessors.

    Provides:
    - observation validation
    - batch-dimension normalization
    - consistent extraction of Specials

    Subclasses should ONLY implement how the board grid
    is transformed into spatial features.
    """

    # Required observation keys
    required_keys = ("grid", "active_kind")

    def _validate_observations(self, observations: Dict[str, torch.Tensor]) -> None:
        for k in self.required_keys:
            if k not in observations:
                raise KeyError(f"observations must contain key '{k}'")

    def _get_grid(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        grid = observations["grid"]

        # Ensure batch dimension
        if grid.dim() == 2:
            grid = grid.unsqueeze(0)  # (1,H,W)

        if grid.dim() != 3:
            raise ValueError(f"obs['grid'] must be (B,H,W), got {tuple(grid.shape)}")

        return grid

    def _normalize_kind(self, x: torch.Tensor, *, name: str) -> torch.Tensor:
        """
        Normalize Discrete-like "kind" tensors to integer indices.

        SB3 MultiInput policies one-hot encode Discrete observations, so we may see:
          - scalar index: () or (1,)
          - batched indices: (B,)
          - one-hot: (B,K) or (K,)

        We normalize to integer indices with shape (B,) (or scalar if passed scalar).
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"obs['{name}'] must be a torch.Tensor, got {type(x)!r}")

        # RolloutBuffer/VecEnv sometimes adds a singleton middle axis:
        #   (B,1,K) -> (B,K)
        #   (B,1)   -> (B,)
        if x.dim() == 3 and int(x.shape[1]) == 1:
            x = x[:, 0, :]
        elif x.dim() == 2 and int(x.shape[1]) == 1:
            x = x[:, 0]

        # One-hot batched: (B,K) -> (B,)
        if x.dim() == 2:
            x = torch.argmax(x, dim=-1)

        # 1D: either (B,) indices or (K,) one-hot
        elif x.dim() == 1:
            # Heuristic:
            # - integer tensors are almost certainly indices
            # - float/bool tensors are almost certainly one-hot/prob-like
            if x.dtype.is_floating_point or x.dtype == torch.bool:
                x = torch.argmax(x, dim=-1)
            # else: keep as indices

        # scalar ok
        elif x.dim() == 0:
            pass
        else:
            raise ValueError(f"obs['{name}'] must be scalar, (B,), (K,) or (B,K); got {tuple(x.shape)}")

        # Ensure int64 dtype (SB3 one-hot comes as float)
        if x.dtype != torch.int64:
            x = x.to(dtype=torch.int64)

        return x

    def _get_specials(self, observations: Dict[str, torch.Tensor]) -> Specials:
        active_kind = self._normalize_kind(observations["active_kind"], name="active_kind")

        next_kind = observations.get("next_kind", None)
        if next_kind is not None:
            next_kind = self._normalize_kind(next_kind, name="next_kind")

        return Specials(
            active_kind=active_kind,
            next_kind=next_kind,
        )

    def __call__(
        self,
        *,
        observations: Dict[str, torch.Tensor],
    ) -> Tuple[SpatialFeatures, Specials]:
        """
        Subclasses must override this.
        """
        raise NotImplementedError


class BinarySpatialPreprocessor(BaseSpatialPreprocessor):
    """
    Categorical board -> binary occupancy grid.

    Board semantics:
      - 0 = empty
      - non-zero = occupied

    Output:
      SpatialFeatures:
        - x: (B,H,W,1) float32 with values in {0.0, 1.0}
        - is_discrete_binary = True

      Specials:
        - active_kind, next_kind (as integer indices; SB3 one-hot is normalized)
    """

    def __call__(
        self,
        *,
        observations: Dict[str, torch.Tensor],
    ) -> Tuple[SpatialFeatures, Specials]:
        self._validate_observations(observations)

        grid = self._get_grid(observations)
        specials = self._get_specials(observations)

        # Binary occupancy
        x = (grid != 0).to(dtype=torch.float32)  # (B,H,W)
        x = x.unsqueeze(-1)  # (B,H,W,1)

        spatial = SpatialFeatures(
            x=x,
            is_discrete_binary=True,
        )

        return spatial, specials

    def out_spec(self, *, board: BoardSpec) -> SpatialSpec:
        return SpatialSpec(h=board.h, w=board.w, c=1, is_discrete_binary=True)


__all__ = [
    "BaseSpatialPreprocessor",
    "BinarySpatialPreprocessor",
]
