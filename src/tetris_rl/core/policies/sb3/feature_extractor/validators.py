# src/tetris_rl/core/policies/sb3/feature_extractor/validators.py
from __future__ import annotations

from gymnasium import spaces

from tetris_rl.core.policies.sb3.api import SpatialFeatures, Specials, TokenStream


def infer_grid_hw_from_obs_space(observation_space: spaces.Space) -> tuple[int, int]:
    if not isinstance(observation_space, spaces.Dict):
        raise TypeError(f"expected spaces.Dict observation_space, got {type(observation_space)!r}")
    if "grid" not in observation_space.spaces:
        raise KeyError("observation_space missing key 'grid'")
    sp = observation_space.spaces["grid"]
    if not isinstance(sp, spaces.Box):
        raise TypeError(f"obs['grid'] must be spaces.Box, got {type(sp)!r}")
    if sp.shape is None or len(sp.shape) != 2:
        raise ValueError(f"obs['grid'] must be shape (H,W), got {sp.shape!r}")
    H, W = int(sp.shape[0]), int(sp.shape[1])
    if H <= 0 or W <= 0:
        raise ValueError(f"invalid grid shape from obs_space: (H,W)=({H},{W})")
    return H, W


def check_spatial(spatial: SpatialFeatures) -> None:
    x = spatial.x
    if x.dim() != 4:
        raise ValueError(f"SpatialFeatures.x must be (B,H,W,C), got {tuple(x.shape)}")
    if not x.is_floating_point():
        raise ValueError("SpatialFeatures.x must be floating point (e.g. float32)")


def check_specials(specials: Specials) -> None:
    if specials.active_kind.dim() not in (0, 1, 2):
        raise ValueError(f"Specials.active_kind must be scalar/(B,)/(B,K), got {tuple(specials.active_kind.shape)}")
    if specials.next_kind is not None and specials.next_kind.dim() not in (0, 1, 2):
        raise ValueError(f"Specials.next_kind must be scalar/(B,)/(B,K), got {tuple(specials.next_kind.shape)}")


def check_stream(stream: TokenStream) -> None:
    x = stream.x
    types = stream.types
    if x.dim() != 3:
        raise ValueError(f"TokenStream.x must be (B,T,D), got {tuple(x.shape)}")
    if types.dim() != 1:
        raise ValueError(f"TokenStream.types must be (T,), got {tuple(types.shape)}")
    if int(types.shape[0]) != int(x.shape[1]):
        raise ValueError(f"type length mismatch: types T={int(types.shape[0])} vs x T={int(x.shape[1])}")


__all__ = [
    "infer_grid_hw_from_obs_space",
    "check_spatial",
    "check_specials",
    "check_stream",
]
