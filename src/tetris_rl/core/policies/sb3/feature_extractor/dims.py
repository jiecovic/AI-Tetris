# src/tetris_rl/core/policies/sb3/feature_extractor/dims.py
from __future__ import annotations

from typing import Optional

from tetris_rl.core.policies.sb3.api import SpatialSpec
from tetris_rl.core.policies.sb3.catalog import FEATURE_AUGMENTER_REGISTRY, SPATIAL_HEAD_REGISTRY
from tetris_rl.core.policies.sb3.feature_augmenters.config import FeatureAugmenterConfig
from tetris_rl.core.policies.sb3.spatial_heads.config import SpatialHeadConfig
from tetris_rl.core.policies.sb3.types import NULL_COMPONENT_TAGS


def resolve_spatial_head_features_dim(*, cfg: SpatialHeadConfig, in_spec: SpatialSpec) -> int:
    """
    Resolve base output dim for spatial heads.

    Supports:
      - explicit int > 0
      - "auto" (delegated to the head class)
    """
    raw = getattr(cfg.params, "features_dim", None)

    if isinstance(raw, bool):
        raise ValueError("spatial_head.params.features_dim must be int > 0 or 'auto' (bool is not allowed)")

    if isinstance(raw, int):
        if int(raw) <= 0:
            raise ValueError(f"spatial_head.params.features_dim must be > 0, got {raw}")
        return int(raw)

    if not isinstance(raw, str):
        raise ValueError(f"spatial_head.params.features_dim must be int > 0 or 'auto', got {raw!r}")

    tag = raw.strip().lower()
    if tag != "auto":
        raise ValueError(f"spatial_head.params.features_dim must be int > 0 or 'auto', got {raw!r}")

    key = str(cfg.type).strip().lower()
    cls = SPATIAL_HEAD_REGISTRY.get(key)
    if cls is None:
        raise KeyError(f"unknown spatial_head type: {cfg.type!r}")

    infer = getattr(cls, "infer_auto_features_dim", None)
    if infer is None:
        raise TypeError(f"{cls.__name__} must implement infer_auto_features_dim(spec=..., in_spec=...)")

    out = int(infer(spec=cfg.params, in_spec=in_spec))
    if out <= 0:
        raise ValueError(f"{cls.__name__}.infer_auto_features_dim returned invalid dim {out}")
    return out


def infer_feature_augmenter_extra_dim(*, cfg: FeatureAugmenterConfig, n_kinds: Optional[int]) -> int:
    """
    Infer how many dimensions the augmenter ADDS on top of base features (F_base).
    """
    key = str(cfg.type).strip().lower()
    if key == "" or key in NULL_COMPONENT_TAGS:
        return 0

    if cfg.params is None:
        raise ValueError("feature_augmenter.params must not be None when enabled")

    cls = FEATURE_AUGMENTER_REGISTRY.get(key)
    if cls is None:
        raise KeyError(f"unknown feature_augmenter type: {cfg.type!r}")

    infer = getattr(cls, "infer_extra_dim", None)
    if infer is None:
        raise TypeError(f"{cls.__name__} must implement infer_extra_dim(params=..., n_kinds=...)")

    extra = int(infer(params=cfg.params, n_kinds=n_kinds))
    if extra < 0:
        raise ValueError(f"feature augmenter extra dim must be >= 0, got {extra}")
    return extra


__all__ = [
    "resolve_spatial_head_features_dim",
    "infer_feature_augmenter_extra_dim",
]
