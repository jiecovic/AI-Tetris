# src/tetris_rl/core/policies/sb3/feature_extractor/builders.py
from __future__ import annotations

import inspect
from typing import Any, Optional, cast

from pydantic import BaseModel
from torch import nn

from tetris_rl.core.policies.sb3.api import SpatialPreprocessor, SpatialSpec, SpatialStem
from tetris_rl.core.policies.sb3.catalog import (
    FEATURE_AUGMENTER_REGISTRY,
    SPATIAL_HEAD_REGISTRY,
    SPATIAL_PREPROCESSOR_REGISTRY,
    STEM_REGISTRY,
    TOKEN_MIXER_REGISTRY,
)
from tetris_rl.core.policies.sb3.feature_augmenters.config import FeatureAugmenterConfig
from tetris_rl.core.policies.sb3.mixers.config import MixerConfig
from tetris_rl.core.policies.sb3.spatial.config import SpatialPreprocessorConfig, StemConfig
from tetris_rl.core.policies.sb3.spatial_heads.config import SpatialHeadConfig
from tetris_rl.core.policies.sb3.tokenizers.config import TokenizerConfig
from tetris_rl.core.policies.sb3.tokenizers.tetris_tokenizer import TetrisTokenizer
from tetris_rl.core.policies.sb3.types import NULL_COMPONENT_TAGS


def build_spatial_preprocessor(*, cfg: SpatialPreprocessorConfig) -> SpatialPreprocessor:
    key = str(cfg.type).strip().lower()
    cls = SPATIAL_PREPROCESSOR_REGISTRY.get(key)
    if cls is None:
        raise KeyError(f"unknown spatial_preprocessor type: {cfg.type!r}")

    params = cfg.params
    if params is None:
        return cast(SpatialPreprocessor, cls())
    if isinstance(params, dict):
        return cast(SpatialPreprocessor, cls(**params))
    return cast(SpatialPreprocessor, cls(params=params))


def build_stem(*, cfg: StemConfig, in_channels: int) -> SpatialStem:
    cls = STEM_REGISTRY[cfg.type]
    if cfg.params is None:
        return cast(SpatialStem, cls(in_channels=int(in_channels)))
    return cast(SpatialStem, cls(in_channels=int(in_channels), spec=cfg.params))


def build_tokenizer(*, cfg: TokenizerConfig, n_kinds: Optional[int], in_spec: SpatialSpec) -> TetrisTokenizer:
    return TetrisTokenizer(
        d_model=int(cfg.d_model),
        layout=cfg.layout,
        board_embedding=cfg.board_embedding,
        add_active_token=bool(cfg.add_active_token),
        add_next_token=bool(cfg.add_next_token),
        share_kind_embedding=bool(getattr(cfg, "share_kind_embedding", True)),
        n_kinds=n_kinds,
        in_spec=in_spec,
    )


def build_token_mixer(*, cfg: MixerConfig, d_model: int, T_total: int) -> nn.Module:
    key = str(cfg.type).strip().lower()
    cls = TOKEN_MIXER_REGISTRY.get(key)
    if cls is None:
        raise KeyError(f"unknown mixer type: {cfg.type!r}")

    sig = inspect.signature(cls.__init__)
    kwargs: dict[str, Any] = {"spec": cfg.params}
    if "d_model" in sig.parameters:
        kwargs["d_model"] = int(d_model)
    if "T_total" in sig.parameters:
        kwargs["T_total"] = int(T_total)
    return cast(nn.Module, cls(**kwargs))


def build_spatial_head(*, cfg: SpatialHeadConfig, in_spec: SpatialSpec, features_dim: int) -> nn.Module:
    key = str(cfg.type).strip().lower()
    cls = SPATIAL_HEAD_REGISTRY.get(key)
    if cls is None:
        raise KeyError(f"unknown spatial_head type: {cfg.type!r}")

    sig = inspect.signature(cls.__init__)
    kwargs: dict[str, Any] = {"spec": cfg.params}

    # Standardized geometry injection (heads may accept any subset)
    if "in_h" in sig.parameters:
        kwargs["in_h"] = int(in_spec.h)
    if "in_w" in sig.parameters:
        kwargs["in_w"] = int(in_spec.w)
    if "in_channels" in sig.parameters:
        kwargs["in_channels"] = int(in_spec.c)

    # Output dim is configured in params but injected as a separate arg.
    if "features_dim" in sig.parameters:
        kwargs["features_dim"] = int(features_dim)

    return cast(nn.Module, cls(**kwargs))


def _maybe_set_param_features_dim(params: Any, *, features_dim: int) -> Any:
    if params is None:
        return None
    if not hasattr(params, "features_dim"):
        return params

    try:
        cur = int(getattr(params, "features_dim"))
        if cur == int(features_dim):
            return params
        if isinstance(params, BaseModel):
            return params.model_copy(update={"features_dim": int(features_dim)})
        d = dict(getattr(params, "__dict__", {}))
        d["features_dim"] = int(features_dim)
        return type(params)(**d)
    except Exception:
        return params


def build_feature_augmenter(
    *,
    cfg: FeatureAugmenterConfig,
    n_kinds: Optional[int],
    in_dim: int,
    features_dim: int,
) -> nn.Module:
    key = str(cfg.type).strip().lower()
    if key == "" or key in NULL_COMPONENT_TAGS:
        raise ValueError("feature_augmenter type is disabled but builder was called")

    cls = FEATURE_AUGMENTER_REGISTRY.get(key)
    if cls is None:
        raise KeyError(f"unknown feature_augmenter type: {cfg.type!r}")
    if cfg.params is None:
        raise ValueError("feature_augmenter params must not be None when enabled")

    params = _maybe_set_param_features_dim(cfg.params, features_dim=int(features_dim))
    sig = inspect.signature(cls.__init__)

    # Support multiple init styles:
    #   __init__(*, spec=...)
    #   __init__(*, params=..., in_dim=..., features_dim=..., n_kinds=...)
    kwargs: dict[str, Any] = {}
    if "spec" in sig.parameters:
        kwargs["spec"] = params
    elif "params" in sig.parameters:
        kwargs["params"] = params
    else:
        raise TypeError(f"{cls.__name__}.__init__ must accept 'spec' or 'params'")

    if "in_dim" in sig.parameters:
        kwargs["in_dim"] = int(in_dim)
    if "features_dim" in sig.parameters:
        kwargs["features_dim"] = int(features_dim)

    # n_kinds is injected (NOT in params anymore)
    if "n_kinds" in sig.parameters:
        if n_kinds is None or int(n_kinds) <= 0:
            raise ValueError("feature_augmenter requires n_kinds injection from env/assets")
        kwargs["n_kinds"] = int(n_kinds)

    return cast(nn.Module, cls(**kwargs))


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
    "build_spatial_preprocessor",
    "build_stem",
    "build_tokenizer",
    "build_token_mixer",
    "build_spatial_head",
    "build_feature_augmenter",
    "resolve_spatial_head_features_dim",
    "infer_feature_augmenter_extra_dim",
]
