# src/tetris_rl/core/policies/sb3/spatial_heads/config.py
from __future__ import annotations

from typing import Literal, Optional

from pydantic import field_validator, model_validator

from tetris_rl.core.config.base import ConfigBase
from tetris_rl.core.config.typed_params import parse_typed_params
from tetris_rl.core.policies.sb3.types import (
    CollapseKindName,
    LayerActivationName,
    PoolAvgMaxCatName,
    SpatialHeadName,
)

Pool2D = PoolAvgMaxCatName
SpatialHeadType = SpatialHeadName


class SpatialHeadParamsBase(ConfigBase):
    features_dim: int | Literal["auto"]

    @field_validator("features_dim", mode="before")
    @classmethod
    def _normalize_features_dim(cls, value: object) -> object:
        # Accept case-insensitive "auto" and numeric strings.
        if isinstance(value, str):
            s = value.strip().lower()
            if s == "auto":
                return "auto"
            try:
                return int(s)
            except Exception:
                return value
        return value

    @field_validator("features_dim", mode="after")
    @classmethod
    def _validate_features_dim(cls, v: int | Literal["auto"]) -> int | Literal["auto"]:
        if isinstance(v, bool):
            raise ValueError("features_dim must be int > 0 or 'auto' (bool is not allowed)")
        if isinstance(v, int):
            if int(v) <= 0:
                raise ValueError(f"features_dim must be > 0, got {v}")
            return v
        if str(v).strip().lower() != "auto":
            raise ValueError(f"features_dim must be int > 0 or 'auto', got {v!r}")
        return "auto"


# -----------------------------
# PARAMS
# -----------------------------


class GlobalPoolParams(SpatialHeadParamsBase):
    conv_channels: tuple[int, ...] = ()
    conv_kernel_sizes: tuple[int, ...] = ()
    conv_strides: Optional[tuple[int, ...]] = None
    activation: LayerActivationName = "gelu"
    use_batchnorm: bool = False
    dropout: float = 0.0
    pool: Pool2D = "avg"
    mlp_hidden: int = 0


class FlattenParams(SpatialHeadParamsBase):
    @model_validator(mode="before")
    @classmethod
    def _drop_legacy_proj(cls, data: object) -> object:
        """
        Back-compat: older configs used `proj: true|false`.
        Flatten now always maps to features_dim, so this key is ignored.
        """
        if isinstance(data, dict) and "proj" in data:
            d = dict(data)
            d.pop("proj", None)
            return d
        return data


class FlattenMLPParams(SpatialHeadParamsBase):
    hidden_dims: tuple[int, ...] = (256, 256)
    activation: LayerActivationName = "gelu"
    dropout: float = 0.0


class AttentionPoolParams(SpatialHeadParamsBase):
    n_queries: int = 1
    mlp_hidden: int = 0
    activation: LayerActivationName = "gelu"
    dropout: float = 0.0


CollapseKind = CollapseKindName
Pool1D = PoolAvgMaxCatName


class ColumnCollapseParams(SpatialHeadParamsBase):
    """
    Params for the Tetris-inductive-bias head (ColumnCollapseHead).

    IMPORTANT:
      This MUST match what the model head expects (see:
        src/tetris_rl/policies/sb3/spatial_heads/col_collapse.py :: ColumnCollapseHeadSpec)

    """

    # --- collapse over height (H) to get per-column vectors ---
    collapse: CollapseKind = "linear"  # default to learned linear collapse
    linear_collapse_per_channel: bool = True  # per-channel weights vs shared

    # --- 1D conv stack over columns (width axis W) ---
    conv_channels: tuple[int, ...] = (128, 128, 128)
    conv_kernel_sizes: tuple[int, ...] = (3, 1, 3)

    activation: LayerActivationName = "relu"
    use_batchnorm: bool = True

    # paper retention prob 0.75 => dropout p = 0.25
    dropout: float = 0.25

    # pooling over columns after conv stack
    pool: Pool1D = "avg"

    # --- FC stack after pooling ---
    fc_hidden: tuple[int, ...] = (128, 512)
    post_fc_activation: bool = True

    # --- optional discrete piece metadata ---
    include_active_onehot: bool = False
    include_next_onehot: bool = False


# -----------------------------
# WRAPPER (type + params)
# -----------------------------


SPATIAL_HEAD_PARAMS_REGISTRY = {
    "global_pool": GlobalPoolParams,
    "flatten": FlattenParams,
    "flatten_mlp": FlattenMLPParams,
    "attn_pool": AttentionPoolParams,
    "col_collapse": ColumnCollapseParams,
}


class SpatialHeadConfig(ConfigBase):
    type: SpatialHeadType
    params: (
        GlobalPoolParams
        | FlattenParams
        | FlattenMLPParams
        | AttentionPoolParams
        | ColumnCollapseParams
    )

    @model_validator(mode="before")
    @classmethod
    def _parse_params(cls, data: object) -> object:
        if isinstance(data, SpatialHeadConfig):
            return data
        if not isinstance(data, dict):
            raise TypeError("spatial_head must be a mapping with keys {type, params}")
        tag, params = parse_typed_params(
            type_value=data.get("type", None),
            params_value=data.get("params", None),
            registry=SPATIAL_HEAD_PARAMS_REGISTRY,
            where="spatial_head",
        )
        return {"type": tag, "params": params}


__all__ = [
    "Pool2D",
    "SpatialHeadType",
    "GlobalPoolParams",
    "FlattenParams",
    "FlattenMLPParams",
    "AttentionPoolParams",
    "CollapseKind",
    "Pool1D",
    "ColumnCollapseParams",
    "SpatialHeadConfig",
    "SPATIAL_HEAD_PARAMS_REGISTRY",
]

