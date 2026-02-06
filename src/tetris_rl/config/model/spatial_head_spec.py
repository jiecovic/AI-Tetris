from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from tetris_rl.config.base import ConfigBase
from tetris_rl.config.typed_params import parse_typed_params

Pool2D = Literal["avg", "max", "avgmax"]
SpatialHeadType = Literal["global_pool", "flatten", "flatten_mlp", "attn_pool", "col_collapse"]


class SpatialHeadParamsBase(ConfigBase):
    features_dim: int = Field(ge=1)


# -----------------------------
# PARAMS
# -----------------------------


class GlobalPoolParams(SpatialHeadParamsBase):
    conv_channels: tuple[int, ...] = ()
    conv_kernel_sizes: tuple[int, ...] = ()
    conv_strides: Optional[tuple[int, ...]] = None
    activation: Literal["gelu", "relu", "silu"] = "gelu"
    use_batchnorm: bool = False
    dropout: float = 0.0
    pool: Pool2D = "avg"
    mlp_hidden: int = 0


class FlattenParams(SpatialHeadParamsBase):
    proj: bool = True


class FlattenMLPParams(SpatialHeadParamsBase):
    hidden_dims: tuple[int, ...] = (256, 256)
    activation: Literal["gelu", "relu", "silu"] = "gelu"
    dropout: float = 0.0


class AttentionPoolParams(SpatialHeadParamsBase):
    n_queries: int = 1
    mlp_hidden: int = 0
    activation: Literal["gelu", "relu", "silu"] = "gelu"
    dropout: float = 0.0


CollapseKind = Literal["avg", "max", "linear"]
Pool1D = Literal["avg", "max", "avgmax"]


class ColumnCollapseParams(SpatialHeadParamsBase):
    """
    Params for the Tetris-inductive-bias head (ColumnCollapseHead).

    IMPORTANT:
    This MUST match what the model head expects (see:
      src/tetris_rl/models/spatial_heads/col_collapse.py :: ColumnCollapseHeadSpec)

    """

    # --- collapse over height (H) to get per-column vectors ---
    collapse: CollapseKind = "linear"  # paper says "a layer"; default to learned linear collapse
    linear_collapse_per_channel: bool = True  # if True: weights per (C,H); else shared over channels

    # --- 1D conv stack over columns (width axis W) ---
    # paper: conv3-128, conv1-128, conv3-128
    conv_channels: tuple[int, ...] = (128, 128, 128)
    conv_kernel_sizes: tuple[int, ...] = (3, 1, 3)

    activation: Literal["gelu", "relu", "silu"] = "relu"
    use_batchnorm: bool = True

    # paper retention prob 0.75 => dropout p = 0.25
    dropout: float = 0.25

    # pooling over columns after conv stack
    pool: Pool1D = "avg"  # paper doesn't specify; avg is a standard default

    # --- FC stack after pooling ---
    # paper: FC-128, FC-512, then task head
    fc_hidden: tuple[int, ...] = (128, 512)

    # apply activation+dropout after the last FC (the 512 layer)
    # paper has a task head after 512, so there is a nonlinearity before that head.
    post_fc_activation: bool = True

    # --- optional discrete piece metadata (one-hot, concatenated before FC stack) ---
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
