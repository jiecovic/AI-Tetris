# src/tetris_rl/config/model/spatial_head_spec.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Mapping, Type, Any

Pool2D = Literal["avg", "max", "avgmax"]
SpatialHeadType = Literal["global_pool", "flatten", "flatten_mlp", "attn_pool", "col_collapse"]


# -----------------------------
# PARAMS (NO features_dim here)
# -----------------------------

@dataclass(frozen=True)
class GlobalPoolParams:
    conv_channels: tuple[int, ...] = ()
    conv_kernel_sizes: tuple[int, ...] = ()
    conv_strides: Optional[tuple[int, ...]] = None
    activation: Literal["gelu", "relu", "silu"] = "gelu"
    use_batchnorm: bool = False
    dropout: float = 0.0
    pool: Pool2D = "avg"
    mlp_hidden: int = 0


@dataclass(frozen=True)
class FlattenParams:
    proj: bool = True


@dataclass(frozen=True)
class FlattenMLPParams:
    hidden_dims: tuple[int, ...] = (256, 256)
    activation: Literal["gelu", "relu", "silu"] = "gelu"
    dropout: float = 0.0


@dataclass(frozen=True)
class AttentionPoolParams:
    n_queries: int = 1
    mlp_hidden: int = 0
    activation: Literal["gelu", "relu", "silu"] = "gelu"
    dropout: float = 0.0


CollapseKind = Literal["avg", "max", "linear"]
Pool1D = Literal["avg", "max", "avgmax"]


@dataclass(frozen=True)
class ColumnCollapseParams:
    """
    Params for the Tetris-inductive-bias head (ColumnCollapseHead).

    IMPORTANT:
    This MUST match what the model head expects (see:
      src/tetris_rl/models/spatial_heads/col_collapse.py :: ColumnCollapseHeadSpec)

    Back-compat:
      - pooling: accepted as alias for pool (older configs)
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
    pool: Pool1D = "avg"  # paper doesn’t specify; avg is a standard default

    # Backward-compat alias; if provided by YAML, resolve.py can map it to `pool`.
    pooling: Optional[str] = None

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
# WRAPPER (owns output dim)
# -----------------------------

@dataclass(frozen=True)
class SpatialHeadConfig:
    type: SpatialHeadType
    features_dim: int
    params: (
            GlobalPoolParams
            | FlattenParams
            | FlattenMLPParams
            | AttentionPoolParams
            | ColumnCollapseParams
    )


# ---------------------------------------------------------------------
# Params registry (resolve.py hydration mapping)
# ---------------------------------------------------------------------

SPATIAL_HEAD_PARAMS_REGISTRY: Mapping[str, Type[Any]] = {
    "global_pool": GlobalPoolParams,
    "flatten": FlattenParams,
    "flatten_mlp": FlattenMLPParams,
    "attn_pool": AttentionPoolParams,
    "col_collapse": ColumnCollapseParams,
}

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
