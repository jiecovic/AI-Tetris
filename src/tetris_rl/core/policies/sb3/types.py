# src/tetris_rl/core/policies/sb3/types.py
from __future__ import annotations

from typing import Literal, TypeAlias

# ---------------------------------------------------------------------
# Activation names
# ---------------------------------------------------------------------

# Module-level activations used in stems/heads/tokenizers/augmenters.
# Includes aliases that normalize to canonical names in activations.py.
LayerActivationName: TypeAlias = Literal[
    "gelu",
    "gelu_none",
    "gelu_tanh",
    "gelu_approx",
    "gelu_fast",
    "relu",
    "silu",
    "swish",
]

# Policy MLP activation_fn options (SB3) include LayerActivationName plus tanh/identity.
PolicyActivationName: TypeAlias = Literal[
    "gelu",
    "gelu_none",
    "gelu_tanh",
    "gelu_approx",
    "gelu_fast",
    "relu",
    "silu",
    "swish",
    "tanh",
    "identity",
    "none",
]

# ---------------------------------------------------------------------
# Pooling / collapse names
# ---------------------------------------------------------------------

# Basic average-or-max pool choice.
PoolAvgMaxName: TypeAlias = Literal["avg", "max"]

# Average / max / concatenated avgmax pool choice.
PoolAvgMaxCatName: TypeAlias = Literal["avg", "max", "avgmax"]

# Token pooling kinds for token mixers.
TokenPoolKind: TypeAlias = Literal[
    "cls",
    "mean",
    "max",
    "meanmax",
    "flatten",
    "cls_mean",
    "cls_max",
    "cls_meanmax",
]

# Column collapse reducer kind.
CollapseKindName: TypeAlias = Literal["avg", "max", "linear"]

# ---------------------------------------------------------------------
# Component names (typed tags)
# ---------------------------------------------------------------------

SpatialHeadName: TypeAlias = Literal["global_pool", "flatten", "flatten_mlp", "attn_pool", "col_collapse"]
FeatureAugmenterName: TypeAlias = Literal["none", "null", "onehot_concat", "mlp_joint", "mlp_split"]
MixerName: TypeAlias = Literal["mlp", "transformer"]
StemName: TypeAlias = Literal["cnn", "conv3x3_32_32_64"]

# Minimal disabled tags accepted in SB3 config sections.
NullComponentTag: TypeAlias = Literal["none", "null"]
NULL_COMPONENT_TAGS: frozenset[str] = frozenset({"none", "null"})


__all__ = [
    "LayerActivationName",
    "PolicyActivationName",
    "PoolAvgMaxName",
    "PoolAvgMaxCatName",
    "TokenPoolKind",
    "CollapseKindName",
    "SpatialHeadName",
    "FeatureAugmenterName",
    "MixerName",
    "StemName",
    "NullComponentTag",
    "NULL_COMPONENT_TAGS",
]
