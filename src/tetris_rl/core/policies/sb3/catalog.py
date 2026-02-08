# src/tetris_rl/core/policies/sb3/catalog.py
"""
Central registries for model components.

This file contains ONLY:
  - imports
  - plain dictionaries mapping string keys -> classes / callables

NO logic. NO functions. NO side effects.
"""

from __future__ import annotations

from typing import Any, Mapping

from tetris_rl.core.policies.sb3.feature_augmenters.mlp_joint import JointMLPAugmenter
from tetris_rl.core.policies.sb3.feature_augmenters.mlp_split import SplitMLPAugmenter
from tetris_rl.core.policies.sb3.feature_augmenters.onehot_concat import OneHotConcatAugmenter
from tetris_rl.core.policies.sb3.mixers.mlp_mixer import MLPMixer
from tetris_rl.core.policies.sb3.mixers.transformer_mixer import TransformerMixer
from tetris_rl.core.policies.sb3.spatial.preprocessor import BinarySpatialPreprocessor
from tetris_rl.core.policies.sb3.spatial.stems.cnn_stem import CNNStem
from tetris_rl.core.policies.sb3.spatial.stems.conv1x3_32x4_64_5l import Conv1x3_32x4_64_5LStem
from tetris_rl.core.policies.sb3.spatial.stems.conv3x3_32_32_64 import Conv3x3_32_32_64Stem
from tetris_rl.core.policies.sb3.spatial.stems.conv3x3_32_32_64_64_128_5l import Conv3x3_32_32_64_64_128_5LStem
from tetris_rl.core.policies.sb3.spatial.stems.conv3x3_32_32_64_row1_col2_128 import (
    Conv3x3_32_32_64Row1Col2_128Stem,
)
from tetris_rl.core.policies.sb3.spatial.stems.conv3x3_32_32_64_row1_col3_128 import (
    Conv3x3_32_32_64Row1Col3_128Stem,
)
from tetris_rl.core.policies.sb3.spatial_heads.attn_pool import AttentionPoolHead
from tetris_rl.core.policies.sb3.spatial_heads.col_collapse import ColumnCollapseHead
from tetris_rl.core.policies.sb3.spatial_heads.flatten import FlattenHead
from tetris_rl.core.policies.sb3.spatial_heads.global_pool import GlobalPoolHead

# ---------------------------------------------------------------------
# Spatial preprocessing
# ---------------------------------------------------------------------

SPATIAL_PREPROCESSOR_REGISTRY: Mapping[str, Any] = {
    "binary": BinarySpatialPreprocessor,
    # "rgb": RGBSpatialPreprocessor,
}

# ---------------------------------------------------------------------
# Spatial stems (spatial -> spatial)
# ---------------------------------------------------------------------

STEM_REGISTRY: Mapping[str, Any] = {
    "cnn": CNNStem,  # generic configurable family
    "conv3x3_32_32_64": Conv3x3_32_32_64Stem,  # fixed, report-exact stem
    "conv1x3_32x4_64_5l": Conv1x3_32x4_64_5LStem,
    "conv3x3_32_32_64_64_128_5l": Conv3x3_32_32_64_64_128_5LStem,
    "conv3x3_32_32_64_row1_col2_128": Conv3x3_32_32_64Row1Col2_128Stem,
    "conv3x3_32_32_64_row1_col3_128": Conv3x3_32_32_64Row1Col3_128Stem,
}


# ---------------------------------------------------------------------
# Token mixers
# ---------------------------------------------------------------------

TOKEN_MIXER_REGISTRY: Mapping[str, Any] = {
    "mlp": MLPMixer,
    "transformer": TransformerMixer,
}

# ---------------------------------------------------------------------
# Spatial heads (spatial -> feature vector)
# ---------------------------------------------------------------------

SPATIAL_HEAD_REGISTRY: Mapping[str, Any] = {
    "global_pool": GlobalPoolHead,       # avg / max / avgmax (+ optional conv stack)
    "flatten": FlattenHead,              # flatten H×W×C only (no MLP)
    "attn_pool": AttentionPoolHead,      # learned attention pooling over spatial grid
    "col_collapse": ColumnCollapseHead,  # column-wise collapse (Tetris inductive bias)
}

# ---------------------------------------------------------------------
# Feature augmenters (feature vector + specials -> feature vector)
# ---------------------------------------------------------------------

FEATURE_AUGMENTER_REGISTRY: Mapping[str, Any] = {
    "onehot_concat": OneHotConcatAugmenter,
    "mlp_joint": JointMLPAugmenter,
    "mlp_split": SplitMLPAugmenter,
}

__all__ = [
    "SPATIAL_PREPROCESSOR_REGISTRY",
    "STEM_REGISTRY",
    "TOKEN_MIXER_REGISTRY",
    "SPATIAL_HEAD_REGISTRY",
    "FEATURE_AUGMENTER_REGISTRY",
]


