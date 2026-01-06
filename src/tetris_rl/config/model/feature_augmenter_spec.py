# src/tetris_rl/config/model/feature_augmenter_spec.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Type, Any, Literal, Optional

FeatureAugmenterType = Literal["none", "onehot_concat", "mlp_joint", "mlp_split"]


# -----------------------------
# PARAMS (NO n_kinds here)
# -----------------------------

@dataclass(frozen=True)
class FeatureAugmenterBaseParams:
    """
    features_dim:
      Final output dim expected by SB3 (the FeatureExtractor's features_dim).

      Pattern B: make this OPTIONAL and inject it from the encoder-derived dim
      inside TetrisFeatureExtractor.__init__.
    """
    features_dim: int | None = None


@dataclass(frozen=True)
class OneHotConcatParams(FeatureAugmenterBaseParams):
    """
    Concatenate one-hot encoded specials to the feature vector,
    then optional projection to features_dim.
    """
    use_active: bool = True
    use_next: bool = False


@dataclass(frozen=True)
class JointMLPParams(FeatureAugmenterBaseParams):
    """
    One-hot specials -> (joint) MLP -> z, concat to features,
    then optional projection to features_dim.
    """
    use_active: bool = True
    use_next: bool = False

    out_dim: int = 64
    hidden_dims: tuple[int, ...] = (64,)
    activation: Literal["gelu", "relu", "silu"] = "gelu"
    dropout: float = 0.0


@dataclass(frozen=True)
class SplitMLPParams(FeatureAugmenterBaseParams):
    """
    Active and next are embedded separately (two MLPs), then concatenated:
      z = [mlp_active(onehot_active), mlp_next(onehot_next)]
    Then concat to features and optional projection to features_dim.
    """
    use_active: bool = True
    use_next: bool = False

    out_dim_total: int = 64
    out_dim_active: int | None = None
    out_dim_next: int | None = None

    hidden_dims: tuple[int, ...] = (64,)
    activation: Literal["gelu", "relu", "silu"] = "gelu"
    dropout: float = 0.0


# -----------------------------
# WRAPPER
# -----------------------------

@dataclass(frozen=True)
class FeatureAugmenterConfig:
    """
    If type="none", params MUST be None and the pipeline must skip augmentation.
    """
    type: FeatureAugmenterType
    params: Optional[OneHotConcatParams | JointMLPParams | SplitMLPParams] = None


FEATURE_AUGMENTER_PARAMS_REGISTRY: Mapping[str, Type[Any]] = {
    "onehot_concat": OneHotConcatParams,
    "mlp_joint": JointMLPParams,
    "mlp_split": SplitMLPParams,
}

__all__ = [
    "FeatureAugmenterType",
    "FeatureAugmenterBaseParams",
    "OneHotConcatParams",
    "JointMLPParams",
    "SplitMLPParams",
    "FeatureAugmenterConfig",
    "FEATURE_AUGMENTER_PARAMS_REGISTRY",
]
