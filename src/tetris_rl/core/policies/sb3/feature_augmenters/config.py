# src/tetris_rl/core/policies/sb3/feature_augmenters/config.py
from __future__ import annotations

from typing import Optional

from pydantic import model_validator

from tetris_rl.core.config.base import ConfigBase
from tetris_rl.core.config.typed_params import parse_typed_params
from tetris_rl.core.policies.sb3.types import FeatureAugmenterName, LayerActivationName

FeatureAugmenterType = FeatureAugmenterName


# -----------------------------
# PARAMS (NO n_kinds here)
# -----------------------------


class FeatureAugmenterBaseParams(ConfigBase):
    """
    features_dim:
      Final output dim expected by SB3 (the FeatureExtractor's features_dim).

      Pattern B: make this OPTIONAL and inject it from the encoder-derived dim
      inside TetrisFeatureExtractor.__init__.
    """

    features_dim: int | None = None


class OneHotConcatParams(FeatureAugmenterBaseParams):
    """
    Concatenate one-hot encoded specials to the feature vector,
    then optional projection to features_dim.
    """

    use_active: bool = True
    use_next: bool = False


class JointMLPParams(FeatureAugmenterBaseParams):
    """
    One-hot specials -> (joint) MLP -> z, concat to features,
    then optional projection to features_dim.
    """

    use_active: bool = True
    use_next: bool = False

    out_dim: int = 64
    hidden_dims: tuple[int, ...] = (64,)
    activation: LayerActivationName = "gelu"
    dropout: float = 0.0


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
    activation: LayerActivationName = "gelu"
    dropout: float = 0.0


FEATURE_AUGMENTER_PARAMS_REGISTRY = {
    "onehot_concat": OneHotConcatParams,
    "mlp_joint": JointMLPParams,
    "mlp_split": SplitMLPParams,
}


# -----------------------------
# WRAPPER
# -----------------------------


class FeatureAugmenterConfig(ConfigBase):
    """
    If type="none", params MUST be None and the pipeline must skip augmentation.
    """

    type: FeatureAugmenterType
    params: Optional[OneHotConcatParams | JointMLPParams | SplitMLPParams] = None

    @model_validator(mode="before")
    @classmethod
    def _parse_params(cls, data: object) -> object:
        if isinstance(data, FeatureAugmenterConfig):
            return data
        if not isinstance(data, dict):
            raise TypeError("feature_augmenter must be a mapping with keys {type, params}")

        t_raw = str(data.get("type", "")).strip().lower()
        if t_raw in {"none", "null", ""}:
            if data.get("params", None) not in (None, {}):
                raise ValueError("feature_augmenter disabled but params were provided")
            return {"type": "none", "params": None}

        tag, params = parse_typed_params(
            type_value=t_raw,
            params_value=data.get("params", None),
            registry=FEATURE_AUGMENTER_PARAMS_REGISTRY,
            where="feature_augmenter",
        )
        return {"type": tag, "params": params}


__all__ = [
    "FeatureAugmenterType",
    "FeatureAugmenterBaseParams",
    "OneHotConcatParams",
    "JointMLPParams",
    "SplitMLPParams",
    "FeatureAugmenterConfig",
    "FEATURE_AUGMENTER_PARAMS_REGISTRY",
]
