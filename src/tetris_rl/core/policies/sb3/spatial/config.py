# src/tetris_rl/core/policies/sb3/spatial/config.py
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import ValidationInfo, field_validator

from tetris_rl.core.config.base import ConfigBase

Activation = Literal["gelu", "relu", "silu"]

# ---------------------------------------------------------------------
# Spatial preprocessor (raw obs dict -> SpatialFeatures + Specials)
# ---------------------------------------------------------------------

SpatialPreprocessorType = Literal[
    "binary",
    # "rgb",  # future
]


class SpatialPreprocessorConfig(ConfigBase):
    """
    Spatial preprocessor selection.

    Most preprocessors are param-less at first (e.g. 'binary').
    Keep `params` optional so YAML can omit it or set {}.
    """

    type: SpatialPreprocessorType
    params: Optional[Dict[str, Any]] = None

    @field_validator("type", mode="before")
    @classmethod
    def _type_lower(cls, v: object) -> str:
        return str(v).strip().lower()


# ---------------------------------------------------------------------
# Spatial stem (spatial -> spatial)
# ---------------------------------------------------------------------


class CNNStemParams(ConfigBase):
    """
    Generic configurable CNN stem (spatial -> spatial).

    All per-layer tuples must have the same length.
    """

    channels: tuple[int, ...]
    kernel_sizes: tuple[int, ...]
    strides: Optional[tuple[int, ...]] = None

    activation: Activation = "gelu"
    use_batchnorm: bool = False
    dropout: float = 0.0


StemType = Literal[
    "cnn",
    "conv3x3_32_32_64",
    "conv1x3_32x4_64_5l",
    "conv3x3_32_32_64_64_128_5l",
    "conv3x3_32_32_64_row1_col2_128",
    "conv3x3_32_32_64_row1_col3_128",
]


class StemConfig(ConfigBase):
    """
    Optional spatial stem configuration.

    - type: selects the stem family
    - params:
        * None for preset stems (e.g. conv3x3_32_32_64)
        * CNNStemParams for type='cnn'
    """

    type: StemType
    params: Optional[CNNStemParams] = None

    @field_validator("type", mode="before")
    @classmethod
    def _type_lower(cls, v: object) -> str:
        return str(v).strip().lower()

    @field_validator("params", mode="after")
    @classmethod
    def _validate_params(cls, v: Optional[CNNStemParams], info: ValidationInfo) -> Optional[CNNStemParams]:
        stem_type = info.data.get("type")
        if stem_type == "cnn":
            if v is None:
                raise ValueError("stem.type='cnn' requires params")
        else:
            if v is not None:
                raise ValueError(f"stem.type='{stem_type}' must not have params")
        return v


__all__ = [
    "Activation",
    "SpatialPreprocessorType",
    "SpatialPreprocessorConfig",
    "CNNStemParams",
    "StemType",
    "StemConfig",
]

