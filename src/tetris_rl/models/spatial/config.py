# src/tetris_rl/models/spatial/config.py
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import field_validator, model_validator

from tetris_rl.config.base import ConfigBase

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

    @model_validator(mode="after")
    def _validate_params(self) -> "StemConfig":
        if self.type == "cnn":
            if self.params is None:
                raise ValueError("stem.type='cnn' requires params")
        else:
            if self.params is not None:
                raise ValueError(f"stem.type='{self.type}' must not have params")
        return self


__all__ = [
    "Activation",
    "SpatialPreprocessorType",
    "SpatialPreprocessorConfig",
    "CNNStemParams",
    "StemType",
    "StemConfig",
]
