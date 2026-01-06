# src/tetris_rl/config/model/spatial_spec.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

Activation = Literal["gelu", "relu", "silu"]

# ---------------------------------------------------------------------
# Spatial preprocessor (raw obs dict -> SpatialFeatures + Specials)
# ---------------------------------------------------------------------

SpatialPreprocessorType = Literal[
    "binary",
    # "rgb",  # future
]


@dataclass(frozen=True)
class SpatialPreprocessorConfig:
    """
    Spatial preprocessor selection.

    Most preprocessors are param-less at first (e.g. 'binary').
    Keep `params` optional so YAML can omit it or set {}.
    """
    type: SpatialPreprocessorType
    params: Optional[object] = None


# ---------------------------------------------------------------------
# Spatial stem (spatial -> spatial)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class CNNStemParams:
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
]


@dataclass(frozen=True)
class StemConfig:
    """
    Optional spatial stem configuration.

    - type: selects the stem family
    - params:
        * None for preset stems (e.g. conv3x3_32_32_64)
        * CNNStemParams for type='cnn'
    """
    type: StemType
    params: Optional[CNNStemParams] = None

    def __post_init__(self) -> None:
        if self.type == "cnn":
            if self.params is None:
                raise ValueError("StemConfig(type='cnn') requires params")
        else:
            if self.params is not None:
                raise ValueError(f"StemConfig(type='{self.type}') must not have params")



__all__ = [
    "Activation",
    "SpatialPreprocessorType",
    "SpatialPreprocessorConfig",
    "CNNStemParams",
    "StemType",
    "StemConfig",
]
