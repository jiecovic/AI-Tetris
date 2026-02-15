# src/tetris_rl/core/policies/sb3/spatial/config.py
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import ValidationInfo, field_validator, model_validator

from tetris_rl.core.config.base import ConfigBase
from tetris_rl.core.policies.sb3.types import LayerActivationName, PoolAvgMaxName, StemName

Activation = LayerActivationName

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


PoolType = PoolAvgMaxName
IntOrPair = int | tuple[int, int]


def _normalize_int_or_pair(value: object, *, field: str, allow_none: bool = False) -> object:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field} must be an int or pair")
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an int or pair (bool is not allowed)")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"{field} pair must have exactly 2 values, got {len(value)}")
        a, b = value
        if isinstance(a, bool) or isinstance(b, bool):
            raise ValueError(f"{field} pair values must be ints (bool is not allowed)")
        return (int(a), int(b))
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1].strip()
        elif s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
        if "," in s:
            parts = [p.strip() for p in s.split(",")]
            if len(parts) != 2:
                raise ValueError(f"{field} pair must have exactly 2 comma-separated values")
            return (int(parts[0]), int(parts[1]))
        return int(s)
    raise ValueError(f"{field} must be an int or pair, got {value!r}")


class CNNStemPoolSpec(ConfigBase):
    """
    Optional pooling op inserted after a conv layer.
    """

    type: PoolType = "avg"
    k: IntOrPair = 2
    s: IntOrPair = 2
    p: IntOrPair = 0

    @field_validator("type", mode="before")
    @classmethod
    def _type_lower(cls, v: object) -> str:
        return str(v).strip().lower()

    @field_validator("k", "s", "p", mode="before")
    @classmethod
    def _normalize_ksp(cls, v: object, info: ValidationInfo) -> object:
        field_name = str(info.field_name or "value")
        return _normalize_int_or_pair(v, field=f"pool.{field_name}")


class CNNStemLayerSpec(ConfigBase):
    """
    Explicit per-layer stem block.
    """

    out: int
    k: IntOrPair
    s: IntOrPair = 1
    p: Optional[IntOrPair] = None
    act: Optional[Activation] = None
    pool: Optional[CNNStemPoolSpec] = None

    @field_validator("act", mode="before")
    @classmethod
    def _act_lower(cls, v: object) -> object:
        if v is None:
            return None
        return str(v).strip().lower()

    @field_validator("k", "s", mode="before")
    @classmethod
    def _normalize_ks(cls, v: object, info: ValidationInfo) -> object:
        field_name = str(info.field_name or "value")
        return _normalize_int_or_pair(v, field=field_name)

    @field_validator("p", mode="before")
    @classmethod
    def _normalize_p(cls, v: object, info: ValidationInfo) -> object:
        field_name = str(info.field_name or "value")
        return _normalize_int_or_pair(v, field=field_name, allow_none=True)


class CNNStemParams(ConfigBase):
    """
    Generic configurable CNN stem (spatial -> spatial).

    Supports two equivalent styles:

    1) Legacy tuple style:
       - channels / kernel_sizes / (optional) strides
       - global activation / batchnorm / dropout

    2) Explicit layer list style:
       - layers: [{out, k, s, p, act, pool}, ...]
       - act is required per-layer.
       - global use_batchnorm / dropout apply to all conv layers.
    """

    # legacy tuple style
    channels: Optional[tuple[int, ...]] = None
    kernel_sizes: Optional[tuple[int, ...]] = None
    strides: Optional[tuple[int, ...]] = None

    # explicit layer list style
    layers: Optional[tuple[CNNStemLayerSpec, ...]] = None

    # legacy mode only (when using channels/kernel_sizes); no default
    activation: Optional[Activation] = None
    use_batchnorm: bool = False
    dropout: float = 0.0

    @field_validator("activation", mode="before")
    @classmethod
    def _activation_lower(cls, v: object) -> object:
        if v is None:
            return None
        return str(v).strip().lower()

    @model_validator(mode="before")
    @classmethod
    def _validate_layout(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data

        has_layers = data.get("layers") is not None
        has_legacy = any(data.get(k) is not None for k in ("channels", "kernel_sizes", "strides"))

        if has_layers and has_legacy:
            raise ValueError("cnn stem params: use either {layers: ...} OR legacy {channels/kernel_sizes/strides}")

        if has_layers:
            layers = data.get("layers")
            if not isinstance(layers, (list, tuple)) or len(layers) == 0:
                raise ValueError("cnn stem params: layers must be non-empty when provided")
            if data.get("activation") is not None:
                raise ValueError("cnn stem params: activation is legacy-only; set per-layer act when using layers")
            for i, layer in enumerate(layers):
                act = layer.get("act") if isinstance(layer, dict) else getattr(layer, "act", None)
                if act is None:
                    raise ValueError(f"cnn stem params: layers[{i}].act is required when using layers")
        else:
            channels = data.get("channels")
            kernels = data.get("kernel_sizes")
            strides = data.get("strides")

            if channels is None or kernels is None:
                raise ValueError("cnn stem params: legacy mode requires both channels and kernel_sizes")

            if not isinstance(channels, (list, tuple)) or len(channels) == 0:
                raise ValueError("cnn stem params: channels must be non-empty")
            if not isinstance(kernels, (list, tuple)):
                raise ValueError("cnn stem params: kernel_sizes must be provided as a sequence")
            if len(channels) != len(kernels):
                raise ValueError("cnn stem params: channels and kernel_sizes must have same length")
            if strides is not None:
                if not isinstance(strides, (list, tuple)):
                    raise ValueError("cnn stem params: strides must be a sequence when provided")
                if len(strides) != len(channels):
                    raise ValueError("cnn stem params: strides must have same length as channels")
            if data.get("activation") is None:
                raise ValueError("cnn stem params: legacy mode requires activation")

        if float(data.get("dropout", 0.0)) < 0.0:
            raise ValueError("cnn stem params: dropout must be >= 0")
        return data


StemType = StemName


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
    "PoolType",
    "CNNStemPoolSpec",
    "CNNStemLayerSpec",
    "SpatialPreprocessorType",
    "SpatialPreprocessorConfig",
    "CNNStemParams",
    "StemType",
    "StemConfig",
]

