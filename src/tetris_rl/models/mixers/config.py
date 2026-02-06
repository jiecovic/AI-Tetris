from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from tetris_rl.config.base import ConfigBase
from tetris_rl.config.typed_params import parse_typed_params

PoolKind = Literal[
    "cls",
    "mean",
    "max",
    "meanmax",
    "flatten",
    "cls_mean",
    "cls_max",
    "cls_meanmax",
]


# ---------------------------------------------------------------------
# MLP Mixer
# ---------------------------------------------------------------------


class MLPMixerParams(ConfigBase):
    features_dim: int = Field(ge=1)
    n_layers: int = 4
    token_mlp_dim: int = 256
    channel_mlp_dim: int = 1024
    dropout: float = 0.0

    use_cls: bool = True
    num_cls_tokens: int = 1
    pool: PoolKind = "cls"

    pre_ln_input: bool = False


# ---------------------------------------------------------------------
# Transformer Mixer
# ---------------------------------------------------------------------


class TransformerMixerParams(ConfigBase):
    features_dim: int = Field(ge=1)

    n_layers: int = 4
    n_heads: int = 8
    mlp_ratio: float = 4.0

    dropout: float = 0.0
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0

    use_cls: bool = True
    num_cls_tokens: int = 1
    pool: PoolKind = "cls"

    pre_ln_input: bool = False


# ---------------------------------------------------------------------
# Top-level mixer section
# ---------------------------------------------------------------------


MixerType = Literal["mlp", "transformer"]


MIXER_PARAMS_REGISTRY = {
    "mlp": MLPMixerParams,
    "transformer": TransformerMixerParams,
}


class MixerConfig(ConfigBase):
    type: MixerType
    params: MLPMixerParams | TransformerMixerParams

    @model_validator(mode="before")
    @classmethod
    def _parse_params(cls, data: object) -> object:
        if isinstance(data, MixerConfig):
            return data
        if not isinstance(data, dict):
            raise TypeError("mixer must be a mapping with keys {type, params}")
        tag, params = parse_typed_params(
            type_value=data.get("type", None),
            params_value=data.get("params", None),
            registry=MIXER_PARAMS_REGISTRY,
            where="mixer",
        )
        return {"type": tag, "params": params}


__all__ = [
    "PoolKind",
    "MLPMixerParams",
    "TransformerMixerParams",
    "MixerType",
    "MixerConfig",
    "MIXER_PARAMS_REGISTRY",
]
