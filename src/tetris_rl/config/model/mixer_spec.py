# src/tetris_rl/config/model/mixer_spec.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Type, Any, Literal

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


@dataclass(frozen=True)
class MLPMixerParams:
    features_dim: int
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


@dataclass(frozen=True)
class TransformerMixerParams:
    features_dim: int

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


MixerType = Literal["mlp_mixer", "transformer"]


@dataclass(frozen=True)
class MixerConfig:
    type: MixerType
    params: MLPMixerParams | TransformerMixerParams

MIXER_PARAMS_REGISTRY: Mapping[str, Type[Any]] = {
    "mlp_mixer": MLPMixerParams,
    "transformer": TransformerMixerParams,
}


__all__ = [
    "PoolKind",
    "MLPMixerParams",
    "TransformerMixerParams",
    "MixerType",
    "MixerConfig",
]
