# src/tetris_rl/core/policies/sb3/config.py
from __future__ import annotations

from typing import Annotated, Any, Dict, Literal, Optional, Union

from pydantic import Field, model_validator

from tetris_rl.core.config.base import ConfigBase
from tetris_rl.core.policies.sb3.feature_augmenters.config import FeatureAugmenterConfig
from tetris_rl.core.policies.sb3.mixers.config import MixerConfig
from tetris_rl.core.policies.sb3.spatial.config import SpatialPreprocessorConfig, StemConfig
from tetris_rl.core.policies.sb3.spatial_heads.config import SpatialHeadConfig
from tetris_rl.core.policies.sb3.tokenizers.config import TokenizerConfig
from tetris_rl.core.policies.sb3.types import PolicyActivationName


class TokenEncoderConfig(ConfigBase):
    type: Literal["token"] = "token"
    tokenizer: TokenizerConfig
    mixer: MixerConfig


class SpatialEncoderConfig(ConfigBase):
    type: Literal["spatial"] = "spatial"
    spatial_head: SpatialHeadConfig


EncoderConfig = Annotated[
    Union[TokenEncoderConfig, SpatialEncoderConfig],
    Field(discriminator="type"),
]


class FeatureExtractorConfig(ConfigBase):
    spatial_preprocessor: SpatialPreprocessorConfig
    stem: Optional[StemConfig] = None
    encoder: EncoderConfig
    feature_augmenter: Optional[FeatureAugmenterConfig] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_optional_stem(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data

        out = dict(data)
        stem = out.get("stem", None)
        if stem is None:
            return out

        null_tags = {"", "none", "null", "off", "disabled"}

        if isinstance(stem, str):
            if str(stem).strip().lower() in null_tags:
                out["stem"] = None
            return out

        if isinstance(stem, dict):
            t = stem.get("type", None)
            if t is None:
                out["stem"] = None
                return out
            if str(t).strip().lower() in null_tags:
                out["stem"] = None
                return out

        return out


class SB3PolicyConfig(ConfigBase):
    policy_kwargs: Dict[str, Any] = Field(default_factory=dict)
    net_arch: Optional[Any] = None
    activation_fn: Optional[PolicyActivationName] = None
    feature_extractor: FeatureExtractorConfig


__all__ = [
    "TokenEncoderConfig",
    "SpatialEncoderConfig",
    "EncoderConfig",
    "FeatureExtractorConfig",
    "SB3PolicyConfig",
]
