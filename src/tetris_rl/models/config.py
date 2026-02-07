# src/tetris_rl/models/config.py
from __future__ import annotations

from typing import Any, Dict, Optional, Union, Annotated, Literal

from pydantic import Field

from tetris_rl.config.base import ConfigBase
from tetris_rl.models.feature_augmenters.config import FeatureAugmenterConfig
from tetris_rl.models.mixers.config import MixerConfig
from tetris_rl.models.spatial_heads.config import SpatialHeadConfig
from tetris_rl.models.spatial.config import SpatialPreprocessorConfig, StemConfig
from tetris_rl.models.tokenizers.config import TokenizerConfig


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


class ModelConfig(ConfigBase):
    policy_kwargs: Dict[str, Any] = Field(default_factory=dict)
    net_arch: Optional[Any] = None
    activation_fn: Optional[str] = None
    feature_extractor: FeatureExtractorConfig


__all__ = [
    "TokenEncoderConfig",
    "SpatialEncoderConfig",
    "EncoderConfig",
    "FeatureExtractorConfig",
    "ModelConfig",
]
