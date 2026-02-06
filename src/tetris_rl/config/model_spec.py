from __future__ import annotations

from typing import Any, Dict, Optional, Union, Annotated, Literal

from pydantic import Field

from tetris_rl.config.base import ConfigBase
from tetris_rl.config.model.feature_augmenter_spec import FeatureAugmenterConfig
from tetris_rl.config.model.mixer_spec import MixerConfig
from tetris_rl.config.model.spatial_head_spec import SpatialHeadConfig
from tetris_rl.config.model.spatial_spec import SpatialPreprocessorConfig, StemConfig
from tetris_rl.config.model.tokenizer_spec import TokenizerSpec


class TokenEncoderConfig(ConfigBase):
    type: Literal["token"] = "token"
    tokenizer: TokenizerSpec
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
