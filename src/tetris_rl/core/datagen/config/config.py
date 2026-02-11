# src/tetris_rl/core/datagen/config/config.py
from __future__ import annotations

from typing import Optional

from pydantic import Field, field_validator

from tetris_rl.core.config.base import ConfigBase


class DataGenShardsConfig(ConfigBase):
    shard_steps: int = Field(default=50_000, ge=1)
    num_shards: int = Field(default=1, ge=1)


class DataGenDatasetConfig(ConfigBase):
    name: str = "bc_dataset"
    out_root: str = "datasets/bc"
    shards: DataGenShardsConfig = Field(default_factory=DataGenShardsConfig)
    compression: bool = False


class DataGenRunConfig(ConfigBase):
    seed: int = 0
    num_workers: int = Field(default=1, ge=1)
    progress_update_every_k: int = Field(default=2000, ge=1)


class DataGenNoiseConfig(ConfigBase):
    enabled: bool = False
    interleave_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    interleave_max_steps: int = Field(default=1, ge=1)


class DataGenGenerationConfig(ConfigBase):
    episode_max_steps: Optional[int] = None
    noise: DataGenNoiseConfig = Field(default_factory=DataGenNoiseConfig)

    @field_validator("episode_max_steps", mode="before")
    @classmethod
    def _normalize_episode_max_steps(cls, v: object) -> Optional[int]:
        if v is None:
            return None
        if not isinstance(v, (int, float, str, bytes, bytearray)):
            return None
        try:
            n = int(v)
        except (TypeError, ValueError):
            return None
        return None if n <= 0 else n


class DataGenExpertParams(ConfigBase):
    beam_from_depth: int = 0
    beam_width: Optional[int] = None
    tail_weight: float = 0.5

    @field_validator("beam_width", mode="before")
    @classmethod
    def _beam_width_positive(cls, v: object) -> Optional[int]:
        if v is None:
            return None
        if not isinstance(v, (int, float, str, bytes, bytearray)):
            return None
        try:
            n = int(v)
        except (TypeError, ValueError):
            return None
        return None if n <= 0 else n


class DataGenExpertConfig(ConfigBase):
    """
    Example:
      expert:
        type: codemy0|codemy1|codemy2|codemy2fast
        params:
          beam_from_depth: 0
          beam_width: 10
          tail_weight: 0.5
    """

    type: str = "codemy1"
    params: DataGenExpertParams = Field(default_factory=DataGenExpertParams)

    @field_validator("type", mode="before")
    @classmethod
    def _type_lower(cls, v: object) -> str:
        return str(v).strip().lower()

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        if v not in {"codemy0", "codemy1", "codemy2", "codemy2fast"}:
            raise ValueError(f"expert.type must be codemy0|codemy1|codemy2|codemy2fast, got {v!r}")
        return v


__all__ = [
    "DataGenShardsConfig",
    "DataGenDatasetConfig",
    "DataGenRunConfig",
    "DataGenNoiseConfig",
    "DataGenGenerationConfig",
    "DataGenExpertParams",
    "DataGenExpertConfig",
]
