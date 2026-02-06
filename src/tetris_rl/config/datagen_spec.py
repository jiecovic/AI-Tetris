from __future__ import annotations

from typing import Optional

from pydantic import Field, field_validator, model_validator

from tetris_rl.config.base import ConfigBase


class DataGenShardsSpec(ConfigBase):
    shard_steps: int = Field(default=50_000, ge=1)
    num_shards: int = Field(default=1, ge=1)


class DataGenDatasetSpec(ConfigBase):
    name: str = "bc_dataset"
    out_root: str = "datasets/bc"
    shards: DataGenShardsSpec = Field(default_factory=DataGenShardsSpec)
    compression: bool = False


class DataGenRunSpec(ConfigBase):
    seed: int = 0
    num_workers: int = Field(default=1, ge=1)
    progress_update_every_k: int = Field(default=2000, ge=1)


class DataGenNoiseSpec(ConfigBase):
    enabled: bool = False
    interleave_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    interleave_max_steps: int = Field(default=1, ge=1)


class DataGenGenerationSpec(ConfigBase):
    episode_max_steps: Optional[int] = None
    noise: DataGenNoiseSpec = Field(default_factory=DataGenNoiseSpec)

    @field_validator("episode_max_steps", mode="before")
    @classmethod
    def _normalize_episode_max_steps(cls, v: object) -> Optional[int]:
        if v is None:
            return None
        try:
            n = int(v)
        except Exception:
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
        n = int(v)
        return None if n <= 0 else n


class DataGenExpertSpec(ConfigBase):
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

    @model_validator(mode="after")
    def _validate_type(self) -> "DataGenExpertSpec":
        if self.type not in {"codemy0", "codemy1", "codemy2", "codemy2fast"}:
            raise ValueError(f"expert.type must be codemy0|codemy1|codemy2|codemy2fast, got {self.type!r}")
        return self


class DataGenSpec(ConfigBase):
    dataset: DataGenDatasetSpec = Field(default_factory=DataGenDatasetSpec)
    run: DataGenRunSpec = Field(default_factory=DataGenRunSpec)
    generation: DataGenGenerationSpec = Field(default_factory=DataGenGenerationSpec)
    expert: DataGenExpertSpec = Field(default_factory=DataGenExpertSpec)


__all__ = ["DataGenSpec"]

