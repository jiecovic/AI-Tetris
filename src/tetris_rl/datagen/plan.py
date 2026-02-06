from __future__ import annotations

from pydantic import Field

from tetris_rl.config.base import ConfigBase
from tetris_rl.datagen.config import (
    DataGenDatasetConfig,
    DataGenRunConfig,
    DataGenGenerationConfig,
    DataGenExpertConfig,
)


class DataGenPlan(ConfigBase):
    dataset: DataGenDatasetConfig = Field(default_factory=DataGenDatasetConfig)
    run: DataGenRunConfig = Field(default_factory=DataGenRunConfig)
    generation: DataGenGenerationConfig = Field(default_factory=DataGenGenerationConfig)
    expert: DataGenExpertConfig = Field(default_factory=DataGenExpertConfig)


__all__ = ["DataGenPlan"]
