from __future__ import annotations

from tetris_rl.config.base import ConfigBase
from tetris_rl.datagen.config import (
    DataGenDatasetConfig,
    DataGenExpertConfig,
    DataGenGenerationConfig,
    DataGenRunConfig,
)
from tetris_rl.envs.config import EnvConfig
from tetris_rl.game.config import GameConfig
from tetris_rl.models.config import ModelConfig
from tetris_rl.runs.config import RunConfig
from tetris_rl.training.config import TrainConfig


class ExperimentConfig(ConfigBase):
    log_level: str = "info"
    run: RunConfig
    env: EnvConfig
    game: GameConfig
    model: ModelConfig
    train: TrainConfig


class DataGenConfig(ConfigBase):
    log_level: str = "info"
    repo_root: str = ""
    use_rich: bool = True
    env: EnvConfig
    game: GameConfig
    dataset: DataGenDatasetConfig
    run: DataGenRunConfig
    generation: DataGenGenerationConfig
    expert: DataGenExpertConfig


__all__ = ["ExperimentConfig", "DataGenConfig"]
