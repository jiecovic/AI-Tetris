from __future__ import annotations

from tetris_rl.config.base import ConfigBase
from tetris_rl.config.datagen_spec import (
    DataGenDatasetSpec,
    DataGenExpertSpec,
    DataGenGenerationSpec,
    DataGenRunSpec,
)
from tetris_rl.config.env_spec import EnvConfig
from tetris_rl.config.game_spec import GameSpec
from tetris_rl.config.model_spec import ModelConfig
from tetris_rl.config.run_spec import RunSpec
from tetris_rl.config.train_spec import TrainSpec


class TrainConfig(ConfigBase):
    log_level: str = "info"
    run: RunSpec
    env: EnvConfig
    game: GameSpec
    model: ModelConfig
    train: TrainSpec


class DataGenConfig(ConfigBase):
    log_level: str = "info"
    repo_root: str = ""
    use_rich: bool = True
    env: EnvConfig
    game: GameSpec
    dataset: DataGenDatasetSpec
    run: DataGenRunSpec
    generation: DataGenGenerationSpec
    expert: DataGenExpertSpec


__all__ = ["TrainConfig", "DataGenConfig"]
