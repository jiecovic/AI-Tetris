# src/tetris_rl/core/config/root.py
from __future__ import annotations

from tetris_rl.core.config.base import ConfigBase
from tetris_rl.core.datagen.config import (
    DataGenDatasetConfig,
    DataGenExpertConfig,
    DataGenGenerationConfig,
    DataGenRunConfig,
)
from tetris_rl.core.envs.config import EnvConfig
from tetris_rl.core.policies.sb3.config import SB3PolicyConfig
from tetris_rl.core.runs.config import RunConfig
from tetris_rl.core.training.config import AlgoConfig, CheckpointsConfig, EvalConfig, ImitationConfig, LearnConfig


class ExperimentConfig(ConfigBase):
    log_level: str = "info"
    run: RunConfig
    env_train: EnvConfig
    env_eval: EnvConfig
    policy: SB3PolicyConfig
    learn: LearnConfig
    algo: AlgoConfig
    checkpoints: CheckpointsConfig = CheckpointsConfig()
    eval: EvalConfig = EvalConfig()
    imitation: ImitationConfig = ImitationConfig()


class DataGenConfig(ConfigBase):
    log_level: str = "info"
    repo_root: str = ""
    use_rich: bool = True
    env: EnvConfig
    dataset: DataGenDatasetConfig
    run: DataGenRunConfig
    generation: DataGenGenerationConfig
    expert: DataGenExpertConfig


__all__ = ["ExperimentConfig", "DataGenConfig"]

