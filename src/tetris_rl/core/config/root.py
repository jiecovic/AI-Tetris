# src/tetris_rl/core/config/root.py
from __future__ import annotations

from pydantic import model_validator

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
from tetris_rl.core.training.config import (
    AlgoConfig,
    CallbacksConfig,
    ImitationAlgoConfig,
    ImitationLearnConfig,
    LearnConfig,
)


class ExperimentConfig(ConfigBase):
    log_level: str = "info"
    run: RunConfig
    env_train: EnvConfig
    env_eval: EnvConfig
    policy: SB3PolicyConfig
    learn: LearnConfig
    algo: AlgoConfig
    callbacks: CallbacksConfig = CallbacksConfig()


class ImitationExperimentConfig(ConfigBase):
    log_level: str = "info"
    run: RunConfig
    env_train: EnvConfig
    env_eval: EnvConfig
    policy: SB3PolicyConfig
    learn: ImitationLearnConfig
    algo: ImitationAlgoConfig = ImitationAlgoConfig()
    callbacks: CallbacksConfig = CallbacksConfig()

    @model_validator(mode="after")
    def _validate_imitation_dataset(self) -> "ImitationExperimentConfig":
        dataset_dir = self.learn.dataset_dir
        if not str(dataset_dir).strip():
            raise ValueError("imitation.dataset_dir must be set")
        return self


class DataGenConfig(ConfigBase):
    log_level: str = "info"
    repo_root: str = ""
    use_rich: bool = True
    env: EnvConfig
    dataset: DataGenDatasetConfig
    run: DataGenRunConfig
    generation: DataGenGenerationConfig
    expert: DataGenExpertConfig


__all__ = ["ExperimentConfig", "ImitationExperimentConfig", "DataGenConfig"]

