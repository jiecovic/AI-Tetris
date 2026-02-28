# src/tetris_rl/core/config/root.py
from __future__ import annotations

from pydantic import field_validator, model_validator

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
    PolicySourceConfig,
)


class ExperimentConfig(ConfigBase):
    log_level: str = "info"
    run: RunConfig
    env_train: EnvConfig
    env_eval: EnvConfig
    policy: SB3PolicyConfig | PolicySourceConfig
    learn: LearnConfig
    algo: AlgoConfig
    callbacks: CallbacksConfig = CallbacksConfig()

    @model_validator(mode="before")
    @classmethod
    def _validate_policy_selector_shape(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if "policy_init" in out:
            raise ValueError(
                "policy_init is removed; use `policy: {source, which}` via `/sb3_policies@policy: pretrained`"
            )
        learn_obj = out.get("learn", None)
        if isinstance(learn_obj, dict) and ("policy_init" in learn_obj or "resume" in learn_obj):
            raise ValueError("learn.policy_init/learn.resume are removed; use `policy: {source, which}`")

        policy_obj = out.get("policy", None)
        if isinstance(policy_obj, dict) and "source" in policy_obj:
            allowed = {"source", "which"}
            extra = [k for k in policy_obj.keys() if k not in allowed]
            if extra:
                raise ValueError(
                    "policy selector mixed with sb3 policy fields; "
                    "use either `/sb3_policies@policy:<name>` or `/sb3_policies@policy:pretrained`"
                )
        return out


class ImitationExperimentConfig(ConfigBase):
    log_level: str = "info"
    run: RunConfig
    env_train: EnvConfig
    env_eval: EnvConfig
    policy: SB3PolicyConfig | PolicySourceConfig
    learn: ImitationLearnConfig
    algo: ImitationAlgoConfig = ImitationAlgoConfig()
    callbacks: CallbacksConfig = CallbacksConfig()

    @model_validator(mode="before")
    @classmethod
    def _validate_policy_selector_shape(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if "policy_init" in out:
            raise ValueError(
                "policy_init is removed; use `policy: {source, which}` via `/sb3_policies@policy: pretrained`"
            )
        learn_obj = out.get("learn", None)
        if isinstance(learn_obj, dict) and ("policy_init" in learn_obj or "resume" in learn_obj):
            raise ValueError("learn.policy_init/learn.resume are removed; use `policy: {source, which}`")

        policy_obj = out.get("policy", None)
        if isinstance(policy_obj, dict) and "source" in policy_obj:
            allowed = {"source", "which"}
            extra = [k for k in policy_obj.keys() if k not in allowed]
            if extra:
                raise ValueError(
                    "policy selector mixed with sb3 policy fields; "
                    "use either `/sb3_policies@policy:<name>` or `/sb3_policies@policy:pretrained`"
                )
        return out

    @field_validator("learn")
    @classmethod
    def _validate_imitation_dataset(cls, v: ImitationLearnConfig) -> ImitationLearnConfig:
        dataset_dir = v.dataset_dir
        if not str(dataset_dir).strip():
            raise ValueError("imitation.dataset_dir must be set")
        return v


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
