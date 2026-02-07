# src/tetris_rl/core/training/config.py
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import Field, field_validator

from tetris_rl.core.config.base import ConfigBase

class EvalCheckpointCallbackConfig(ConfigBase):
    """
    Training-time evaluation hook.

    This is a training hook (for TB + best checkpoints), not a benchmarking suite.
    """

    enabled: bool = False
    every: int = Field(default=0, ge=0)
    steps: int = Field(default=100_000, ge=1)

    deterministic: bool = True
    seed_offset: int = 10_000
    num_envs: int = Field(default=1, ge=1)
    workers: int = Field(default=1, ge=1)
    mode: Literal["vectorized", "workers"] = "vectorized"

class LatestCallbackConfig(ConfigBase):
    enabled: bool = True
    every: int = Field(default=200_000, ge=1)


class CallbacksConfig(ConfigBase):
    latest: LatestCallbackConfig = LatestCallbackConfig()
    eval_checkpoint: EvalCheckpointCallbackConfig = EvalCheckpointCallbackConfig()


class ImitationLearnConfig(ConfigBase):
    # offline dataset (required)
    dataset_dir: str

    # training loop
    epochs: int = Field(default=1, ge=1)
    batch_size: int = Field(default=256, ge=1)
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    shuffle: bool = True

    # optional: limit how many samples to use (0 = all)
    max_samples: int = Field(default=0, ge=0)

    # archive (copy of latest.zip)
    save_archive: bool = True
    archive_dir: str = "checkpoints/imitation"

    # optional: initialize from another run/checkpoint
    resume: Optional[str] = None

    @field_validator("resume", mode="before")
    @classmethod
    def _resume_empty_to_none(cls, v: object) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None


class ImitationAlgoParams(ConfigBase):
    # policy backend used for BC (must match PPO policy class)
    policy_backend: Literal["ppo", "maskable_ppo"] = "maskable_ppo"


class AlgoConfig(ConfigBase):
    type: Literal["ppo", "maskable_ppo"] = "ppo"
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("type", mode="before")
    @classmethod
    def _type_lower(cls, v: object) -> str:
        return str(v).strip().lower()


class LearnConfig(ConfigBase):
    total_timesteps: int = Field(default=200_000, ge=0)

    # Resume training from a previous run folder.
    resume: Optional[str] = None

    @field_validator("resume", mode="before")
    @classmethod
    def _resume_empty_to_none(cls, v: object) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None


class ImitationAlgoConfig(ConfigBase):
    type: Literal["imitation"] = "imitation"
    params: ImitationAlgoParams = ImitationAlgoParams()


__all__ = [
    "AlgoConfig",
    "CallbacksConfig",
    "EvalCheckpointCallbackConfig",
    "ImitationAlgoConfig",
    "ImitationLearnConfig",
    "ImitationAlgoParams",
    "LatestCallbackConfig",
    "LearnConfig",
]
