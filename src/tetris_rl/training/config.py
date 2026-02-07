from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import Field, field_validator, model_validator

from tetris_rl.config.base import ConfigBase

TrainEvalMode = Literal["off", "rl", "imitation", "both"]


class TrainCheckpointsConfig(ConfigBase):
    """
    Checkpoint cadence for training.

    Semantics:
      - latest_every: save checkpoints/latest.zip every N environment steps.
    """

    latest_every: int = Field(default=200_000, ge=1)


class TrainEvalConfig(ConfigBase):
    """
    Training-time evaluation semantics.

    This is a training hook (for TB + best checkpoints), not a benchmarking suite.
    """

    mode: TrainEvalMode = "off"
    steps: int = Field(default=100_000, ge=1)
    eval_every: int = Field(default=0, ge=0)

    deterministic: bool = True
    seed_offset: int = 10_000
    num_envs: int = Field(default=1, ge=1)

    @field_validator("mode", mode="before")
    @classmethod
    def _mode_lower(cls, v: object) -> str:
        return str(v).strip().lower()


class ImitationConfig(ConfigBase):
    enabled: bool = False

    # offline dataset (required when enabled)
    dataset_dir: str = ""

    # training
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

    @model_validator(mode="after")
    def _validate_dataset_dir(self) -> "ImitationConfig":
        if self.enabled and not str(self.dataset_dir).strip():
            raise ValueError("train.imitation.enabled=true requires train.imitation.dataset_dir to be set")
        return self


class RLAlgoConfig(ConfigBase):
    type: Literal["ppo", "maskable_ppo", "dqn"] = "ppo"
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("type", mode="before")
    @classmethod
    def _type_lower(cls, v: object) -> str:
        return str(v).strip().lower()


class RLConfig(ConfigBase):
    enabled: bool = True
    total_timesteps: int = Field(default=200_000, ge=0)

    # Resume training from a previous run folder.
    resume: Optional[str] = None

    algo: RLAlgoConfig = Field(default_factory=RLAlgoConfig)

    @field_validator("resume", mode="before")
    @classmethod
    def _resume_empty_to_none(cls, v: object) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None


class TrainConfig(ConfigBase):
    checkpoints: TrainCheckpointsConfig = Field(default_factory=TrainCheckpointsConfig)
    eval: TrainEvalConfig = Field(default_factory=TrainEvalConfig)
    imitation: ImitationConfig = Field(default_factory=ImitationConfig)
    rl: RLConfig = Field(default_factory=RLConfig)


__all__ = [
    "TrainConfig",
    "TrainCheckpointsConfig",
    "TrainEvalConfig",
    "ImitationConfig",
    "RLConfig",
    "RLAlgoConfig",
    "TrainEvalMode",
]
