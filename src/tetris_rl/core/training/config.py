# src/tetris_rl/core/training/config.py
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import Field, field_validator, model_validator

from tetris_rl.core.config.base import ConfigBase


class EvalCheckpointCallbackConfig(ConfigBase):
    """
    Training-time evaluation hook.

    This is a training hook (for TB + best checkpoints), not a benchmarking suite.
    """

    enabled: bool = False
    every: int = Field(default=0, ge=0)
    episodes: int = Field(default=100, ge=1)
    min_steps: int = Field(default=0, ge=0)
    max_steps_per_episode: Optional[int] = Field(default=None, ge=1)
    steps: Optional[int] = None

    deterministic: bool = True
    seed_offset: int = 10_000
    n_envs: int = Field(default=1, ge=1)
    # Back-compat alias for older run configs persisted to disk.
    # Prefer n_envs everywhere; this field exists only so Pydantic won't reject old configs.
    num_envs: Optional[int] = Field(default=None, ge=1)
    workers: int = Field(default=1, ge=1)
    mode: Literal["vectorized", "workers"] = "vectorized"

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_fields(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data

        out = dict(data)

        # Back-compat: steps -> min_steps
        if "min_steps" not in out and "steps" in out:
            out["min_steps"] = out.get("steps")

        # Naming consistency: num_envs -> n_envs
        if "num_envs" in out:
            if "n_envs" not in out:
                out["n_envs"] = out.get("num_envs")
            else:
                try:
                    n = out.get("n_envs")
                    legacy = out.get("num_envs")
                    if n is None or legacy is None:
                        raise ValueError
                    if int(n) != int(legacy):
                        raise ValueError("eval_checkpoint.num_envs and eval_checkpoint.n_envs disagree; use n_envs only")
                except Exception:
                    raise ValueError("eval_checkpoint.num_envs and eval_checkpoint.n_envs disagree; use n_envs only")

        return out

class LatestCallbackConfig(ConfigBase):
    enabled: bool = True
    every: int = Field(default=200_000, ge=1)


class CallbacksConfig(ConfigBase):
    latest: LatestCallbackConfig = LatestCallbackConfig()
    eval_checkpoint: EvalCheckpointCallbackConfig = EvalCheckpointCallbackConfig()


class PolicySourceConfig(ConfigBase):
    """
    Policy source selector.

    source supports:
      - run dir path (loads policy config + checkpoint by `which`)
      - run config path (.yaml) (loads policy config only)
      - policy config path (.yaml) (loads policy config only)
    """

    source: str
    which: Literal["latest", "reward", "lines", "survival", "final"] = "latest"

    @field_validator("source", mode="before")
    @classmethod
    def _source_nonempty(cls, v: object) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("policy.source must be a non-empty path")
        return s

    @field_validator("which", mode="before")
    @classmethod
    def _which_lower(cls, v: object) -> str:
        return str(v).strip().lower()


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
    max_steps_per_episode: Optional[int] = Field(default=None, ge=1)


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
    "PolicySourceConfig",
    "LearnConfig",
]
