# src/planning_rl/td/config.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TDConfig:
    total_timesteps: int = 200_000
    rollout_steps: int = 128
    n_envs: int = 1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    grad_clip: float = 0.0
    clip_range_vf: float = 0.0
    batch_size: int = 256
    n_epochs: int = 4
    weight_init_std: float = 0.01
    stats_window: int = 100
    seed: int = 12345
    max_steps_per_episode: int | None = None


__all__ = ["TDConfig"]
