# src/planning_rl/ga/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

import gymnasium as gym

from planning_rl.policies import VectorParamPolicy


@dataclass(frozen=True)
class GAStats:
    generation: int
    best_score: float
    mean_score: float
    best_index: int
    best_weights: list[float] | None = None
    eval_best_score: float | None = None


GymEnv: TypeAlias = gym.Env


@runtime_checkable
class GAWorkerFactory(Protocol):
    def build_env(self, *, seed: int, worker_index: int) -> GymEnv:
        ...

    def build_policy(self) -> VectorParamPolicy:
        ...


__all__ = ["GAStats", "GAWorkerFactory", "GymEnv"]
