# src/planning_rl/ga/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 100
    elite_frac: float = 0.1
    parent_pool_frac: float | None = None
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    mutation_sigma: float = 0.1
    weight_init_range: tuple[float, float] = (-1.0, 1.0)
    weight_clip: tuple[float, float] | None = None
    seed: int = 12345

    def parent_pool(self) -> float:
        return self.parent_pool_frac if self.parent_pool_frac is not None else self.elite_frac


@dataclass(frozen=True)
class GAEvalConfig:
    episodes: int = 100
    max_steps: int = 500
    return_metric: Literal["total_reward", "reward_per_step"] = "total_reward"
    seed: int = 12345


__all__ = ["GAConfig", "GAEvalConfig"]
