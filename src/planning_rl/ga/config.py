# src/planning_rl/ga/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 1000
    selection: Literal["elite_pool", "tournament"] = "tournament"
    elite_frac: float = 0.1
    parent_pool_frac: float | None = None
    offspring_frac: float = 0.3
    tournament_frac: float = 0.1
    tournament_winners: int = 2
    crossover_kind: Literal["uniform_mask", "weighted_avg"] = "weighted_avg"
    crossover_rate: float = 1.0
    mutation_kind: Literal["gaussian", "single_component"] = "single_component"
    mutation_rate: float = 0.05
    mutation_sigma: float = 0.1
    mutation_delta: float = 0.2
    weight_init_range: tuple[float, float] = (-1.0, 1.0)
    weight_clip: tuple[float, float] | None = None
    normalize_weights: bool = True
    seed: int = 12345

    def parent_pool(self) -> float:
        return self.parent_pool_frac if self.parent_pool_frac is not None else self.elite_frac


@dataclass(frozen=True)
class GAFitnessConfig:
    episodes: int = 100
    max_steps: int = 500
    fitness_metric: Literal["total_reward", "reward_per_step"] = "total_reward"
    seed: int = 12345


__all__ = ["GAConfig", "GAFitnessConfig"]
