# src/planning_rl/ga/types.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GAStats:
    generation: int
    best_score: float
    mean_score: float
    best_index: int
    best_weights: list[float] | None = None
    eval_best_score: float | None = None


__all__ = ["GAStats"]
