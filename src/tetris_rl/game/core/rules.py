# src/tetris_rl/game/core/rules.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreConfig:
    single: int = 100
    double: int = 300
    triple: int = 500
    tetris: int = 800


def score_for_clears(cleared: int, cfg: ScoreConfig) -> int:
    if cleared == 1:
        return cfg.single
    if cleared == 2:
        return cfg.double
    if cleared == 3:
        return cfg.triple
    if cleared >= 4:
        return cfg.tetris
    return 0
