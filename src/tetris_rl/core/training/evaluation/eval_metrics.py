# src/tetris_rl/core/training/evaluation/eval_metrics.py
from __future__ import annotations

from typing import Any, Optional, Tuple


def as_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def pick_best_values(
    metrics: dict[str, Any],
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    score_per_step = as_float(metrics.get("game/score_per_step"))
    lines_per_step = as_float(metrics.get("game/lines_per_step"))
    level_max = as_float(metrics.get("game/level_max"))

    score_like = as_float(metrics.get("episode/final_score_mean"))
    if score_like is None:
        score_like = as_float(metrics.get("game/score_mean"))

    go_rate = as_float(metrics.get("tf/game_over_rate"))
    survival_like = None if go_rate is None else (1.0 - float(go_rate))

    reward_like = score_per_step
    return reward_like, score_like, lines_per_step, level_max, survival_like


__all__ = ["as_float", "pick_best_values", "safe_int"]
