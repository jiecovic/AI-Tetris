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
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    return_mean = as_float(metrics.get("episode/return_mean"))
    steps_mean = as_float(metrics.get("episode/steps_mean"))
    reward_per_step = None
    if return_mean is not None and steps_mean is not None and steps_mean > 0:
        reward_per_step = float(return_mean) / float(steps_mean)
    if reward_per_step is None:
        reward_per_step = as_float(metrics.get("episode/return_per_step"))

    lines_per_step = as_float(metrics.get("game/lines_per_step"))

    go_rate = as_float(metrics.get("tf/game_over_rate"))
    survival_like = None if go_rate is None else (1.0 - float(go_rate))

    reward_like = reward_per_step
    return reward_like, lines_per_step, survival_like


__all__ = ["as_float", "pick_best_values", "safe_int"]
